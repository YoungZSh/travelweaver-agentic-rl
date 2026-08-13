"""Signal-aware veRL replay buffer for TravelWeaver GRPO groups."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import transfer_queue as tq
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


def classify_reward_group(
    reward_infos: list[dict[str, Any]], *, group_size: int, tolerance: float
) -> tuple[str, float | None]:
    """Classify a complete GRPO group from the shared online Reward fields."""

    if len(reward_infos) != group_size:
        return "invalid", None
    if any(
        "travelweaver_reward" not in info
        or "reward_valid" not in info
        or not bool(float(info["reward_valid"]))
        for info in reward_infos
    ):
        return "invalid", None
    rewards = [float(info["travelweaver_reward"]) for info in reward_infos]
    if max(rewards) - min(rewards) <= tolerance:
        return "zero_variance", rewards[0]
    return "usable", None


class ConsecutiveNoSignalStop(RuntimeError):
    travelweaver_no_signal_stop = True

    def __init__(self, report: dict[str, Any]):
        super().__init__("TravelWeaver stopped after consecutive zero-variance groups.")
        self.report = report


class TravelWeaverReplayBuffer(ReplayBuffer):
    """Filter every zero-variance group and stop after a configurable dry streak."""

    def __init__(self, *args: Any, **kwargs: Any):
        sampler_kwargs = kwargs["sampler_kwargs"]
        self.group_size = int(sampler_kwargs.get("group_size", 8))
        self.zero_variance_tolerance = float(
            sampler_kwargs.get("zero_variance_tolerance", 1e-8)
        )
        self.max_consecutive_no_signal = int(
            sampler_kwargs.get("max_consecutive_no_signal", 10)
        )
        if self.group_size <= 1:
            raise ValueError("TravelWeaver GRPO group_size must be greater than one.")
        if self.zero_variance_tolerance < 0:
            raise ValueError("zero_variance_tolerance must be non-negative.")
        if self.max_consecutive_no_signal <= 0:
            raise ValueError("max_consecutive_no_signal must be positive.")
        self.state_path = Path(
            str(sampler_kwargs.get("state_path", "training/outputs/grpo-sampler-state.json"))
        )
        self._classification: dict[str, dict[str, tuple[str, float | None]]] = defaultdict(dict)
        self.consecutive_no_signal = 0
        self.no_signal_history: list[dict[str, Any]] = []
        self._load_state()
        super().__init__(
            *args,
            **kwargs,
            filter_groups_metric="travelweaver_reward",
            train_batch_size=int(sampler_kwargs.get("train_batch_size", 1)),
            gen_batch_size=1,
            max_inflight_gen_batches=1,
            sync_refill_failed_groups=True,
        )

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        if partition_id == "val":
            return set(), Counter()
        finished_uids = self.finished_keys[partition_id]
        cache = self._classification[partition_id]
        for uid in set(cache) - finished_uids:
            del cache[uid]
        new_uids = finished_uids - cache.keys()
        trajectory_keys = [
            key for key in self.partitions[partition_id] if key.split("_")[0] in new_uids
        ]
        metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields_list = list(data["extra_fields"])
        else:
            extra_fields_list = []
        for key, extra_fields in zip(trajectory_keys, extra_fields_list, strict=True):
            uid = key.split("_")[0]
            extra_fields = getattr(extra_fields, "data", extra_fields)
            reward_info = (
                extra_fields.get("reward_extra_info", {})
                if isinstance(extra_fields, dict)
                else {}
            )
            metrics[uid].append(dict(reward_info))
        ordered = sorted(
            new_uids,
            key=lambda uid: (self.prompt_global_steps[partition_id].get(uid, -1), uid),
        )
        for uid in ordered:
            reward_infos = metrics.get(uid, [])
            classification, shared = classify_reward_group(
                reward_infos,
                group_size=self.group_size,
                tolerance=self.zero_variance_tolerance,
            )
            cache[uid] = (classification, shared)
            self._record_signal_result(
                uid,
                classification,
                shared,
                reward_infos,
                self.prompt_global_steps[partition_id].get(uid),
            )
        filtered = {
            uid
            for uid, (classification, _) in cache.items()
            if classification in {"zero_variance", "invalid"}
        }
        counts = Counter(
            shared
            for classification, shared in cache.values()
            if classification == "zero_variance" and shared is not None
        )
        return filtered, counts

    def _record_signal_result(
        self,
        uid: str,
        classification: str,
        shared_reward: float | None,
        reward_infos: list[dict[str, Any]],
        global_step: int | None,
    ) -> None:
        if classification == "invalid":
            return
        if classification == "usable":
            self.consecutive_no_signal = 0
            self.no_signal_history.clear()
            self._save_state()
            return
        if classification != "zero_variance" or shared_reward is None:
            raise ValueError(f"Unknown TravelWeaver group classification: {classification}")

        dimension_keys = ("artifact_score", "validity_score", "goal_score")
        dimension_means = {
            key: sum(float(info.get(key, 0.0)) for info in reward_infos) / len(reward_infos)
            for key in dimension_keys
        }
        self.consecutive_no_signal += 1
        self.no_signal_history.append(
            {
                "prompt_uid": uid,
                "shared_reward": shared_reward,
                "global_step": global_step,
                "dimension_means": dimension_means,
            }
        )
        self.no_signal_history = self.no_signal_history[-self.max_consecutive_no_signal :]
        self._save_state()
        if self.consecutive_no_signal >= self.max_consecutive_no_signal:
            raise ConsecutiveNoSignalStop(self.stop_report())

    def stop_report(self) -> dict[str, Any]:
        return {
            "format_version": "travelweaver-grpo-stop-v1",
            "reason": "consecutive_zero_variance_groups",
            "threshold": self.max_consecutive_no_signal,
            "consecutive_no_signal": self.consecutive_no_signal,
            "group_size": self.group_size,
            "zero_variance_tolerance": self.zero_variance_tolerance,
            "groups": list(self.no_signal_history),
        }

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.consecutive_no_signal = int(payload.get("consecutive_no_signal", 0))
        history = payload.get("groups", [])
        self.no_signal_history = list(history) if isinstance(history, list) else []

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.state_path.parent, delete=False
        ) as handle:
            json.dump(self.stop_report(), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, self.state_path)
