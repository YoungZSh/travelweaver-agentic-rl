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

GroupRecord = tuple[str, list[dict[str, Any]]]
QUALITY_DIMENSIONS = ("artifact_score", "validity_score", "goal_score")


def build_sampling_metrics(
    *,
    initial_uids: set[str],
    group_records: dict[str, GroupRecord],
    refill_groups: int,
    selected_uids: set[str],
    group_size: int,
) -> dict[str, float | int]:
    """Summarize fixed initial sampling separately from adaptive refill traffic."""

    prefix = "training"
    initial_records = {
        uid: record for uid, record in group_records.items() if uid in initial_uids
    }
    initial_infos = [info for _, infos in initial_records.values() for info in infos]
    valid_rewards = [
        float(info["travelweaver_reward"])
        for info in initial_infos
        if bool(float(info.get("reward_valid", 0.0)))
        and "travelweaver_reward" in info
    ]
    initial_successes = [
        bool(float(info.get("reward_valid", 0.0)))
        and bool(float(info.get("all_hard_pass", 0.0)))
        for info in initial_infos
    ]
    complete_initial_groups = [
        infos for _, infos in initial_records.values() if len(infos) == group_size
    ]
    outcome_counts: Counter[str] = Counter()
    for infos in complete_initial_groups:
        successes = sum(
            bool(float(info.get("reward_valid", 0.0)))
            and bool(float(info.get("all_hard_pass", 0.0)))
            for info in infos
        )
        if successes == 0:
            outcome_counts["unsolved"] += 1
        elif successes == group_size:
            outcome_counts["mastered"] += 1
        else:
            outcome_counts["frontier"] += 1

    classification_counts = Counter(classification for classification, _ in group_records.values())
    filtered_outcomes: Counter[str] = Counter()
    for classification, infos in group_records.values():
        if classification != "zero_variance" or len(infos) != group_size:
            continue
        successes = sum(
            bool(float(info.get("reward_valid", 0.0)))
            and bool(float(info.get("all_hard_pass", 0.0)))
            for info in infos
        )
        if successes == 0:
            filtered_outcomes["unsolved"] += 1
        elif successes == group_size:
            filtered_outcomes["mastered"] += 1
        else:
            filtered_outcomes["other"] += 1

    initial_group_count = len(initial_uids)
    completed_initial_count = len(complete_initial_groups)
    dispatched_groups = initial_group_count + refill_groups
    completed_trajectories = sum(len(infos) for _, infos in group_records.values())
    metrics: dict[str, float | int] = {
        f"{prefix}/policy_quality/raw_initial/completed_groups": completed_initial_count,
        f"{prefix}/policy_quality/raw_initial/reward_valid_rate": (
            sum(bool(float(info.get("reward_valid", 0.0))) for info in initial_infos)
            / len(initial_infos)
            if initial_infos
            else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/reward_mean": (
            sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/success_avg_at_{group_size}": (
            sum(initial_successes) / len(initial_successes) if initial_successes else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/pass_at_{group_size}": (
            (outcome_counts["frontier"] + outcome_counts["mastered"])
            / completed_initial_count
            if completed_initial_count
            else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/unsolved_at_{group_size}": (
            outcome_counts["unsolved"] / completed_initial_count
            if completed_initial_count
            else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/frontier_at_{group_size}": (
            outcome_counts["frontier"] / completed_initial_count
            if completed_initial_count
            else 0.0
        ),
        f"{prefix}/policy_quality/raw_initial/mastered_at_{group_size}": (
            outcome_counts["mastered"] / completed_initial_count
            if completed_initial_count
            else 0.0
        ),
        f"{prefix}/sampler/initial_groups": initial_group_count,
        f"{prefix}/sampler/refill_groups": refill_groups,
        f"{prefix}/sampler/dispatched_groups": dispatched_groups,
        f"{prefix}/sampler/completed_groups": len(group_records),
        f"{prefix}/sampler/selected_groups": len(selected_uids),
        f"{prefix}/sampler/filtered_zero_variance_groups": classification_counts[
            "zero_variance"
        ],
        f"{prefix}/sampler/filtered_unsolved_groups": filtered_outcomes["unsolved"],
        f"{prefix}/sampler/filtered_mastered_groups": filtered_outcomes["mastered"],
        f"{prefix}/sampler/filtered_other_zero_variance_groups": filtered_outcomes[
            "other"
        ],
        f"{prefix}/sampler/invalid_groups": classification_counts["invalid"],
        f"{prefix}/sampler/length_exceeded_groups": classification_counts[
            "length_exceeded"
        ],
        f"{prefix}/sampler/refill_overhead_ratio": (
            refill_groups / initial_group_count if initial_group_count else 0.0
        ),
        f"{prefix}/sampler/dispatched_trajectories": dispatched_groups * group_size,
        f"{prefix}/sampler/completed_trajectories": completed_trajectories,
    }
    for dimension in QUALITY_DIMENSIONS:
        values = [
            float(info[dimension])
            for info in initial_infos
            if bool(float(info.get("reward_valid", 0.0))) and dimension in info
        ]
        metrics[f"{prefix}/policy_quality/raw_initial/{dimension}_mean"] = (
            sum(values) / len(values) if values else 0.0
        )
    return metrics


def classify_reward_group(
    reward_infos: list[dict[str, Any]], *, group_size: int, tolerance: float
) -> tuple[str, float | None]:
    """Classify a complete GRPO group from the shared online Reward fields."""

    if len(reward_infos) != group_size:
        return "invalid", None
    if any(bool(float(info.get("trajectory_length_exceeded", 0.0))) for info in reward_infos):
        return "length_exceeded", None
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
        self._length_diagnostics: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
        self._length_max_since_sample = 0.0
        self._upstream_refill_fn = kwargs.pop("refill_fn")
        self._sampling_initial_uids: set[str] | None = None
        self._sampling_group_records: dict[str, GroupRecord] | None = None
        self._sampling_refill_groups = 0
        self.consecutive_no_signal = 0
        self.no_signal_history: list[dict[str, Any]] = []
        self._load_state()
        super().__init__(
            *args,
            **kwargs,
            refill_fn=self._tracked_refill,
            filter_groups_metric="travelweaver_reward",
            train_batch_size=int(sampler_kwargs.get("train_batch_size", 1)),
            gen_batch_size=1,
            max_inflight_gen_batches=1,
            sync_refill_failed_groups=True,
        )

    def _tracked_refill(self, num_prompts: int) -> int:
        if self._sampling_group_records is not None:
            self._sampling_refill_groups += num_prompts
        return int(self._upstream_refill_fn(num_prompts))

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        if partition_id == "val":
            return set(), Counter()
        finished_uids = self.finished_keys[partition_id]
        cache = self._classification[partition_id]
        length_diagnostics = self._length_diagnostics[partition_id]
        for uid in set(cache) - finished_uids:
            del cache[uid]
            length_diagnostics.pop(uid, None)
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
            if self._sampling_group_records is not None:
                self._sampling_group_records[uid] = (classification, reward_infos)
            length_diagnostics[uid] = (
                sum(
                    bool(float(info.get("trajectory_length_exceeded", 0.0)))
                    for info in reward_infos
                ),
                max(
                    (float(info.get("sequence_tokens", 0.0)) for info in reward_infos),
                    default=0.0,
                ),
            )
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
            if classification in {"zero_variance", "invalid", "length_exceeded"}
        }
        counts = Counter(
            shared
            for classification, shared in cache.values()
            if classification == "zero_variance" and shared is not None
        )
        return filtered, counts

    def _evict_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> tuple[set[str], int, int, dict]:
        length_uids = {
            uid
            for uid in eviction_reasons[1]
            if self._classification[partition_id].get(uid, (None, None))[0]
            == "length_exceeded"
        }
        length_diagnostics = self._length_diagnostics[partition_id]
        overflow_trajectories = sum(length_diagnostics.get(uid, (0, 0.0))[0] for uid in length_uids)
        max_observed = max(
            (length_diagnostics.get(uid, (0, 0.0))[1] for uid in length_uids),
            default=0.0,
        )
        self._length_max_since_sample = max(self._length_max_since_sample, max_observed)
        evicted, stale_count, dapo_count, metrics = super()._evict_terminal_groups(
            global_steps,
            partition_id,
            eviction_reasons,
        )
        if length_uids:
            prefix = self._metrics_prefix(partition_id)
            metrics.update(
                {
                    f"{prefix}/trajectory_length/overflow_groups": len(length_uids),
                    f"{prefix}/trajectory_length/overflow_trajectories": overflow_trajectories,
                    f"{prefix}/trajectory_length/refill_groups": len(length_uids),
                }
            )
        return evicted, stale_count, dapo_count, metrics

    def sample(
        self, global_steps: int, partition_id: str, batch_size: int
    ) -> tuple[Any, dict[str, Any]]:
        self._length_max_since_sample = 0.0
        if partition_id == "train":
            self._sync_metadata_from_transfer_queue()
            self._sampling_initial_uids = set(self.prompt_global_steps[partition_id])
            self._sampling_group_records = {}
            self._sampling_refill_groups = 0
        try:
            batch, metrics = super().sample(global_steps, partition_id, batch_size)
            if partition_id == "train":
                selected_uids = {
                    key.split("_")[0]
                    for key, tag in zip(batch.keys, batch.tags, strict=True)
                    if not tag.get("is_padding", False)
                }
                metrics.update(
                    build_sampling_metrics(
                        initial_uids=self._sampling_initial_uids or set(),
                        group_records=self._sampling_group_records or {},
                        refill_groups=self._sampling_refill_groups,
                        selected_uids=selected_uids,
                        group_size=self.group_size,
                    )
                )
        finally:
            self._sampling_initial_uids = None
            self._sampling_group_records = None
            self._sampling_refill_groups = 0
        selected_lengths = [
            float(tag["seq_len"])
            for tag in batch.tags
            if "seq_len" in tag and not tag.get("is_padding", False)
        ]
        max_selected = max(selected_lengths, default=0.0)
        prefix = self._metrics_prefix(partition_id)
        metrics[f"{prefix}/trajectory_length/max_observed_tokens"] = max(
            max_selected, self._length_max_since_sample
        )
        return batch, metrics

    def _record_signal_result(
        self,
        uid: str,
        classification: str,
        shared_reward: float | None,
        reward_infos: list[dict[str, Any]],
        global_step: int | None,
    ) -> None:
        if classification in {"invalid", "length_exceeded"}:
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
