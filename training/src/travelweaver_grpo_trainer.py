"""Graceful-stop veRL trainer extension for TravelWeaver."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import open_dict
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@register_trainer("travelweaver_sync")
class TravelWeaverPPOTrainerSync(PPOTrainerSync):
    def __init__(self, config: Any):
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config)

    def fit(self, agent_loop_manager: Any) -> None:
        try:
            super().fit(agent_loop_manager)
        except Exception as error:
            if not getattr(error, "travelweaver_no_signal_stop", False):
                raise
            try:
                self._stop_profiling()
            except Exception:
                pass
            interrupted_step = int(self.global_steps)
            last_completed_step = max(0, interrupted_step - 1)
            self.global_steps = last_completed_step
            try:
                self._save_checkpoint()
            finally:
                self.global_steps = interrupted_step
            report = dict(getattr(error, "report", {}))
            report["last_completed_global_step"] = last_completed_step
            report["checkpoint_global_step"] = last_completed_step
            report["reward_version"] = "travelweaver-reward-v4"
            destination = Path(self.config.trainer.default_local_dir) / "stop-report.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=destination.parent, delete=False
            ) as handle:
                json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, destination)
            self.on_train_end()
            self._shutdown_dump_executor()
