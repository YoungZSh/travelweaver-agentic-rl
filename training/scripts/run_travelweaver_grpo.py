"""Hydra entry point that registers TravelWeaver's graceful-stop trainer."""

from __future__ import annotations

from pprint import pprint

import hydra
import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class TravelWeaverTaskRunner:
    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self) -> None:
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        manager_cls = (
            load_class_from_fqn(manager_fqn, "AgentLoopManager")
            if manager_fqn
            else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()
        from travelweaver_grpo_trainer import TravelWeaverPPOTrainerSync

        del TravelWeaverPPOTrainerSync
        from verl.trainer.ppo.v1 import get_trainer_cls

        trainer_cls = get_trainer_cls("travelweaver_sync")
        with open_dict(config):
            config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    with open_dict(config):
        config.trainer.v1.trainer_mode = "travelweaver_sync"
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=TravelWeaverTaskRunner)


if __name__ == "__main__":
    main()
