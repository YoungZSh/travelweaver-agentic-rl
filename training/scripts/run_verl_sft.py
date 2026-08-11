"""Add TravelWeaver SFT validation metrics before entering veRL's Hydra launcher."""

from __future__ import annotations

import json
import math
import os
from functools import partial
from typing import Any

import torch

_ACCURACY_MASK_KEY = "_travelweaver_accuracy_mask"
_ACCURACY_CORRECT_KEY = "_travelweaver_token_accuracy_correct"
_ACCURACY_TOTAL_KEY = "_travelweaver_token_accuracy_total"


def configure_tf32() -> dict[str, bool | str]:
    """Enable TF32 for float32 matmul and cuDNN operations in every trainer rank."""

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def configure_seeded_train_sampler(trainer: Any, runtime: Any) -> dict[str, int | bool]:
    """Rebuild veRL's train loader with the configured trainer seed."""

    seed = int(trainer.config.trainer.seed)
    dp_rank = trainer.engine.get_data_parallel_rank()
    dp_size = trainer.engine.get_data_parallel_size()
    trainer.train_sampler = runtime.DistributedSampler(
        trainer.train_dataset,
        shuffle=True,
        num_replicas=dp_size,
        rank=dp_rank,
        drop_last=True,
        seed=seed,
    )
    trainer.train_dataloader = runtime.StatefulDataLoader(
        dataset=trainer.train_dataset,
        batch_size=trainer.train_batch_size_per_dp,
        sampler=trainer.train_sampler,
        collate_fn=trainer.collate_fn,
        num_workers=trainer.config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        pin_memory_device=runtime.get_device_name(),
    )
    return {
        "shuffle": True,
        "seed": seed,
        "data_parallel_rank": dp_rank,
        "data_parallel_size": dp_size,
    }


def configure_validation_sampler(trainer: Any, runtime: Any) -> dict[str, int | bool] | None:
    """Keep validation deterministic and include its final partial batch."""

    if trainer.val_dataset is None:
        return None
    dp_rank = trainer.engine.get_data_parallel_rank()
    dp_size = trainer.engine.get_data_parallel_size()
    trainer.val_sampler = runtime.DistributedSampler(
        trainer.val_dataset,
        shuffle=False,
        num_replicas=dp_size,
        rank=dp_rank,
        drop_last=False,
    )
    trainer.val_dataloader = runtime.StatefulDataLoader(
        dataset=trainer.val_dataset,
        batch_size=trainer.train_batch_size_per_dp,
        sampler=trainer.val_sampler,
        collate_fn=trainer.collate_fn,
        num_workers=trainer.config.data.num_workers,
        pin_memory=False,
        drop_last=False,
        pin_memory_device=runtime.get_device_name(),
    )
    return {
        "shuffle": False,
        "drop_last": False,
        "data_parallel_rank": dp_rank,
        "data_parallel_size": dp_size,
    }


def _shifted_local_loss_mask(
    engine: Any, micro_batch: Any, output_args: dict[str, Any]
) -> torch.Tensor:
    """Align the supervised-token mask with the packed, Ulysses-sharded labels."""

    from verl.utils.ulysses import ulysses_pad_and_slice_inputs

    labels = output_args["input_ids_rmpad_rolled"]
    global_mask = torch.roll(micro_batch["loss_mask"].values().to(torch.bool), shifts=-1, dims=0)
    sp_size = int(engine.ulysses_sequence_parallel_size)
    padded_global_length = labels.numel() * sp_size
    if padded_global_length < global_mask.numel():
        raise RuntimeError("Validation labels are shorter than the global SFT loss mask.")
    padded_mask = torch.zeros(padded_global_length, dtype=torch.bool, device=global_mask.device)
    padded_mask[: global_mask.numel()] = global_mask
    if sp_size > 1:
        local_mask, _, _ = ulysses_pad_and_slice_inputs(
            padded_mask.unsqueeze(0), position_ids_rmpad=None, sp_size=sp_size, pad_value=False
        )
        local_mask = local_mask.squeeze(0)
    else:
        local_mask = padded_mask
    if local_mask.shape != labels.shape:
        raise RuntimeError("Validation accuracy mask does not align with the model labels.")
    return local_mask


def install_validation_accuracy_model_patch() -> None:
    """Add a sparse top-1 path to Qwen's existing Triton fused-loss forward pass."""

    from torch.distributed.tensor import DTensor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
    from verl.utils.ulysses import (
        get_ulysses_sequence_parallel_world_size,
        ulysses_pad_and_slice_inputs,
    )

    model_class = Qwen3_5ForConditionalGeneration
    if getattr(model_class.forward, "_travelweaver_validation_accuracy", False):
        return
    original_forward = model_class.forward

    def forward_with_validation_accuracy(
        self: Any,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        temperature: float = 1.0,
        shift_labels: torch.LongTensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        accuracy_mask = kwargs.pop(_ACCURACY_MASK_KEY, None)
        if accuracy_mask is None:
            return original_forward(
                self,
                input_ids=input_ids,
                labels=labels,
                temperature=temperature,
                shift_labels=shift_labels,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                **kwargs,
            )
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        if cu_seqlens_cpu is not None:
            kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
        outputs = self.model(input_ids, **kwargs)
        hidden_states = outputs[0]
        if shift_labels is not None:
            rolled_labels = shift_labels
        elif labels is not None:
            rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
        elif input_ids is not None:
            rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
        else:
            raise RuntimeError("Validation accuracy requires labels or input_ids.")
        if shift_labels is None and get_ulysses_sequence_parallel_world_size() > 1:
            rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
                rolled_labels,
                position_ids_rmpad=None,
                sp_size=get_ulysses_sequence_parallel_world_size(),
            )
        vocab_weights = self.lm_head.weight
        hidden_states = hidden_states.to(vocab_weights.dtype)
        if isinstance(vocab_weights, DTensor):
            vocab_weights = vocab_weights.full_tensor()
        log_probs, entropy = linear_cross_entropy(
            hidden_states,
            vocab_weights,
            rolled_labels,
            temperature,
            "none",
        )
        # veRL passes fused ``shift_labels`` as ``[1, local_tokens]`` while the
        # packed loss mask is intentionally one-dimensional.  Compare and index
        # their flattened forms: the leading singleton is transport metadata,
        # not an additional sequence dimension.
        accuracy_mask = accuracy_mask.to(device=rolled_labels.device, dtype=torch.bool).reshape(-1)
        flat_labels = rolled_labels.reshape(-1)
        if accuracy_mask.numel() != flat_labels.numel():
            raise RuntimeError(
                "Validation accuracy mask does not match local Ulysses labels: "
                f"mask={accuracy_mask.numel()}, labels={flat_labels.numel()}."
            )
        selected_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])[accuracy_mask]
        selected_labels = flat_labels[accuracy_mask]
        if selected_labels.numel() == 0:
            correct = torch.zeros((), dtype=torch.long, device=rolled_labels.device)
        else:
            predictions = torch.nn.functional.linear(selected_hidden, vocab_weights).argmax(dim=-1)
            correct = (predictions == selected_labels).sum(dtype=torch.long)
        result = Qwen3_5CausalLMOutputForPPO(
            log_probs=log_probs,
            entropy=entropy,
            hidden_states=outputs.hidden_states,
        )
        setattr(result, _ACCURACY_CORRECT_KEY, correct)
        setattr(
            result,
            _ACCURACY_TOTAL_KEY,
            torch.tensor(selected_labels.numel(), dtype=torch.long, device=rolled_labels.device),
        )
        return result

    forward_with_validation_accuracy._travelweaver_validation_accuracy = True
    model_class.forward = forward_with_validation_accuracy


def install_validation_accuracy_output_patch() -> None:
    """Move sparse correct/total counts from the model output into the SFT loss function."""

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    engine_class = FSDPEngineWithLMHead
    if getattr(engine_class.prepare_model_outputs, "_travelweaver_validation_accuracy", False):
        return
    original_prepare_model_inputs = engine_class.prepare_model_inputs
    original_prepare_model_outputs = engine_class.prepare_model_outputs

    def prepare_model_inputs_with_validation_accuracy(
        self: Any,
        micro_batch: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_inputs, output_args = original_prepare_model_inputs(self, micro_batch)
        if tu.get_non_tensor_data(micro_batch, key="calculate_token_accuracy", default=False):
            model_inputs[_ACCURACY_MASK_KEY] = _shifted_local_loss_mask(
                self, micro_batch, output_args
            )
        return model_inputs, output_args

    def prepare_model_outputs_with_validation_accuracy(
        self: Any,
        output: Any,
        output_args: dict[str, Any],
        micro_batch: Any,
        logits_processor_func: Any,
    ) -> dict[str, Any]:
        model_output = original_prepare_model_outputs(
            self,
            output,
            output_args,
            micro_batch,
            logits_processor_func,
        )
        if not tu.get_non_tensor_data(micro_batch, key="calculate_token_accuracy", default=False):
            return model_output
        correct = getattr(output, _ACCURACY_CORRECT_KEY, None)
        total = getattr(output, _ACCURACY_TOTAL_KEY, None)
        if correct is None or total is None:
            raise RuntimeError(
                "Qwen validation forward did not return sparse token-accuracy counts."
            )
        model_output[_ACCURACY_CORRECT_KEY] = correct
        model_output[_ACCURACY_TOTAL_KEY] = total
        return model_output

    prepare_model_outputs_with_validation_accuracy._travelweaver_validation_accuracy = True
    prepare_model_inputs_with_validation_accuracy._travelweaver_validation_accuracy = True
    engine_class.prepare_model_inputs = prepare_model_inputs_with_validation_accuracy
    engine_class.prepare_model_outputs = prepare_model_outputs_with_validation_accuracy


def sft_loss_with_validation_accuracy(
    config: Any,
    model_output: dict[str, Any],
    data: Any,
    dp_group: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extend veRL's SFT loss with globally aggregated supervised-token counts."""

    from verl.workers.utils.losses import sft_loss

    correct = model_output.pop(_ACCURACY_CORRECT_KEY, None)
    total = model_output.pop(_ACCURACY_TOTAL_KEY, None)
    loss, _ = sft_loss(
        config=config,
        model_output=model_output,
        data=data,
        dp_group=dp_group,
    )
    if correct is None and total is None:
        return loss, {}
    if correct is None or total is None:
        raise RuntimeError("Validation token-accuracy counts are incomplete.")
    from verl.utils.metric import AggregationType, Metric
    from verl.utils.ulysses import get_ulysses_sequence_parallel_group

    counts = torch.stack((correct, total)).to(dtype=torch.long)
    sp_group = get_ulysses_sequence_parallel_group()
    if sp_group is not None:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=sp_group)
    return loss, {
        "token_accuracy_correct": Metric(AggregationType.SUM, counts[0]),
        "token_accuracy_total": Metric(AggregationType.SUM, counts[1]),
    }


def _metric_sum(value: Any) -> float:
    """Extract a scalar count from veRL Metric/list aggregation containers."""

    from verl.utils.metric import Metric

    if isinstance(value, Metric):
        return float(value.aggregate())
    if isinstance(value, list):
        return sum(_metric_sum(item) for item in value)
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def install_seeded_sft_trainer(runtime: Any) -> None:
    """Install local veRL extensions without modifying the pinned dependency package."""

    base_trainer = runtime.SFTTrainer
    install_validation_accuracy_output_patch()

    class TravelWeaverSFTTrainer(base_trainer):
        def _build_engine(self) -> None:
            from verl.workers.engine_workers import TrainingWorkerConfig

            self.loss_fn = partial(sft_loss_with_validation_accuracy, config=None)
            config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=self.model_config,
                engine_config=self.engine_config,
                optimizer_config=self.optimizer_config,
                checkpoint_config=self.checkpoint_config,
                profiler_config=self.profiler_config,
            )
            self.training_client = runtime.TrainingWorker(config=config)
            self.training_client.set_loss_fn(loss_fn=self.loss_fn)
            self.engine = self.training_client.engine

        def _build_dataloader(self) -> None:
            super()._build_dataloader()
            sampler_report = configure_seeded_train_sampler(self, runtime)
            validation_report = configure_validation_sampler(self, runtime)
            if self.rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "sft_dataloader_configured",
                            "train": sampler_report,
                            "validation": validation_report,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        def _init_engine(self) -> None:
            super()._init_engine()
            install_validation_accuracy_model_patch()

        def _run_validation(
            self, meta_info: dict[str, Any], tracking: Any, global_step: int
        ) -> dict[str, float]:
            if self.val_dataloader is None:
                raise RuntimeError("Validation was requested without a validation dataloader.")
            val_losses: list[float] = []
            correct_count = 0.0
            total_count = 0.0
            for val_data in self.val_dataloader:
                val_data = runtime.tu.get_tensordict(
                    tensor_dict=val_data,
                    non_tensor_dict=meta_info,
                )
                runtime.tu.assign_non_tensor(
                    val_data,
                    calculate_token_accuracy=True,
                )
                output = self.training_client.infer_batch(val_data)
                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = runtime.tu.get(output, "metrics")
                    val_losses.append(float(metrics["loss"]))
                    correct_count += _metric_sum(metrics["token_accuracy_correct"])
                    total_count += _metric_sum(metrics["token_accuracy_total"])
            metric: dict[str, float] = {}
            if self.engine.is_mp_src_rank_with_outputs():
                val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                dp_group = self.engine.get_data_parallel_group()
                if dp_group is not None:
                    torch.distributed.all_reduce(
                        val_loss,
                        op=torch.distributed.ReduceOp.AVG,
                        group=dp_group,
                    )
                if total_count <= 0:
                    raise RuntimeError("Validation contains no supervised assistant tokens.")
                metric = {
                    "val/loss": val_loss.detach().item(),
                    "val/token_accuracy": correct_count / total_count,
                    "val/token_correct": correct_count,
                    "val/token_count": total_count,
                    "val/perplexity": math.exp(min(val_loss.detach().item(), 80.0)),
                }
            if (
                self.engine.is_mp_src_rank_with_outputs()
                and self.engine.get_data_parallel_rank() == 0
            ):
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()
            return metric

        def fit(self) -> None:
            is_logging = (
                self.engine.is_mp_src_rank_with_outputs()
                and self.engine.get_data_parallel_rank() == 0
            )
            tracking = None
            if is_logging:
                tracking = runtime.Tracking(
                    project_name=self.config.trainer.project_name,
                    experiment_name=self.config.trainer.experiment_name,
                    default_backend=self.config.trainer.logger,
                    config=runtime.OmegaConf.to_container(self.config, resolve=True),
                )

            global_step = self.resume_global_step
            last_valid_metric: dict[str, float] | None = None
            runtime.log_with_rank(
                f"Total training steps: {self.total_training_steps},",
                logger=runtime.logger,
                rank=0,
                log_only_rank_0=True,
            )
            if global_step > 0:
                runtime.log_with_rank(
                    f"StatefulDataLoader will automatically resume from global step: {global_step}",
                    logger=runtime.logger,
                    rank=0,
                    log_only_rank_0=True,
                )

            start_epoch = global_step // self.steps_per_epoch
            meta_info = {
                "use_remove_padding": self.config.model.use_remove_padding,
                "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
                "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
                "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
                "temperature": 1.0,
                "global_batch_size": self.global_batch_size,
                "pad_mode": self.config.data.pad_mode,
                "pad_token_id": self.model_config.tokenizer.pad_token_id,
            }
            train_time = 0.0
            total_tokens = 0
            for epoch in range(start_epoch, self.config.trainer.total_epochs):
                self.train_sampler.set_epoch(epoch=epoch)
                runtime.aggressive_empty_cache(force_sync=True)
                runtime.log_gpu_memory_usage(
                    f"rank {self.rank}: At start of epoch {epoch}",
                    logger=runtime.logger,
                )
                for step_in_epoch, data in enumerate(
                    runtime.tqdm(
                        self.train_dataloader,
                        initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                        total=self.steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                        disable=not is_logging,
                    )
                ):
                    del step_in_epoch
                    global_step += 1
                    data = runtime.tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                    batch_seqlens = self._get_batch_seqlens(data=data)
                    runtime.tu.assign_non_tensor(
                        data,
                        update_lr_scheduler=True,
                        global_token_num=runtime.NonTensorData(batch_seqlens),
                    )
                    if global_step == self.start_profile_step:
                        self.training_client.start_profile()
                    output = self.training_client.train_batch(data=data)
                    self.training_client.step_profile()
                    if global_step == self.end_profile_step:
                        self.training_client.stop_profile()
                    if self.engine.is_mp_src_rank_with_outputs():
                        metrics = runtime.tu.get(output, "metrics")
                        for key in ("loss", "grad_norm", "lr", "mfu"):
                            if key in metrics:
                                metrics[f"train/{key}"] = metrics.pop(key)
                        metrics["train/global_tokens"] = torch.sum(
                            torch.tensor(batch_seqlens, device=self.device_name)
                        ).item()
                        total_tokens += metrics["train/global_tokens"]
                        metrics["train/total_tokens(B)"] = total_tokens / 1e9
                        if self.engine.get_data_parallel_rank() == 0:
                            assert tracking is not None
                            tracking.log(data=metrics, step=global_step)

                    is_last_step = global_step >= self.total_training_steps
                    is_valid_step = self.test_freq > 0 and global_step % self.test_freq == 0
                    is_save_step = self.save_freq > 0 and global_step % self.save_freq == 0
                    if self.val_dataloader is not None and (is_last_step or is_valid_step):
                        assert tracking is not None or not is_logging
                        last_valid_metric = self._run_validation(meta_info, tracking, global_step)
                    if is_last_step or is_save_step:
                        runtime.aggressive_empty_cache(force_sync=True)
                        self.ckpt_handler.save_checkpoint(step=global_step)
                    if is_last_step:
                        if is_logging:
                            print(f"Total time for train steps: {train_time:.2f}s")
                            print(f"Final validation metrics: {last_valid_metric}")
                        return

    runtime.SFTTrainer = TravelWeaverSFTTrainer


def main() -> None:
    """Configure process-wide math settings and delegate CLI parsing to veRL."""

    report = configure_tf32()
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"event": "tf32_enabled", **report}, sort_keys=True), flush=True)

    from verl.trainer import sft_trainer as runtime  # noqa: PLC0415

    install_seeded_sft_trainer(runtime)
    runtime.main()


if __name__ == "__main__":
    main()
