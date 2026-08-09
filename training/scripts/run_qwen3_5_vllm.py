"""Run a reproducible text-only Qwen3.5 smoke inference with vLLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

DEFAULT_PROMPT = "请用不超过三句话说明第一次去重庆旅游最值得注意的三件事。"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("ckpts/Qwen3.5-4B"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/outputs/qwen3_5_vllm_smoke.json"),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_prompt(model_path: Path, prompt: str, enable_thinking: bool) -> tuple[str, Any]:
    """Load local metadata and render the official Qwen chat template."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    architectures = config.architectures or []
    unsupported = set(architectures) - set(ModelRegistry.get_supported_archs())
    if unsupported:
        raise RuntimeError(f"vLLM does not support checkpoint architectures: {unsupported}")
    if config.model_type != "qwen3_5":
        raise RuntimeError(f"Expected qwen3_5 checkpoint, found {config.model_type!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    metadata = {
        "config_class": config.__class__.__name__,
        "model_type": config.model_type,
        "architectures": architectures,
        "prompt_tokens": len(tokenizer.encode(rendered_prompt)),
    }
    return rendered_prompt, metadata


def main() -> None:
    """Validate the checkpoint, run vLLM, and write the generated response."""
    args = parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    if args.tensor_parallel_size < 1:
        raise SystemExit("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise SystemExit("--gpu-memory-utilization must be between 0 and 1")

    enable_thinking = not args.disable_thinking
    rendered_prompt, metadata = build_prompt(model_path, args.prompt, enable_thinking)
    print(
        json.dumps(
            {
                "event": "preflight_complete",
                "model": str(model_path),
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "enable_thinking": enable_thinking,
                **metadata,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    print(json.dumps({"event": "engine_loading"}), flush=True)
    load_started = time.perf_counter()
    engine = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        language_model_only=True,
        seed=args.seed,
        enable_prefix_caching=False,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
    )
    load_seconds = time.perf_counter() - load_started
    print(json.dumps({"event": "engine_ready", "seconds": load_seconds}), flush=True)

    sampling = SamplingParams(
        temperature=1.0 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    generation_started = time.perf_counter()
    requests = engine.generate([rendered_prompt], sampling_params=sampling, use_tqdm=True)
    generation_seconds = time.perf_counter() - generation_started
    if len(requests) != 1 or len(requests[0].outputs) != 1:
        raise RuntimeError("Expected exactly one vLLM request output")

    completion = requests[0].outputs[0]
    result = {
        "model": str(model_path),
        "prompt": args.prompt,
        "response": completion.text,
        "finish_reason": completion.finish_reason,
        "prompt_tokens": len(requests[0].prompt_token_ids),
        "completion_tokens": len(completion.token_ids),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "enable_thinking": enable_thinking,
        "seed": args.seed,
        **metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "generation_complete", **result}, ensure_ascii=False), flush=True)
    print(f"Wrote result to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
