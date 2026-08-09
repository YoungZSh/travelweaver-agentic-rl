# TravelWeaver

TravelWeaver is a deterministic, replayable environment for training long-horizon
travel-planning agents. The first milestone exposes 13 JSON tools over a pinned
[ChinaTravel](https://github.com/LAMDA-NeSy/ChinaTravel) snapshot and implements the
complete in-process query → candidate → plan submission episode lifecycle.

## Quick start

```bash
git submodule update --init --recursive
uv sync --dev
uv run travelweaver bootstrap chinatravel
uv run travelweaver import-tasks --split benchmark
uv run travelweaver smoke-env
uv run travelweaver run-agent --task-id e20241028160248698752
uv run pytest
```

The ChinaTravel database is not redistributed by this repository. The bootstrap command
can download the official Google Drive folder, import a local archive, or verify an
existing manual installation.

See [the MVP guide](docs/travelweaver-env-mvp.md), the frozen
[Reward and evaluation contract](docs/reward-and-evaluation.md), and the broader
[project design](docs/project-design.md).

## Version baseline

- Python: `3.10.19`
- Environment protocol: `travelweaver-environment-v0.3`
- Tool protocol: `travelweaver-tools-v2-agent`
- TaskSpec / Reward: `travelweaver-task-spec-v2` / `travelweaver-reward-v1`
- Trajectory / model response: `travelweaver-trajectory-v5` /
  `travelweaver-model-tool-response-v1`
- RL training stack: pinned `verl==0.9.0.dev0` in the separate `training/` Linux/CUDA environment

ChinaTravel remains pinned as a Git submodule. Its dataset is published under
CC BY-NC-SA 4.0; review upstream terms before redistributing data or using it
commercially.

## Package layout

TravelWeaver uses one top-level Python namespace with explicit component boundaries:

```text
src/travelweaver/
├── env/          # deterministic episode state, tools, and ChinaTravel backend
├── data/         # pinned task snapshots and database preparation
├── tasks/        # source-independent TaskSpec and safe source adapters
├── llm/          # shared OpenAI-compatible client and provider presets
├── synthesis/    # witness-first task synthesis and LLM surface polishing
├── rollout/      # agent policies and trajectory collection
├── sft/          # deterministic action replay and neutral SFT reconstruction
├── reward/       # deterministic constraint verification and strict RFT filter
├── evaluation/   # blind offline LLM Judge and separated evaluation reports
└── cli/          # command-line entry points
```

The lightweight environment remains independent of the Linux/CUDA SFT and GRPO
training stack. Public imports use component namespaces such as
`from travelweaver.env import TravelWeaverEnv`.

The isolated [training project](training/README.md) pins the CUDA 12.8-compatible veRL,
vLLM, and PyTorch stack. Create or update its separate virtual environment with
`uv sync --project training --dev`; training code, configuration, and launchers belong
under `training/`.

Accepted rollouts can be deterministically rebuilt into action-only SFT data with
`travelweaver rebuild-sft`. The root command replays valid actions and writes an audited neutral
JSONL; the isolated training adapter renders it with Qwen3.5's official chat template. See
[the SFT reconstruction design](docs/sft-trajectory-reconstruction-v1.md).

## DeepSeek API rollout

Install the optional API client and create a local dotenv file:

```bash
uv sync --extra api --dev
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`, then run one model-driven tool episode:

```bash
uv run travelweaver rollout-api --task-id e20241028160248698752
```

The command loads `.env`, calls the official OpenAI-compatible DeepSeek API, and appends
the full replayable record to `data/trajectories/deepseek-v4-flash.jsonl`. The rollout
runner itself is provider-neutral: DeepSeek is currently a configuration preset over the
shared OpenAI-compatible function-calling client. Each `travelweaver-trajectory-v5`
record contains canonical `messages`, `tools`, executed `steps`, terminal Reward details,
RFT acceptance, and a separate audit event stream. Local dotenv files and generated
trajectories are ignored by Git.

Model-visible tool messages default to the versioned `delta` response mode, so each turn
adds only the new tool result, an optional error, and the remaining-step count. Full
environment snapshots remain in `steps[].result` for replay and audit. Pass
`--tool-response-mode snapshot` only when reproducing the legacy model context.

For a generated task directory, run one resumable rollout per task with the tested batch
command. It defaults to 256 concurrent episodes:

```bash
uv run travelweaver rollout-generated \
  --input-dir data/generated/chinatravel-blended-200-v1.1-repolished-minimal \
  --output data/trajectories/chinatravel-blended-v1.1-thinking.jsonl \
  --concurrency 256
```

## Grounded task synthesis

The pilot generator derives typed constraints from a plan that already passes the real
environment, then uses DeepSeek only to polish the Chinese query surface. It never asks
the LLM to invent budgets, entities, transport modes, or other scored semantics.

```bash
uv run travelweaver synthesize-tasks \
  --count 100 \
  --seed 20260807 \
  --max-api-calls 300 \
  --output-dir data/generated/pilot-100-v2.1
```

The command exposes one seed and is resumable. Public tasks, hidden TaskSpecs, explicit
replayable Scenarios, Blueprints, surfaces, witnesses, diversity metrics, a manifest,
quarantine log, and a Markdown preview are written below the output directory.
`data/generated/` is local-only and ignored by Git. See
[the synthesis contract](docs/task-synthesis-pilot.md).

The mixed-coverage profile uses the accepted 200-task recipe as a deterministic ratio
baseline. It also supports larger `--count` values with seed-derived remainder allocation:

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1 \
  --count 200 \
  --seed 20260808 \
  --max-api-calls 400 \
  --output-dir data/generated/chinatravel-blended-200-v1
```

The V1.1 quality trial uses `--profile chinatravel_blended_v1_1` and writes to
`data/generated/chinatravel-blended-200-v1.1`.

For example, one 500-task production batch can use:

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1_1 \
  --count 500 \
  --seed 20260811 \
  --max-api-calls 1000 \
  --output-dir data/generated/chinatravel-blended-500-v1.1-b01
```

To preserve the accepted Blueprints and witnesses while rerunning only the LLM surface
rewrites, use the concurrent repolish command. It defaults to 256 simultaneous requests
and writes every accepted or rejected raw tool response to `polish-audit.jsonl`:

```bash
uv run travelweaver repolish-tasks \
  --input-dir data/generated/chinatravel-blended-200-v1.1 \
  --output-dir data/generated/chinatravel-blended-200-v1.1-repolished \
  --llm-concurrency 256 \
  --validation-policy minimal_semantic \
  --max-api-calls 400
```

TravelWeaver deliberately keeps tool execution in-process for the environment milestone.
MCP is not part of the current architecture. Deterministic Reward and strict RFT filtering
are implemented; SFT preprocessing and veRL/GRPO training integration remain later
milestones.
