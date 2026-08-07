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
- Future RL training stack: `verl==0.8.0` in a separate Linux/CUDA environment

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
├── reward/       # deterministic constraint verification and strict RFT filter
├── evaluation/   # blind offline LLM Judge and separated evaluation reports
└── cli/          # command-line entry points
```

The lightweight environment remains independent of the future Linux/CUDA SFT and GRPO
training stack. Public imports use component namespaces such as
`from travelweaver.env import TravelWeaverEnv`.

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
shared OpenAI-compatible function-calling client. Each `travelweaver-trajectory-v3`
record contains canonical `messages`, `tools`, executed `steps`, terminal Reward details,
RFT acceptance, and a separate audit event stream. Local dotenv files and generated
trajectories are ignored by Git.

## Grounded task synthesis

The pilot generator derives typed constraints from a plan that already passes the real
environment, then uses DeepSeek only to polish the Chinese query surface. It never asks
the LLM to invent budgets, entities, transport modes, or other scored semantics.

```bash
uv run travelweaver synthesize-tasks \
  --count 50 \
  --seed 20260807 \
  --output-dir data/generated/pilot-50-v1
```

The command is resumable and capped at 200 API requests. Public tasks, hidden TaskSpecs,
Blueprints, surfaces, replayable witnesses, a manifest, quarantine log, and a Markdown
preview are written below the output directory. `data/generated/` is local-only and
ignored by Git. See [the synthesis pilot contract](docs/task-synthesis-pilot.md).

TravelWeaver deliberately keeps tool execution in-process for the environment milestone.
MCP is not part of the current architecture. Deterministic Reward and strict RFT filtering
are implemented; SFT preprocessing and veRL/GRPO training integration remain later
milestones.
