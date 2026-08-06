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
uv run travelweaver import-tasks --split easy
uv run travelweaver smoke-env
uv run travelweaver run-agent --task-id e20241028160248698752
uv run pytest
```

The ChinaTravel database is not redistributed by this repository. The bootstrap command
can download the official Google Drive folder, import a local archive, or verify an
existing manual installation.

See [the MVP guide](docs/travelweaver-env-mvp.md) and the broader
[project design](docs/project-design.md).

## Version baseline

- Python: `3.10.19`
- Environment protocol: `travelweaver-environment-v0.2`
- Tool protocol: `travelweaver-tools-v1-agent`
- Future RL training stack: `verl==0.8.0` in a separate Linux/CUDA environment

ChinaTravel remains pinned as a Git submodule. Its dataset is published under
CC BY-NC-SA 4.0; review upstream terms before redistributing data or using it
commercially.
