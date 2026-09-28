# Photonic PV Pilot

This repository is a reproducible Python 3.12 pilot for studying photonic quantum
machine-learning methods on photovoltaic data.

## Development setup

Install the pinned environment and run the quality gates with
[uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run ruff check .
uv run mypy
uv run pytest
```

The first milestone only bootstraps the package and verifies the Perceval 1.3 runtime.
Configuration, command-line, and data-pipeline features are intentionally not included yet.
