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

The implemented G0/G1 surface initializes immutable run directories and validates CSV or
Parquet PV observations into a canonical five-minute Parquet dataset. For one-minute input,
bins are right-closed: the interval ending at `t` contains exactly `t-4m, ..., t`. Incomplete
bins are excluded rather than interpolated. Source timestamps are normalized to UTC, data
availability is preserved, and every exclusion is recorded in a typed QC ledger.
Configured source-column names beginning with `_pv_pilot_` are rejected because that prefix is
reserved for collision-proof adapter bookkeeping. The data card distinguishes raw rows accepted
into canonical intervals from raw rows excluded; synthetic gap records are counted separately as
ledger records and never as raw input rows.

The bundled fixture is synthetic and exists only for software acceptance:

```powershell
uv run pv-pilot init-run --config configs/pilot.synthetic.yaml
uv run pv-pilot validate-data --run-dir <printed-run-directory>
```

Synthetic runs and unapproved protocols always remain `NOT_EVALUATED`; they cannot produce a
scientific GO/STOP decision.
