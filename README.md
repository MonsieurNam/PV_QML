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
Every published `stage_result.json` is declared by its stage manifest. Its normalized terminal
status must match the marker before publication, during crash recovery, on resume, and whenever a
downstream stage verifies prerequisites.

Canonical and QC Parquet artifacts use explicit Arrow schemas: UTC timestamps are stored at
nanosecond precision, quality flags are non-null `list<string>`, and a configured daylight field
is a non-null boolean after invalid/missing values are excluded. CSV headers are parsed before
pandas ingestion so duplicate names are rejected and identifier strings such as `NA` are retained.
Gap reporting is compact and never materializes the complete missing timestamp range; a per-site
span above 3,660 days is treated as an invalid source timestamp contract.
Validation checks the configured wall-clock and memory limits around source materialization,
quality control, serialization, and final publication. Exceeding a cap discards candidate outputs
and publishes a bounded typed `INVALID` diagnostic bundle without reapplying the expired cap.

The bundled fixture is synthetic and exists only for software acceptance:

```powershell
uv run pv-pilot init-run --config configs/pilot.synthetic.yaml
uv run pv-pilot validate-data --run-dir <printed-run-directory>
```

Synthetic runs and unapproved protocols always remain `NOT_EVALUATED`; they cannot produce a
scientific GO/STOP decision.
