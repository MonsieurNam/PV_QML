# Photonic PV Pilot

This repository is a reproducible Python 3.12 pilot for studying photonic quantum
machine-learning methods on photovoltaic data.

## Development setup

Install the pinned environment and run the quality gates with
[uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen
uv run ruff check .
uv run mypy src
uv run pytest --cov=src/pv_pilot --cov-fail-under=85
uv run pytest tests/test_compat.py -q
```

The implemented G0-G3 surface initializes immutable run directories and validates CSV or
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
Schema version 1.0 therefore accepts timestamps only in the physical `timestamp[ns, UTC]` range,
from `1677-09-21T00:12:43.145224193Z` through `2262-04-11T23:47:16.854775807Z` inclusive;
interval or availability values outside that range are deterministically excluded.
Gap reporting is compact and never materializes the complete missing timestamp range; a per-site
span above 3,660 days is treated as an invalid source timestamp contract.
Validation checks the configured wall-clock and memory limits around source materialization,
quality control, serialization, and final publication. Exceeding a cap discards candidate outputs
and publishes a bounded typed `INVALID` diagnostic bundle without reapplying the expired cap.

Human cleaning uses a separate fixed 10 engineer-hour timebox. Optional `data.cleaning` declares
cumulative `human_cleaning_hours` (finite and nonnegative) and boolean `qc_policy_locked`; for
example, `cleaning: {human_cleaning_hours: 4.5, qc_policy_locked: false}`. At or above 10 hours,
the QC policy must be locked before a run can start. Retain valid canonical blocks without
interpolation; each feature sample requires every same-site 5-minute interval from t-60 through
t+15, inclusive. Future interval continuity is retrospective target eligibility only.
Omitting cleaning or setting it to null records `NOT_RECORDED`, including synthetic runs;
compute wall-clock time never supplies engineer-hours. Changing a declaration requires a new
run, and resumed immutable ledgers retain their original declaration.

The bundled fixture is synthetic and exists only for software acceptance:

```powershell
uv run pv-pilot init-run --config configs/pilot.synthetic.yaml
uv run pv-pilot validate-data --run-dir <printed-run-directory>
uv run pv-pilot build-splits --run-dir <printed-run-directory>
uv run pv-pilot train-baselines --run-dir <printed-run-directory>
uv run pv-pilot build-photonic-features --run-dir <printed-run-directory>
uv run pv-pilot benchmark --run-dir <printed-run-directory>
uv run pv-pilot report --run-dir <printed-run-directory>
```

The same immutable sequence can be started or resumed with one command:

```powershell
uv run pv-pilot run-pilot --config configs/pilot.synthetic.yaml
uv run pv-pilot run-pilot --run-dir <printed-run-directory>
```

The checked-in acceptance configuration uses a compact, deterministic 45-day fixture with
15 train, 10 validation, and 20 holdout samples after leakage-safe feature construction. It
still executes the complete frozen model grids, five circuit seeds, seven quantile heads,
Perceval SLOS distributions, and 2,000-replicate bootstrap protocol. Quantile heads belonging
to one candidate are executed as a single resource-monitored worker batch so Windows process
startup does not dominate the scientific computation.

`train-baselines` selects and refits models using only the development partition. After
photonic selection is frozen, `benchmark` opens one authenticated holdout snapshot and uses
that same frame for baseline and photonic predictions. Parsers consume descriptor-captured
bytes, and stage input hashes describe those exact bytes.

Initialization records source inventories for `src/` and `scripts/`, the configuration file,
`pyproject.toml`, `uv.lock`, and the executing Python/package environment. Resume and stage
execution reject provenance drift, including newly added source modules; research notes,
PDFs, documentation, and bytecode caches do not affect the source inventory. The read-only
acceptance verifier can inspect historical artifacts after execution provenance has changed.

The v1 primary scientific contract uses two photons and retains a hard cap of three photons.
Optional three-photon profiling and benchmarking require explicit frozen opt-ins and are
diagnostic only. `configs/pilot.synthetic.three-photon.yaml` enables both in a separate artifact
directory. The three-photon resource gate must pass before benchmark selection; both selections
freeze before one shared holdout materialization. The report keeps three-photon results separate
from the GO/STOP decision. Validation origins are UTC calendar days; they are blocked scoring
groups over fixed train/validation partitions, not rolling refits.

`configs/pilot.example.yaml` carries the real-pilot caps (12 hours and 24 GiB per stage). The
synthetic CI configuration deliberately retains its smaller acceptance-only caps.

Run and verify the complete software acceptance path with:

```powershell
uv run pv-pilot run-pilot --config configs/pilot.synthetic.yaml
uv run python scripts/verify_acceptance_run.py --config configs/pilot.synthetic.yaml
```

For this synthetic protocol the terminal report is deliberately `BLOCKED/not_evaluated`.
`run-pilot` returns exit code 0 only for that completed synthetic safeguard. An upstream
`BLOCKED`, any `INVALID` or `FAILED` stage, malformed evidence, or a resource-cap violation
remains a nonzero failure. The verifier is read-only: it checks immutable stage inventories,
CPU-only resource ledgers, the finite-shot limitation, and the absence of a scientific decision
without training or recomputing features.

The report reads only finalized artifacts; it does not retrain models, rebuild photonic
features, rerun the benchmark, or reopen the configured raw input and physical split holdout.
It reports exact quantum-only, full-system, and matched-baseline scores separately. Finite-shot
results are explicitly `ideal_shot_noise_only` diagnostics and never drive GO/STOP.

Synthetic runs and unapproved protocols always remain `NOT_EVALUATED`; they cannot produce a
scientific GO/STOP decision or a quantum-advantage claim.
