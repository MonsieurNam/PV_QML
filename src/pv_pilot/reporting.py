"""Read-only G3 report construction from immutable stage artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .baselines import LOCKED_QUANTILES, prediction_arrow_schema, quantile_columns
from .benchmark import exact_prediction_arrow_schema, finite_shot_prediction_arrow_schema
from .config import PilotConfig
from .domain import ResourceGateEvidence, ScoreBundle, deterministic_data
from .evaluation import (
    EvaluationPolicy,
    EvaluationResult,
    ExpectedPrimaryConfiguration,
    evaluate_predictions,
    score_bundle_csv_row,
    timestamp_normalized_wis,
)
from .seeds import SeedManager

ResourceCheckpoint = Callable[[int], None]


class ReportContractError(ValueError):
    """Raised when frozen artifacts cannot support an auditable report."""


@dataclass(frozen=True, slots=True)
class ReportProducts:
    """Deterministic report payloads plus the typed evaluation result."""

    evaluation: EvaluationResult
    expected_primary: ExpectedPrimaryConfiguration
    timestamp_scores_parquet: bytes
    evaluated_predictions_parquet: bytes
    bootstrap_replicates_parquet: bytes
    scores_json: bytes
    scores_csv: bytes
    decision_json: bytes
    report_manifest_json: bytes
    report_markdown: bytes
    declared_live_bytes: int


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportContractError(f"{label} must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ReportContractError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _freeze_sha256(manifest: Mapping[str, Any], post_freeze_fields: set[str]) -> str:
    frozen = {
        key: value
        for key, value in manifest.items()
        if key != "selection_freeze_sha256" and key not in post_freeze_fields
    }
    return hashlib.sha256(_canonical_json_bytes(frozen)).hexdigest()


def validate_model_and_selection_lineage(
    *,
    config: PilotConfig,
    selection_manifest: Mapping[str, Any],
    benchmark_model_manifest: Mapping[str, Any],
    baseline_selection: Mapping[str, Any],
    baseline_model_manifest: Mapping[str, Any],
) -> None:
    """Validate both frozen selections and their serialized model inventories."""

    quantiles = list(config.evaluation.quantiles)
    if quantiles != list(LOCKED_QUANTILES):
        raise ReportContractError("configured quantile grid is not the locked grid")
    if selection_manifest.get("quantiles") != quantiles:
        raise ReportContractError("benchmark quantiles drifted from the frozen protocol")
    if selection_manifest.get("selection_role") != "indistinguishable_primary":
        raise ReportContractError("benchmark selection role is invalid")
    if selection_manifest.get("seed_policy") != "equal_weight_mean_no_best_seed":
        raise ReportContractError("benchmark seed selection policy is invalid")
    if selection_manifest.get("selection_data") != ["train", "validation"]:
        raise ReportContractError("benchmark selection data policy is invalid")
    if selection_manifest.get("refit") != "heads_only_on_train_plus_validation":
        raise ReportContractError("benchmark refit policy is invalid")
    freeze = _require_sha256(
        selection_manifest.get("selection_freeze_sha256"),
        "benchmark selection freeze",
    )
    recomputed = _freeze_sha256(
        selection_manifest,
        {
            "holdout_audit",
            "prerequisite_manifest_hashes_after_freeze",
            "baseline_predictions",
            "scientific_decision",
        },
    )
    if freeze != recomputed:
        raise ReportContractError("benchmark selection freeze checksum is invalid")
    holdout = _require_object(selection_manifest.get("holdout_audit"), "holdout audit")
    if (
        holdout.get("opened_after_selection_freeze_sha256") != freeze
        or holdout.get("materialization_count") != 1
        or holdout.get("prediction_passes") != 1
    ):
        raise ReportContractError("benchmark holdout evidence is inconsistent")
    if benchmark_model_manifest.get("selection_freeze_sha256") != freeze:
        raise ReportContractError("benchmark model selection lineage is inconsistent")
    _require_sha256(benchmark_model_manifest.get("models_sha256"), "benchmark models")
    if (
        benchmark_model_manifest.get("schema_version") != "1.0"
        or benchmark_model_manifest.get("format") != "python_pickle_protocol_5"
        or benchmark_model_manifest.get("serialization_roundtrip_verified") is not True
        or benchmark_model_manifest.get("holdout_reinference_for_serialization") is not False
    ):
        raise ReportContractError("benchmark model manifest policy is invalid")
    expected_keys = {
        f"{control}:seed={seed}"
        for control in ("distinguishable", "indistinguishable")
        for seed in config.seeds.circuit_seeds
    }
    model_keys = benchmark_model_manifest.get("model_keys")
    if (
        not isinstance(model_keys, list)
        or len(model_keys) != len(set(model_keys))
        or set(model_keys) != expected_keys
    ):
        raise ReportContractError("benchmark model key inventory is invalid")

    if baseline_selection.get("quantiles") != quantiles:
        raise ReportContractError("baseline quantiles drifted from the frozen protocol")
    if baseline_selection.get("quantile_columns") != list(quantile_columns(LOCKED_QUANTILES)):
        raise ReportContractError("baseline quantile column policy is invalid")
    baseline_freeze = _require_sha256(
        baseline_selection.get("selection_freeze_sha256"), "baseline selection freeze"
    )
    if baseline_freeze != _freeze_sha256(baseline_selection, set()):
        raise ReportContractError("baseline selection freeze checksum is invalid")
    if (
        baseline_selection.get("holdout_access")
        != "benchmark_only_after_photonic_freeze"
        or baseline_selection.get("test_prediction_passes") != 0
    ):
        raise ReportContractError("baseline development-only policy is inconsistent")
    if baseline_model_manifest.get("selection_freeze_sha256") != baseline_freeze:
        raise ReportContractError("baseline model selection lineage is inconsistent")
    _require_sha256(baseline_model_manifest.get("models_sha256"), "baseline models")
    selected_models = baseline_selection.get("selected")
    if not isinstance(selected_models, dict) or set(selected_models) != {
        "persistence",
        "raw",
        "rff",
        "xgboost",
    }:
        raise ReportContractError("baseline selected model inventory is invalid")
    baseline_models = baseline_model_manifest.get("models")
    if (
        not isinstance(baseline_models, list)
        or len(baseline_models) != len(set(baseline_models))
        or set(baseline_models) != set(selected_models)
    ):
        raise ReportContractError("baseline model key inventory is invalid")


def _validate_schema(frame: pd.DataFrame, schema: pa.Schema, label: str) -> None:
    if list(frame.columns) != schema.names:
        raise ReportContractError(f"{label} columns do not match the frozen schema")
    try:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise ReportContractError(f"{label} violates the frozen Arrow schema") from error
    if table.schema.remove_metadata() != schema:
        raise ReportContractError(f"{label} physical schema does not match the frozen schema")


def expected_primary_configuration(
    config: PilotConfig,
    selection_manifest: Mapping[str, Any],
    circuit_manifest: Mapping[str, Any],
    state_spaces: Mapping[str, Any],
) -> tuple[ExpectedPrimaryConfiguration, str]:
    """Bind Task 9 evaluation to the exact Task 8 selection and five circuits."""

    selected = _require_object(selection_manifest.get("selected"), "selected configuration")
    try:
        modes = int(selected["modes"])
        photons = int(selected["photons"])
        phase_scale = float(selected["phase_scale"])
        alpha = float(selected["alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReportContractError("selected primary configuration is invalid") from error
    configured_seeds = tuple(config.seeds.circuit_seeds)
    raw_seeds = selection_manifest.get("circuit_seeds")
    if not isinstance(raw_seeds, list) or tuple(raw_seeds) != configured_seeds:
        raise ReportContractError("benchmark circuit seeds do not match the frozen config")
    circuits = circuit_manifest.get("circuits")
    if not isinstance(circuits, list):
        raise ReportContractError("circuit manifest has no circuit inventory")
    selected_circuits: dict[int, str] = {}
    selected_state_orders: set[str] = set()
    for raw in circuits:
        record = _require_object(raw, "circuit record")
        if record.get("modes") != modes or record.get("photons") != photons:
            continue
        try:
            seed = int(record["circuit_seed"])
            checksum = str(record["circuit_sha256"])
            state_order = str(record["state_order_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ReportContractError("selected circuit metadata is invalid") from error
        if seed in selected_circuits:
            raise ReportContractError("selected circuit inventory contains duplicate seeds")
        selected_circuits[seed] = checksum
        selected_state_orders.add(state_order)
        _require_sha256(checksum, "selected circuit checksum")
        _require_sha256(state_order, "selected state-order checksum")
        _require_sha256(record.get("npz_sha256"), "selected circuit NPZ checksum")
    if set(selected_circuits) != set(configured_seeds):
        raise ReportContractError("selected circuit inventory is not the exact five-seed set")
    if len(selected_state_orders) != 1:
        raise ReportContractError("selected circuits disagree on state ordering")
    spaces = state_spaces.get("spaces")
    if not isinstance(spaces, list):
        raise ReportContractError("state-space manifest has no spaces")
    matches = [
        item
        for item in spaces
        if isinstance(item, dict)
        and item.get("modes") == modes
        and item.get("photons") == photons
    ]
    if len(matches) != 1:
        raise ReportContractError("selected state-space identity is missing or ambiguous")
    state_order_sha256 = matches[0].get("state_order_sha256")
    _require_sha256(state_order_sha256, "selected state-order checksum")
    if state_order_sha256 not in selected_state_orders:
        raise ReportContractError("circuit and state-space order checksums disagree")
    expected = ExpectedPrimaryConfiguration(
        modes=modes,
        photons=photons,
        phase_scale=phase_scale,
        alpha=alpha,
        circuit_sha256_by_seed=tuple(
            (seed, selected_circuits[seed]) for seed in configured_seeds
        ),
    )
    return expected, str(state_order_sha256)


def resource_gate_evidence(
    ledgers: Mapping[str, Mapping[str, Any]],
    *,
    config: PilotConfig,
) -> ResourceGateEvidence:
    """Construct complete resource evidence only from authoritative stage ledgers."""

    expected_stages = (
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
    )
    if set(ledgers) != set(expected_stages):
        raise ReportContractError("resource evidence must contain the exact five stages")
    passed: dict[str, bool] = {}
    expected_memory = int(config.resources.max_memory_gb * 1024**3)
    for stage in expected_stages:
        ledger = ledgers.get(stage)
        if ledger is None:
            raise ReportContractError(f"resource ledger is missing for {stage}")
        if (
            ledger.get("schema_version") != "1.0"
            or ledger.get("stage") != stage
            or ledger.get("status") != "success"
            or ledger.get("resource_status") != "WITHIN_LIMIT"
            or ledger.get("cpu_only") is not True
        ):
            raise ReportContractError(f"resource ledger policy is invalid for {stage}")
        numeric = (
            "wall_clock_seconds",
            "peak_rss_bytes",
            "peak_declared_bytes",
            "max_wall_clock_seconds",
            "max_memory_bytes",
        )
        for field in numeric:
            value = ledger.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0
            ):
                raise ReportContractError(
                    f"resource ledger field {field} is invalid for {stage}"
                )
        if (
            float(ledger["max_wall_clock_seconds"])
            != float(config.resources.max_wall_clock_seconds)
            or int(ledger["max_memory_bytes"]) != expected_memory
        ):
            raise ReportContractError(f"resource ledger limits drifted for {stage}")
        if float(ledger["wall_clock_seconds"]) > float(
            ledger["max_wall_clock_seconds"]
        ):
            raise ReportContractError(f"resource ledger wall-clock cap was exceeded for {stage}")
        for field in ("peak_rss_bytes", "peak_declared_bytes"):
            if float(ledger[field]) > float(ledger["max_memory_bytes"]):
                raise ReportContractError(
                    f"resource ledger memory cap was exceeded by {field} for {stage}"
                )
        component_fields = (
            "peak_parent_rss_bytes",
            "peak_child_tree_rss_bytes",
            "peak_combined_rss_bytes",
        )
        present_components = {field for field in component_fields if field in ledger}
        subprocess_stages = {"train-baselines", "benchmark"}
        expected_components = set(component_fields) if stage in subprocess_stages else set()
        if present_components != expected_components:
            raise ReportContractError(
                f"resource ledger component schema is invalid for {stage}"
            )
        if expected_components:
            for field in component_fields:
                value = ledger[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(float(value))
                    or float(value) < 0
                    or float(value) > float(ledger["max_memory_bytes"])
                ):
                    raise ReportContractError(
                        f"resource ledger component {field} is invalid for {stage}"
                    )
            combined = float(ledger["peak_combined_rss_bytes"])
            # These are independently sampled maxima; their sum need not be
            # contemporaneous. Each component must still fit under combined peak.
            if any(
                float(ledger[field]) > combined
                for field in (
                    "peak_parent_rss_bytes",
                    "peak_child_tree_rss_bytes",
                    "peak_rss_bytes",
                )
            ):
                raise ReportContractError(
                    f"resource ledger combined RSS is inconsistent for {stage}"
                )
        passed[stage] = True
    return ResourceGateEvidence.from_mapping(passed)


def _timestamp_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("target_ac_power_kw", pa.float64(), nullable=False),
        pa.field("capacity_kw", pa.float64(), nullable=False),
        pa.field("quantum_success", pa.bool_(), nullable=False),
        pa.field("fallback_used", pa.bool_(), nullable=False),
        pa.field("fallback_success", pa.bool_(), nullable=False),
        pa.field("quantum_loss", pa.float64(), nullable=True),
        pa.field("system_loss", pa.float64(), nullable=True),
        pa.field("baseline_loss", pa.float64(), nullable=False),
    ]
    for scope in ("quantum", "system"):
        for metric in ("coverage", "width"):
            for coverage in (50, 80, 90):
                fields.append(
                    pa.field(f"{scope}_{metric}_{coverage}", pa.float64(), nullable=True)
                )
    return pa.schema(fields)


def _audit_schema() -> pa.Schema:
    return pa.schema(
        [
            *exact_prediction_arrow_schema(),
            pa.field("seed_quantum_loss", pa.float64(), nullable=True),
            pa.field("timestamp_quantum_loss", pa.float64(), nullable=True),
            pa.field("system_loss", pa.float64(), nullable=True),
            pa.field("matched_baseline_loss", pa.float64(), nullable=False),
            pa.field("quantum_success", pa.bool_(), nullable=False),
            pa.field("fallback_used", pa.bool_(), nullable=False),
            pa.field("fallback_success", pa.bool_(), nullable=False),
        ]
    )


def _bootstrap_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("scope", pa.string(), nullable=False),
            pa.field("block_days", pa.int32(), nullable=False),
            pa.field("replicate", pa.int32(), nullable=False),
            pa.field("sample_count", pa.int32(), nullable=False),
            pa.field("system_sample_count", pa.int32(), nullable=False),
            pa.field("skill", pa.float64(), nullable=False),
        ]
    )


def report_parquet_schemas() -> dict[str, pa.Schema]:
    """Return the fixed schemas for worker-produced report Parquet artifacts."""

    return {
        "timestamp_scores.parquet": _timestamp_schema(),
        "evaluated_predictions.parquet": _audit_schema(),
        "bootstrap_replicates.parquet": _bootstrap_schema(),
    }


def _parquet_bytes(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    normalized = frame.copy()
    for field in schema:
        if field.name not in normalized:
            normalized[field.name] = pd.Series(dtype="object")
    normalized = normalized.loc[:, schema.names]
    for field in schema:
        if pa.types.is_floating(field.type) and field.nullable:
            normalized[field.name] = normalized[field.name].where(
                np.isfinite(pd.to_numeric(normalized[field.name], errors="coerce")), None
            )
    table = pa.Table.from_pandas(
        normalized, schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def _audit_predictions(
    exact: pd.DataFrame, per_timestamp: pd.DataFrame
) -> pd.DataFrame:
    primary = exact.loc[exact["control"].astype(str) == "indistinguishable"].copy()
    lookup = per_timestamp.set_index("sample_id", drop=False)
    q_columns = quantile_columns(LOCKED_QUANTILES)
    records: list[dict[str, Any]] = []
    for row in primary.sort_values(
        ["site_id", "issued_at_utc", "sample_id", "circuit_seed"], kind="stable"
    ).itertuples(index=False):
        timestamp = lookup.loc[str(row.sample_id)]
        success = str(row.status).lower() == "success"
        seed_loss: float | None = None
        if success:
            seed_loss = float(
                timestamp_normalized_wis(
                    np.asarray([float(row.target_ac_power_kw)]),
                    np.asarray([[float(getattr(row, name)) for name in q_columns]]),
                    np.asarray([float(row.capacity_kw)]),
                    LOCKED_QUANTILES,
                )[0]
            )
        record = row._asdict()
        record.update(
            {
                "seed_quantum_loss": seed_loss,
                "timestamp_quantum_loss": timestamp["quantum_loss"],
                "system_loss": timestamp["system_loss"],
                "matched_baseline_loss": timestamp["baseline_loss"],
                "quantum_success": bool(timestamp["quantum_success"]),
                "fallback_used": bool(timestamp["fallback_used"]),
                "fallback_success": bool(timestamp["fallback_success"]),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _validate_physical_quantiles(frame: pd.DataFrame, label: str) -> None:
    q_columns = quantile_columns(LOCKED_QUANTILES)
    values = frame.loc[:, q_columns].to_numpy(dtype=np.float64)
    capacity = frame["capacity_kw"].to_numpy(dtype=np.float64)
    targets = frame["target_ac_power_kw"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(capacity).all()
        or not np.isfinite(targets).all()
        or (capacity <= 0).any()
        or (targets < 0).any()
        or (targets > capacity).any()
        or (values < 0).any()
        or (values > capacity[:, None]).any()
    ):
        raise ReportContractError(f"{label} target/capacity/quantile range is invalid")
    if (np.diff(values, axis=1) < 0).any():
        raise ReportContractError(f"{label} quantiles are not monotone")


def _physical_control_summary(
    exact: pd.DataFrame, expected: ExpectedPrimaryConfiguration, matched_alpha: float
) -> dict[str, Any]:
    controls = sorted(set(exact["control"].astype(str)))
    if controls != ["distinguishable", "indistinguishable"]:
        raise ReportContractError("exact predictions do not contain both physical controls")
    counts = {control: int((exact["control"] == control).sum()) for control in controls}
    if len(set(counts.values())) != 1:
        raise ReportContractError("physical controls are not matched row-for-row")
    if set(exact["status"].astype(str)) != {"success"}:
        raise ReportContractError("physical control status must be successful")
    if set(exact["fit_phase"].astype(str)) != {"train_plus_validation_refit"}:
        raise ReportContractError("physical control fit phase is invalid")
    _validate_physical_quantiles(exact, "physical control")
    identity = [
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "target_ac_power_kw",
        "capacity_kw",
        "circuit_seed",
        "modes",
        "photons",
        "phase_scale",
        "circuit_sha256",
    ]
    memberships: list[set[tuple[Any, ...]]] = []
    for control in controls:
        subset = exact.loc[exact["control"] == control]
        if subset.duplicated(identity).any():
            raise ReportContractError("physical control contains duplicate identity rows")
        memberships.append(
            set(subset.loc[:, identity].itertuples(index=False, name=None))
        )
    if memberships[0] != memberships[1]:
        raise ReportContractError("physical controls do not share an exact identity grid")
    primary = exact.loc[exact["control"] == "indistinguishable"]
    control = exact.loc[exact["control"] == "distinguishable"]
    if (
        set(primary["modes"].astype(int)) != {expected.modes}
        or set(primary["photons"].astype(int)) != {expected.photons}
        or set(primary["phase_scale"].astype(float)) != {expected.phase_scale}
        or set(primary["alpha"].astype(float)) != {expected.alpha}
        or set(control["alpha"].astype(float)) != {matched_alpha}
    ):
        raise ReportContractError("physical control frozen configuration drifted")
    checksums = expected.circuit_checksums
    if any(
        str(row.circuit_sha256) != checksums.get(int(row.circuit_seed))
        for row in exact.itertuples(index=False)
    ):
        raise ReportContractError("physical control circuit identity drifted")
    return {
        "controls": controls,
        "row_counts": counts,
        "interpretation": "matched physical controls only; no advantage claim",
    }


def _finite_shot_summary(
    finite: pd.DataFrame,
    exact: pd.DataFrame,
    config: PilotConfig,
    expected: ExpectedPrimaryConfiguration,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_schema(finite, finite_shot_prediction_arrow_schema(), "finite-shot predictions")
    required_budgets = set(config.photonics.raw_attempt_budgets)
    budgets = set(pd.to_numeric(finite["budget"], errors="raise").astype(int))
    if budgets != required_budgets:
        raise ReportContractError("finite-shot budgets do not match the frozen protocol")
    if finite.empty or not finite["ideal_shot_noise_only"].astype(bool).all():
        raise ReportContractError("finite-shot rows are not ideal-shot-noise-only")
    if finite["loss_modeled"].astype(bool).any():
        raise ReportContractError("finite-shot rows unexpectedly model optical loss")
    if not (
        finite["raw_attempts"].to_numpy(dtype=np.int64)
        == finite["budget"].to_numpy(dtype=np.int64)
    ).all() or not (
        finite["accepted_samples"].to_numpy(dtype=np.int64)
        == finite["budget"].to_numpy(dtype=np.int64)
    ).all():
        raise ReportContractError("finite-shot accepted/raw counts must equal the budget")
    if set(finite["status"].astype(str)) != {"success"}:
        raise ReportContractError("finite-shot status must be successful")
    if (
        set(finite["modes"].astype(int)) != {expected.modes}
        or set(finite["photons"].astype(int)) != {expected.photons}
        or set(finite["phase_scale"].astype(float)) != {expected.phase_scale}
        or set(finite["alpha"].astype(float)) != {expected.alpha}
    ):
        raise ReportContractError("finite-shot frozen configuration drifted")
    _validate_physical_quantiles(finite, "finite-shot")
    identity = ["sample_id", "site_id", "issued_at_utc", "target_at_utc"]
    primary = exact.loc[exact["control"] == "indistinguishable"]
    exact_membership = set(
        primary.loc[:, [*identity, "circuit_seed"]].itertuples(index=False, name=None)
    )
    finite_membership = set(
        finite.loc[:, [*identity, "circuit_seed"]].itertuples(index=False, name=None)
    )
    if exact_membership != finite_membership:
        raise ReportContractError("finite-shot rows do not match exact-primary membership")
    key = [*identity, "circuit_seed", "budget"]
    if finite.duplicated(key).any() or len(finite) != len(exact_membership) * len(budgets):
        raise ReportContractError("finite-shot sample/seed/budget grid is incomplete")
    checksums = expected.circuit_checksums
    if any(
        str(row.circuit_sha256) != checksums[int(row.circuit_seed)]
        for row in finite.itertuples(index=False)
    ):
        raise ReportContractError("finite-shot circuit identity drifted from exact primary")
    primary_identity = primary.set_index([*identity, "circuit_seed"])
    finite_identity = finite.set_index([*identity, "circuit_seed"])
    for field in ("target_ac_power_kw", "capacity_kw"):
        expected_values = primary_identity[field].sort_index()
        actual_values = finite_identity.groupby(level=list(range(5)))[field].first().sort_index()
        if not expected_values.equals(actual_values):
            raise ReportContractError(f"finite-shot {field} drifted from exact primary")
    finite_diagnostics = diagnostics.get("finite_shot")
    if not isinstance(finite_diagnostics, dict) or (
        finite_diagnostics.get("ideal_shot_noise_only") is not True
        or finite_diagnostics.get("used_for_go_stop") is not False
        or finite_diagnostics.get("loss_modeled") is not False
    ):
        raise ReportContractError("finite-shot decision firewall metadata is invalid")
    return {
        "budgets": sorted(budgets),
        "row_count": len(finite),
        "ideal_shot_noise_only": True,
        "loss_modeled": False,
        "used_for_go_stop": False,
        "role": "diagnostic_only",
    }


def _csv_bytes(bundle: ScoreBundle) -> bytes:
    row = score_bundle_csv_row(bundle)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().replace("\r\n", "\n").encode("utf-8")


def _render_markdown(
    bundle: ScoreBundle,
    *,
    provenance: str,
    finite_shot: Mapping[str, Any],
) -> str:
    def metric(value: float | None) -> str:
        return "NA" if value is None else f"{value:.8f}"

    decision = bundle.decision.value.upper()
    synthetic = provenance == "synthetic"
    eligibility = (
        "Synthetic software validation only; this is not scientific evidence."
        if synthetic
        else "Scientific eligibility is determined only by the frozen protocol and gates."
    )
    return "\n".join(
        (
            "# Photonic PV Pilot - G0-G3 Decision Report",
            "",
            f"**Status:** {bundle.stage_status.value.upper()} / {decision}",
            "",
            eligibility,
            "",
            "## Exact primary evaluation",
            "",
            f"- Quantum-only WIS/capacity: {metric(bundle.quantum_only_score)}",
            f"- System WIS/capacity (Quantum + fallback): {metric(bundle.system_score)}",
            f"- Matched baseline WIS/capacity: {metric(bundle.system_baseline_score)}",
            f"- Quantum success rate: {bundle.success_rate:.6f}",
            f"- Fallback rate: {bundle.fallback_rate:.6f}",
            "",
            "Quantum-only, System, and Matched baseline results are reported separately and are "
            "paired on the audit rows defined by the frozen protocol.",
            "",
            "## Finite-shot diagnostic",
            "",
            f"- Budgets: {finite_shot['budgets']}",
            "- Flag: ideal_shot_noise_only",
            "- This diagnostic is excluded from GO/STOP.",
            "",
            "## Scope and limitations",
            "",
            "- CPU/SLOS simulation only; no QPU execution.",
            "- No optical loss or mitigation model.",
            "- G4/G5 are outside this pilot.",
            "- Distinguishable and indistinguishable outputs are physical controls, "
            "not proof of advantage.",
            "- There is no quantum-advantage claim.",
            "",
            f"Decision rationale: `{bundle.conclusion}`.",
            "",
        )
    )


def build_report_products(
    *,
    config: PilotConfig,
    exact_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    finite_shot_predictions: pd.DataFrame,
    selection_manifest: Mapping[str, Any],
    circuit_manifest: Mapping[str, Any],
    state_spaces: Mapping[str, Any],
    benchmark_diagnostics: Mapping[str, Any],
    benchmark_model_manifest: Mapping[str, Any],
    resource_ledgers: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    versions: Mapping[str, Any],
    baseline_selection: Mapping[str, Any],
    baseline_model_manifest: Mapping[str, Any],
    initial_live_bytes: int,
    resource_checkpoint: ResourceCheckpoint,
) -> ReportProducts:
    """Evaluate and serialize a report without fitting or reading source data."""

    _validate_schema(exact_predictions, exact_prediction_arrow_schema(), "exact predictions")
    _validate_schema(baseline_predictions, prediction_arrow_schema(), "baseline predictions")
    model_payloads = {
        "baseline": (
            baseline_model_manifest,
            "train-baselines/models/baseline_models.pkl",
        ),
        "benchmark": (
            benchmark_model_manifest,
            "benchmark/models/photonic_models.pkl",
        ),
    }
    for label, (manifest, path) in model_payloads.items():
        captured_digest = input_hashes.get(path)
        if (
            not _is_sha256(captured_digest)
            or manifest.get("models_sha256") != captured_digest
        ):
            raise ReportContractError(
                f"{label} models checksum does not match the captured serialized model"
            )
    validate_model_and_selection_lineage(
        config=config,
        selection_manifest=selection_manifest,
        benchmark_model_manifest=benchmark_model_manifest,
        baseline_selection=baseline_selection,
        baseline_model_manifest=baseline_model_manifest,
    )
    expected, state_order_sha256 = expected_primary_configuration(
        config, selection_manifest, circuit_manifest, state_spaces
    )
    baseline_link = _require_object(
        selection_manifest.get("baseline_predictions"), "baseline prediction lineage"
    )
    baseline_path = "benchmark/baseline_predictions.parquet"
    if (
        baseline_link.get("path") != baseline_path
        or baseline_link.get("sha256") != input_hashes.get(baseline_path)
        or baseline_link.get("policy") != "same_authenticated_holdout_after_photonic_freeze"
    ):
        raise ReportContractError("baseline prediction lineage is inconsistent")
    expected_model_keys = {
        f"{control}:seed={seed}"
        for control in ("distinguishable", "indistinguishable")
        for seed in config.seeds.circuit_seeds
    }
    exact_holdout = benchmark_diagnostics.get("exact_holdout")
    if (
        benchmark_diagnostics.get("ensemble_prediction") is not False
        or benchmark_diagnostics.get("model_count") != len(expected_model_keys)
        or benchmark_diagnostics.get("exact_prediction_rows") != len(exact_predictions)
        or not isinstance(exact_holdout, dict)
        or set(exact_holdout) != expected_model_keys
    ):
        raise ReportContractError("benchmark diagnostics/model/holdout evidence is inconsistent")
    gates = resource_gate_evidence(resource_ledgers, config=config)
    if initial_live_bytes < 0:
        raise ReportContractError("initial live-byte evidence is invalid")
    declared = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in (exact_predictions, baseline_predictions, finite_shot_predictions)
    )
    evaluation_base = initial_live_bytes + declared

    def evaluation_checkpoint(additional_bytes: int = 0) -> None:
        resource_checkpoint(evaluation_base + additional_bytes)

    resource_checkpoint(evaluation_base)
    policy = EvaluationPolicy.from_config(config)
    evaluation = evaluate_predictions(
        exact_predictions,
        baseline_predictions,
        policy=policy,
        expected_primary=expected,
        seed_manager=SeedManager(config.seeds.master_seed),
        resource_gates=gates,
        finite_shot_predictions=finite_shot_predictions,
        resource_checkpoint=evaluation_checkpoint,
    )
    matched = _require_object(selection_manifest.get("matched_control"), "matched control")
    try:
        matched_alpha = float(matched["selected_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReportContractError("matched-control alpha is invalid") from error
    if (
        int(matched.get("modes", -1)) != expected.modes
        or float(matched.get("phase_scale", float("nan"))) != expected.phase_scale
        or matched.get("alpha_policy")
        != "separate_validation_common_alpha_all_seeds_fixed_primary_m_c"
    ):
        raise ReportContractError("matched-control selection policy is invalid")
    physical_controls = _physical_control_summary(
        exact_predictions, expected, matched_alpha
    )
    finite_summary = _finite_shot_summary(
        finite_shot_predictions,
        exact_predictions,
        config,
        expected,
        benchmark_diagnostics,
    )
    audit = _audit_predictions(exact_predictions, evaluation.per_timestamp)
    frame_live_bytes = initial_live_bytes + declared + sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in (evaluation.per_timestamp, audit, evaluation.bootstrap_replicates)
    )
    resource_checkpoint(frame_live_bytes)
    serialized_live_bytes = 0

    def serialize_parquet(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
        nonlocal serialized_live_bytes
        frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        # The source pandas frame is already included in frame_live_bytes. Reserve
        # one additional frame-sized buffer for Arrow materialization and another
        # for the Parquet output before entering the serializer.
        resource_checkpoint(frame_live_bytes + serialized_live_bytes + 2 * frame_bytes)
        payload = _parquet_bytes(frame, schema)
        serialized_live_bytes += len(payload)
        resource_checkpoint(frame_live_bytes + serialized_live_bytes)
        return payload

    timestamp_bytes = serialize_parquet(
        evaluation.per_timestamp, _timestamp_schema()
    )
    audit_bytes = serialize_parquet(audit, _audit_schema())
    bootstrap_bytes = serialize_parquet(
        evaluation.bootstrap_replicates, _bootstrap_schema()
    )
    bundle = evaluation.score_bundle
    scores_data = deterministic_data(bundle)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    scores_json = _canonical_json_bytes(scores_data)
    serialized_live_bytes += len(scores_json)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    scores_csv = _csv_bytes(bundle)
    serialized_live_bytes += len(scores_csv)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    decision_data = {
        "schema_version": "1.0",
        "stage_status": bundle.stage_status.value,
        "scientific_decision": bundle.decision.value,
        "conclusion": bundle.conclusion,
        "science_eligible": bundle.science_eligible,
        "decision_basis": "exact_primary_system_score_calibration_resource_gates",
        "finite_shot_used": False,
    }
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    decision_json = _canonical_json_bytes(decision_data)
    serialized_live_bytes += len(decision_json)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    provenance_object = dict(provenance)
    report_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline_scope": "G0-G3",
        "protocol_version": config.protocol.version,
        "data_provenance": config.protocol.provenance.value,
        "protocol_approved_by": config.protocol.approved_by,
        "protocol_approved_at": (
            config.protocol.approved_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z")
            if config.protocol.approved_at is not None
            else None
        ),
        "versions": dict(versions),
        "scientific_decision": bundle.decision.value,
        "conclusion": bundle.conclusion,
        "provenance": provenance_object,
        "artifact_input_hashes": dict(sorted(input_hashes.items())),
        "selected_primary": {
            "modes": expected.modes,
            "photons": expected.photons,
            "phase_scale": expected.phase_scale,
            "alpha": expected.alpha,
            "circuit_sha256_by_seed": {
                str(seed): digest for seed, digest in expected.circuit_sha256_by_seed
            },
            "state_order_sha256": state_order_sha256,
            "selection_freeze_sha256": selection_manifest.get("selection_freeze_sha256"),
        },
        "model_selection": {
            "baseline": baseline_selection.get("selected"),
            "photonic_primary": selection_manifest.get("selected"),
            "photonic_matched_control": selection_manifest.get("matched_control"),
        },
        "scores": scores_data,
        "resource_gates": deterministic_data(gates),
        "resource_usage": {
            stage: dict(ledger) for stage, ledger in sorted(resource_ledgers.items())
        },
        "physical_controls": physical_controls,
        "finite_shot": finite_summary,
        "limitations": [
            "no_qpu_execution",
            "no_optical_loss_model",
            "no_mitigation",
            "g4_g5_not_implemented",
            "no_quantum_advantage_claim",
        ],
    }
    report_markdown_text = _render_markdown(
        bundle,
        provenance=config.protocol.provenance.value,
        finite_shot=finite_summary,
    )
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    report_manifest_json = _canonical_json_bytes(report_manifest)
    serialized_live_bytes += len(report_manifest_json)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    report_markdown = report_markdown_text.encode("utf-8")
    serialized_live_bytes += len(report_markdown)
    resource_checkpoint(frame_live_bytes + serialized_live_bytes)
    return ReportProducts(
        evaluation=evaluation,
        expected_primary=expected,
        timestamp_scores_parquet=timestamp_bytes,
        evaluated_predictions_parquet=audit_bytes,
        bootstrap_replicates_parquet=bootstrap_bytes,
        scores_json=scores_json,
        scores_csv=scores_csv,
        decision_json=decision_json,
        report_manifest_json=report_manifest_json,
        report_markdown=report_markdown,
        declared_live_bytes=frame_live_bytes + serialized_live_bytes,
    )


__all__ = [
    "ReportContractError",
    "ReportProducts",
    "build_report_products",
    "expected_primary_configuration",
    "report_parquet_schemas",
    "resource_gate_evidence",
]
