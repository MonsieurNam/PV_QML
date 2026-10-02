"""Read-only G3 report construction from immutable stage artifacts."""

from __future__ import annotations

import csv
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
    report_manifest: dict[str, Any]
    report_markdown: str


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
) -> ResourceGateEvidence:
    """Construct complete resource evidence only from authoritative stage ledgers."""

    passed: dict[str, bool] = {}
    for stage in (
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
    ):
        ledger = ledgers.get(stage)
        if ledger is None:
            raise ReportContractError(f"resource ledger is missing for {stage}")
        status = str(ledger.get("status", "")).lower()
        resource_status = ledger.get("resource_status")
        passed[stage] = status == "success" and resource_status == "WITHIN_LIMIT"
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


def _physical_control_summary(exact: pd.DataFrame) -> dict[str, Any]:
    controls = sorted(set(exact["control"].astype(str)))
    if controls != ["distinguishable", "indistinguishable"]:
        raise ReportContractError("exact predictions do not contain both physical controls")
    counts = {control: int((exact["control"] == control).sum()) for control in controls}
    if len(set(counts.values())) != 1:
        raise ReportContractError("physical controls are not matched row-for-row")
    identity = [
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
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
    resource_ledgers: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    versions: Mapping[str, Any],
    baseline_selection: Mapping[str, Any],
    resource_checkpoint: ResourceCheckpoint,
) -> ReportProducts:
    """Evaluate and serialize a report without fitting or reading source data."""

    _validate_schema(exact_predictions, exact_prediction_arrow_schema(), "exact predictions")
    _validate_schema(baseline_predictions, prediction_arrow_schema(), "baseline predictions")
    expected, state_order_sha256 = expected_primary_configuration(
        config, selection_manifest, circuit_manifest, state_spaces
    )
    gates = resource_gate_evidence(resource_ledgers)
    declared = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in (exact_predictions, baseline_predictions, finite_shot_predictions)
    )
    resource_checkpoint(declared)
    policy = EvaluationPolicy.from_config(config)
    evaluation = evaluate_predictions(
        exact_predictions,
        baseline_predictions,
        policy=policy,
        expected_primary=expected,
        seed_manager=SeedManager(config.seeds.master_seed),
        resource_gates=gates,
        finite_shot_predictions=finite_shot_predictions,
        resource_checkpoint=resource_checkpoint,
    )
    physical_controls = _physical_control_summary(exact_predictions)
    finite_summary = _finite_shot_summary(
        finite_shot_predictions,
        exact_predictions,
        config,
        expected,
        benchmark_diagnostics,
    )
    audit = _audit_predictions(exact_predictions, evaluation.per_timestamp)
    timestamp_bytes = _parquet_bytes(evaluation.per_timestamp, _timestamp_schema())
    audit_bytes = _parquet_bytes(audit, _audit_schema())
    bootstrap_bytes = _parquet_bytes(evaluation.bootstrap_replicates, _bootstrap_schema())
    resource_checkpoint(declared + len(timestamp_bytes) + len(audit_bytes) + len(bootstrap_bytes))
    bundle = evaluation.score_bundle
    scores_data = deterministic_data(bundle)
    scores_json = _canonical_json_bytes(scores_data)
    scores_csv = _csv_bytes(bundle)
    decision_data = {
        "schema_version": "1.0",
        "stage_status": bundle.stage_status.value,
        "scientific_decision": bundle.decision.value,
        "conclusion": bundle.conclusion,
        "science_eligible": bundle.science_eligible,
        "decision_basis": "exact_primary_system_score_calibration_resource_gates",
        "finite_shot_used": False,
    }
    decision_json = _canonical_json_bytes(decision_data)
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
    report_markdown = _render_markdown(
        bundle,
        provenance=config.protocol.provenance.value,
        finite_shot=finite_summary,
    )
    for payload in (
        scores_json,
        scores_csv,
        decision_json,
        _canonical_json_bytes(report_manifest),
        report_markdown.encode("utf-8"),
    ):
        resource_checkpoint(declared + len(payload))
    return ReportProducts(
        evaluation=evaluation,
        expected_primary=expected,
        timestamp_scores_parquet=timestamp_bytes,
        evaluated_predictions_parquet=audit_bytes,
        bootstrap_replicates_parquet=bootstrap_bytes,
        scores_json=scores_json,
        scores_csv=scores_csv,
        decision_json=decision_json,
        report_manifest=report_manifest,
        report_markdown=report_markdown,
    )


__all__ = [
    "ReportContractError",
    "ReportProducts",
    "build_report_products",
    "expected_primary_configuration",
    "resource_gate_evidence",
]
