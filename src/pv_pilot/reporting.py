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

from .baselines import LOCKED_QUANTILES, normalized_wis, prediction_arrow_schema, quantile_columns
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
from .three_photon import (
    DIAGNOSTIC_PATHS,
    PROFILE_FIXED_PATHS,
    PROFILE_PREFIX,
    require_profile_pass,
)

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


@dataclass(frozen=True, slots=True)
class _DiagnosticControlConfiguration:
    """Frozen three-photon control identity, separate from the primary contract."""

    modes: int
    photons: int
    phase_scale: float
    alpha: float
    circuit_sha256_by_seed: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if (
            self.modes not in {4, 6}
            or self.photons != 3
            or not np.isfinite(self.phase_scale)
            or self.phase_scale <= 0
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
            or len(self.circuit_sha256_by_seed) != 5
            or len({seed for seed, _digest in self.circuit_sha256_by_seed}) != 5
            or any(not _is_sha256(digest) for _seed, digest in self.circuit_sha256_by_seed)
        ):
            raise ReportContractError("diagnostic control configuration is invalid")

    @property
    def circuit_checksums(self) -> dict[int, str]:
        return dict(self.circuit_sha256_by_seed)


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


def _holdout_freeze(config: PilotConfig, selection: Mapping[str, Any]) -> str:
    primary = _require_sha256(selection.get("selection_freeze_sha256"), "primary freeze")
    audit = _require_object(selection.get("holdout_audit"), "holdout audit")
    if not config.photonics.enable_three_photon_benchmark:
        if "selection_freezes" in audit:
            raise ReportContractError("disabled diagnostic has unexpected selection freezes")
        return primary
    freezes = _require_object(audit.get("selection_freezes"), "combined selection freezes")
    if set(freezes) != {"primary_2photon", "diagnostic_3photon"}:
        raise ReportContractError("combined selection freeze inventory is invalid")
    if freezes["primary_2photon"] != primary:
        raise ReportContractError("combined primary selection freeze is invalid")
    _require_sha256(freezes["diagnostic_3photon"], "diagnostic freeze")
    combined = hashlib.sha256(_canonical_json_bytes(freezes)).hexdigest()
    if (
        audit.get("selection_freeze_sha256") != combined
        or audit.get("opened_after_selection_freeze_sha256") != combined
        or _require_object(selection.get("baseline_predictions"), "baseline lineage").get(
            "opened_after_selection_freeze_sha256"
        ) != combined
    ):
        raise ReportContractError("combined holdout selection freeze is invalid")
    return combined


def optional_report_summary(
    config: PilotConfig,
    captured: Mapping[str, bytes],
    primary: Mapping[str, Any],
    profile_inputs: Mapping[str, str],
    *,
    primary_exact: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate the authenticated optional inventory without fitting or simulation."""
    from .photonics import (
        circuit_checksum,
        enumerate_fock_states,
        load_circuit_npz,
        state_order_checksum,
    )

    if not config.photonics.enable_three_photon_profile:
        return {"profile_status": "DISABLED", "benchmark_status": "DISABLED"}

    def object_at(path: str) -> dict[str, Any]:
        try:
            return _require_object(json.loads(captured[path]), path)
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReportContractError(f"optional artifact is invalid: {path}") from error

    gate_path, feature_path, state_path, circuit_path = PROFILE_FIXED_PATHS
    gate = object_at(gate_path)
    summary: dict[str, Any] = {
        "label": "diagnostic_3photon",
        "profile_status": gate.get("status"),
        "benchmark_status": "DISABLED",
        "used_for_go_stop": False,
        "scientific_decision": "NOT_EVALUATED",
        "resource_gate": gate,
    }
    if gate.get("status") == "FAIL" and not config.photonics.enable_three_photon_benchmark:
        if (
            gate.get("config_sha256") != config.config_hash()
            or gate.get("input_hashes") != dict(profile_inputs)
            or gate.get("holdout_opened") is not False
            or not gate.get("failure_gate")
            or object_at(state_path).get("spaces") != []
            or object_at(circuit_path).get("circuits") != []
            or not pd.read_parquet(io.BytesIO(captured[feature_path])).empty
        ):
            raise ReportContractError("failed optional profile evidence is invalid")
        return summary
    try:
        require_profile_pass(gate, config, captured[feature_path], dict(profile_inputs))
    except ValueError as error:
        raise ReportContractError(str(error)) from error
    expected_spaces = [
        {
            "modes": modes,
            "photons": 3,
            "dimension": len(enumerate_fock_states(modes, 3)),
            "states": [list(state) for state in enumerate_fock_states(modes, 3)],
            "state_order_sha256": state_order_checksum(enumerate_fock_states(modes, 3)),
        }
        for modes in config.photonics.modes
    ]
    if object_at(state_path).get("spaces") != expected_spaces:
        raise ReportContractError("optional canonical state spaces are invalid")
    records = object_at(circuit_path).get("circuits")
    if not isinstance(records, list):
        raise ReportContractError("optional circuit inventory is invalid")
    circuits: dict[tuple[int, int], str] = {}
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ReportContractError("optional circuit record is invalid")
        path = record.get("path")
        if (
            not isinstance(path, str)
            or not path.startswith(f"{PROFILE_PREFIX}/circuits/")
            or path in paths
            or path not in captured
            or hashlib.sha256(captured[path]).hexdigest() != record.get("npz_sha256")
        ):
            raise ReportContractError("optional circuit checksum/path is invalid")
        circuit = load_circuit_npz(captured[path])
        key = (circuit.modes, circuit.circuit_seed)
        checksum = circuit_checksum(circuit)
        if (
            key in circuits
            or circuit.photons != 3
            or record.get("photons") != 3
            or record.get("modes") != circuit.modes
            or record.get("circuit_seed") != circuit.circuit_seed
            or record.get("state_order_sha256") != circuit.state_order_sha256
            or record.get("circuit_sha256") != checksum
        ):
            raise ReportContractError("optional circuit identity is invalid")
        circuits[key], paths = checksum, paths | {path}
    if set(circuits) != {
        (modes, seed) for modes in config.photonics.modes for seed in config.seeds.circuit_seeds
    } or paths != {path for path in captured if path.startswith(f"{PROFILE_PREFIX}/circuits/")}:
        raise ReportContractError("optional circuit grid is incomplete")
    summary["state_dimensions"] = [space["dimension"] for space in expected_spaces]
    if not config.photonics.enable_three_photon_benchmark:
        return summary
    selection = object_at(DIAGNOSTIC_PATHS[0])
    freeze = _require_sha256(selection.get("selection_freeze_sha256"), "diagnostic freeze")
    if freeze != _freeze_sha256(selection, set()):
        raise ReportContractError("diagnostic selection freeze checksum is invalid")
    if selection.get("three_photon_profile") != {
        "status": "PASS",
        "gate_sha256": hashlib.sha256(captured[gate_path]).hexdigest(),
    }:
        raise ReportContractError("diagnostic profile gate lineage is invalid")
    combined = _holdout_freeze(config, primary)
    if primary["holdout_audit"]["selection_freezes"]["diagnostic_3photon"] != freeze:
        raise ReportContractError("diagnostic selection freeze does not match combined freeze")
    if (
        selection.get("selection_role") != "diagnostic_3photon"
        or selection.get("seed_policy") != "equal_weight_mean_no_best_seed"
        or selection.get("circuit_seeds") != list(config.seeds.circuit_seeds)
        or selection.get("selection_data") != ["train", "validation"]
        or selection.get("scaler_policy") != "fit_train_once_fixed_for_validation_refit_and_holdout"
        or selection.get("refit") != "heads_only_on_train_plus_validation"
        or selection.get("quantiles") != list(config.evaluation.quantiles)
    ):
        raise ReportContractError("diagnostic selection policy is invalid")
    selected = _require_object(selection.get("selected"), "diagnostic selected configuration")
    if selected.get("photons") != 3 or selected.get("seed_count") != 5:
        raise ReportContractError("diagnostic must keep all five three-photon seeds")
    metrics = object_at(DIAGNOSTIC_PATHS[-1])
    seed_scores = metrics.get("seed_scores")
    if (
        metrics.get("label") != "diagnostic_3photon"
        or metrics.get("used_for_go_stop") is not False
        or metrics.get("scientific_decision") != "NOT_EVALUATED"
        or metrics.get("selection_freeze_sha256") != freeze
        or metrics.get("joint_selection_freeze_sha256") != combined
        or metrics.get("seed_policy") != "equal_weight_mean_no_best_seed"
        or not isinstance(seed_scores, list)
        or len(seed_scores) != 10
    ):
        raise ReportContractError("diagnostic metrics/firewall evidence is invalid")
    model_evidence = _require_object(metrics.get("model_inventory"), "diagnostic model inventory")
    model_payload = captured.get(DIAGNOSTIC_PATHS[4])
    if (
        not isinstance(model_payload, bytes)
        or model_evidence.get("models_sha256") != hashlib.sha256(model_payload).hexdigest()
    ):
        raise ReportContractError("diagnostic model digest is invalid")
    expected_models = [
        {
            "control": control,
            "circuit_seed": seed,
            "modes": selected["modes"],
            "photons": 3,
            "phase_scale": selected["phase_scale"],
            "alpha": selected["alpha"] if control == "indistinguishable"
                     else selection["matched_control"]["selected_alpha"],
            "circuit_sha256": circuits[(int(selected["modes"]), seed)],
            "quantiles": list(LOCKED_QUANTILES),
        }
        for control in ("distinguishable", "indistinguishable")
        for seed in sorted(config.seeds.circuit_seeds)
    ]
    if (
        model_evidence.get("model_keys") != [
            f"{model['control']}:seed={model['circuit_seed']}" for model in expected_models
        ]
        or model_evidence.get("models") != expected_models
        or model_evidence.get("serialization_roundtrip_verified") is not True
        or model_evidence.get("holdout_reinference_for_serialization") is not False
    ):
        raise ReportContractError("diagnostic model inventory/frozen metadata is invalid")
    exact = pd.read_parquet(io.BytesIO(captured[DIAGNOSTIC_PATHS[3]]))
    _validate_schema(exact, exact_prediction_arrow_schema(), "diagnostic exact predictions")
    expected = _DiagnosticControlConfiguration(
        modes=int(selected["modes"]), photons=3, phase_scale=float(selected["phase_scale"]),
        alpha=float(selected["alpha"]),
        circuit_sha256_by_seed=tuple((seed, circuits[(int(selected["modes"]), seed)])
                                    for seed in config.seeds.circuit_seeds),
    )
    _physical_control_summary(
        exact, expected, float(selection["matched_control"]["selected_alpha"])
    )
    losses = normalized_wis(
        exact["target_ac_power_kw"].to_numpy(),
        exact.loc[:, list(quantile_columns(LOCKED_QUANTILES))].to_numpy(),
        exact["capacity_kw"].to_numpy(),
        LOCKED_QUANTILES,
    )
    scored = exact.loc[:, ["control", "circuit_seed"]].copy()
    scored["normalized_wis"] = losses
    grouped = scored.groupby(["control", "circuit_seed"])["normalized_wis"].agg(
        ["mean", "size"]
    )
    expected_keys = {
        (control, seed)
        for control in ("indistinguishable", "distinguishable")
        for seed in config.seeds.circuit_seeds
    }
    if set(grouped.index) != expected_keys or any(not isinstance(row, dict) for row in seed_scores):
        raise ReportContractError("diagnostic seed WIS inventory is invalid")
    reported = {(row.get("control"), row.get("circuit_seed")): row for row in seed_scores}
    if set(reported) != expected_keys or len(reported) != len(seed_scores):
        raise ReportContractError("diagnostic seed WIS inventory is invalid")
    for key, row in reported.items():
        score = row.get("normalized_wis")
        count = row.get("row_count")
        actual = grouped.loc[key]
        if (
            set(row) != {"control", "circuit_seed", "normalized_wis", "row_count"}
            or type(score) not in (int, float)
            or not np.isfinite(score)
            or score < 0
            or type(count) is not int
            or count != int(actual["size"])
            or count <= 0
            or not np.isclose(score, actual["mean"], rtol=0.0, atol=1e-12)
        ):
            raise ReportContractError("diagnostic seed WIS does not match exact predictions")
    identity = ["sample_id", "site_id", "issued_at_utc", "target_at_utc",
                "target_ac_power_kw", "capacity_kw", "control", "circuit_seed"]
    if primary_exact is None:
        primary_exact = pd.read_parquet(io.BytesIO(captured["benchmark/exact_predictions.parquet"]))
    if not exact[identity].sort_values(identity).reset_index(drop=True).equals(
        primary_exact[identity].sort_values(identity).reset_index(drop=True)
    ):
        raise ReportContractError("diagnostic predictions do not share primary holdout membership")
    summary.update(benchmark_status="EVALUATED", selected=selected, seed_scores=seed_scores,
                   selection_freeze_sha256=freeze, joint_selection_freeze_sha256=combined,
                   prediction_rows=len(exact))
    return summary


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
    expected_holdout_freeze = _holdout_freeze(config, selection_manifest)
    if (
        holdout.get("opened_after_selection_freeze_sha256") != expected_holdout_freeze
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
    exact: pd.DataFrame,
    expected: ExpectedPrimaryConfiguration | _DiagnosticControlConfiguration,
    matched_alpha: float,
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
    diagnostic_3photon: Mapping[str, Any],
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
            "## Three-photon diagnostic",
            "",
            f"- Profile: {diagnostic_3photon['profile_status']}",
            f"- Benchmark: {diagnostic_3photon['benchmark_status']}",
            *(
                (
                    "- Selected modes/photons: "
                    f"{diagnostic_3photon['selected']['modes']}/"
                    f"{diagnostic_3photon['selected']['photons']}",
                    f"- Phase scale: {diagnostic_3photon['selected']['phase_scale']}",
                    f"- Common alpha: {diagnostic_3photon['selected']['alpha']}",
                    f"- State dimensions: {diagnostic_3photon['state_dimensions']}",
                    f"- Per-seed scores: {len(diagnostic_3photon['seed_scores'])}",
                    *(
                        f"  - {row['control']} seed {row['circuit_seed']}: "
                        f"{row['normalized_wis']:.8f}"
                        for row in diagnostic_3photon["seed_scores"]
                    ),
                    "- Joint holdout freeze: "
                    f"{diagnostic_3photon['joint_selection_freeze_sha256']}",
                )
                if diagnostic_3photon["benchmark_status"] == "EVALUATED"
                else ()
            ),
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
    optional_summary: Mapping[str, Any] | None = None,
) -> ReportProducts:
    """Evaluate and serialize a report without fitting or reading source data."""

    diagnostic_3photon = dict(optional_summary or {})
    if not diagnostic_3photon and not config.photonics.enable_three_photon_profile:
        diagnostic_3photon = {"profile_status": "DISABLED", "benchmark_status": "DISABLED"}
    if config.photonics.enable_three_photon_profile:
        if (
            diagnostic_3photon.get("profile_status") not in {"PASS", "FAIL"}
            or diagnostic_3photon.get("used_for_go_stop") is not False
            or diagnostic_3photon.get("scientific_decision") != "NOT_EVALUATED"
            or diagnostic_3photon.get("benchmark_status")
            != ("EVALUATED" if config.photonics.enable_three_photon_benchmark else "DISABLED")
        ):
            raise ReportContractError("optional diagnostic summary is invalid")
    elif diagnostic_3photon != {"profile_status": "DISABLED", "benchmark_status": "DISABLED"}:
        raise ReportContractError("disabled optional diagnostic summary is invalid")

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
        "diagnostic_3photon": diagnostic_3photon,
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
        diagnostic_3photon=diagnostic_3photon,
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
