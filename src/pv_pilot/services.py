"""Application services for the staged pilot command line."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import psutil
import pyarrow.dataset as pds

from . import baselines as baseline_models
from . import benchmark as photonic_benchmark
from . import photonics as photonic_features
from .artifacts import RunStore, sha256_file
from .config import PilotConfig, load_config
from .data import (
    CANONICAL_SCHEMA_VERSION,
    DataContractError,
    DataValidationResult,
    canonical_checksum,
    canonicalize_data,
    diagnostic_parquet_bytes,
    empty_canonical_frame,
    empty_exclusion_frame,
)
from .data import dataframe_parquet_bytes as dataframe_parquet_bytes
from .domain import DetailEntry, StageResult, StageStatus, deterministic_data
from .seeds import SeedManager
from .splits import (
    COMBINED_MEMBERSHIP_POLICY,
    FEATURE_SCHEMA_VERSION,
    MATCHED4_COLUMNS,
    MATCHED6_COLUMNS,
    PARTITION_ORDER_COLUMNS,
    SplitContractError,
    build_feature_splits,
    diagnostic_split_parquet_bytes,
    empty_clip_metrics,
    feature_dataframe_parquet_bytes,
    feature_schema_sha256,
    split_exclusion_dataframe_parquet_bytes,
)


class _ResourceLimitExceeded(RuntimeError):
    def __init__(self, gate: str) -> None:
        super().__init__(f"configured resource cap exceeded: {gate}")
        self.gate = gate


@dataclass(slots=True)
class _ResourceGuard:
    started: float
    wall_seconds: int
    memory_bytes: int
    peak_rss_bytes: int = 0
    peak_declared_bytes: int = 0
    peak_parent_rss_bytes: int = 0
    peak_child_tree_rss_bytes: int = 0
    peak_combined_rss_bytes: int = 0

    def checkpoint(self, declared_bytes: int = 0) -> None:
        self.peak_declared_bytes = max(self.peak_declared_bytes, declared_bytes)
        rss = psutil.Process().memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_parent_rss_bytes = max(self.peak_parent_rss_bytes, rss)
        self.peak_combined_rss_bytes = max(self.peak_combined_rss_bytes, rss)
        if time.perf_counter() - self.started > self.wall_seconds:
            raise _ResourceLimitExceeded("wall_clock")
        if rss > self.memory_bytes:
            raise _ResourceLimitExceeded("memory_rss")
        if declared_bytes > self.memory_bytes:
            raise _ResourceLimitExceeded("dataframe_or_output_bytes")

    def sample_without_enforcement(self) -> None:
        rss = psutil.Process().memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_parent_rss_bytes = max(self.peak_parent_rss_bytes, rss)
        self.peak_combined_rss_bytes = max(self.peak_combined_rss_bytes, rss)

    def record_fit_rss(
        self, parent_rss_bytes: int, child_tree_rss_bytes: int, combined_rss_bytes: int
    ) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, parent_rss_bytes)
        self.peak_parent_rss_bytes = max(self.peak_parent_rss_bytes, parent_rss_bytes)
        self.peak_child_tree_rss_bytes = max(self.peak_child_tree_rss_bytes, child_tree_rss_bytes)
        self.peak_combined_rss_bytes = max(self.peak_combined_rss_bytes, combined_rss_bytes)


class DefaultPilotServices:
    """Concrete init service with explicit placeholders for future stages."""

    @staticmethod
    def _unimplemented(stage: str) -> None:
        raise NotImplementedError(
            f"Pipeline stages are not implemented yet; cannot execute {stage!r}."
        )

    def init_run(self, config: Path) -> Path:
        loaded = load_config(config)
        return RunStore.create(loaded).root

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact must be an object: {path.name}")
        return payload

    @classmethod
    def _run_config(cls, store: RunStore) -> PilotConfig:
        payload = cls._read_object(store.root / "config.json")
        paths = payload.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("run config snapshot has invalid paths")
        restored = dict(payload)
        restored_paths: dict[str, object] = {}
        for key in ("input_data", "artifact_dir"):
            record = paths.get(key)
            if not isinstance(record, dict) or not isinstance(record.get("value"), str):
                raise ValueError("run config snapshot has invalid path record")
            restored_paths[key] = record["value"]
        restored["paths"] = restored_paths
        config = PilotConfig.model_validate(restored)
        if config.config_hash() != store.config_hash:
            raise ValueError("run config snapshot does not reproduce its config hash")
        return config

    @classmethod
    def _existing_validate_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "validate-data/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "BLOCKED", "INVALID"}:
            raise ValueError("validate-data has an invalid terminal marker")
        store.verify_terminal_stage("validate-data", accepted_statuses=frozenset({str(status)}))
        result_path = store.root / "validate-data/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if status == "SUCCESS":
            return result.model_copy(update={"message": "validate-data already successful"})
        return result.model_copy(update={"message": f"validate-data already {status.lower()}"})

    @classmethod
    def _existing_build_splits_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "build-splits/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "INVALID"}:
            raise ValueError("build-splits has an invalid terminal marker")
        store.verify_terminal_stage("build-splits", accepted_statuses=frozenset({str(status)}))
        result_path = store.root / "build-splits/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        message = (
            "build-splits already successful"
            if status == "SUCCESS"
            else "build-splits already invalid"
        )
        return result.model_copy(update={"message": message})

    @classmethod
    def _existing_train_baselines_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "train-baselines/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "INVALID"}:
            raise ValueError("train-baselines has an invalid terminal marker")
        store.verify_terminal_stage("train-baselines", accepted_statuses=frozenset({str(status)}))
        result_path = store.root / "train-baselines/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        message = (
            "train-baselines already successful"
            if status == "SUCCESS"
            else "train-baselines already invalid"
        )
        return result.model_copy(update={"message": message})

    @classmethod
    def _existing_photonic_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "build-photonic-features/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "INVALID"}:
            raise ValueError("build-photonic-features has an invalid terminal marker")
        allowlist = {
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        }
        if status == "SUCCESS":
            store.verify_terminal_stage(
                "build-photonic-features",
                accepted_statuses=frozenset({"SUCCESS"}),
                prerequisite_output_allowlist=allowlist,
            )
        else:
            store.verify_terminal_diagnostic_stage("build-photonic-features")
        result_path = store.root / "build-photonic-features/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return result.model_copy(
            update={
                "message": (
                    "build-photonic-features already successful"
                    if status == "SUCCESS"
                    else "build-photonic-features already invalid"
                )
            }
        )

    @classmethod
    def _existing_benchmark_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "benchmark/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "INVALID"}:
            raise ValueError("benchmark has an invalid terminal marker")
        if status == "SUCCESS":
            store.verify_terminal_stage("benchmark", accepted_statuses=frozenset({"SUCCESS"}))
        else:
            store.verify_terminal_diagnostic_stage("benchmark")
        result = StageResult.model_validate_json(
            (store.root / "benchmark/stage_result.json").read_text(encoding="utf-8")
        )
        return result.model_copy(update={"message": f"benchmark already {status.lower()}"})

    @staticmethod
    def _publish_benchmark_invalid(
        *,
        store: RunStore,
        guard: _ResourceGuard,
        gate: str,
        message: str,
    ) -> StageResult:
        empty_scores = photonic_benchmark.parquet_bytes(
            pd.DataFrame(columns=photonic_benchmark.validation_score_arrow_schema().names),
            photonic_benchmark.validation_score_arrow_schema(),
        )
        empty_exact = photonic_benchmark.parquet_bytes(
            pd.DataFrame(columns=photonic_benchmark.exact_prediction_arrow_schema().names),
            photonic_benchmark.exact_prediction_arrow_schema(),
        )
        empty_shots = photonic_benchmark.parquet_bytes(
            pd.DataFrame(columns=photonic_benchmark.finite_shot_count_arrow_schema().names),
            photonic_benchmark.finite_shot_count_arrow_schema(),
        )
        empty_shot_predictions = photonic_benchmark.parquet_bytes(
            pd.DataFrame(columns=photonic_benchmark.finite_shot_prediction_arrow_schema().names),
            photonic_benchmark.finite_shot_prediction_arrow_schema(),
        )
        empty_feature_bytes = photonic_features.feature_parquet_bytes(
            pd.DataFrame(columns=photonic_features.feature_arrow_schema().names)
        )
        model_bytes = pickle.dumps({}, protocol=5)
        result = StageResult(
            stage="benchmark",
            status=StageStatus.INVALID,
            message=message,
            artifacts=(
                Path("benchmark/selection_manifest.json"),
                Path("benchmark/validation_scores.parquet"),
                Path("benchmark/holdout_photonic_features.parquet"),
                Path("benchmark/exact_predictions.parquet"),
                Path("benchmark/finite_shot_counts.parquet"),
                Path("benchmark/finite_shot_predictions.parquet"),
                Path("benchmark/models/photonic_models.pkl"),
                Path("benchmark/models/manifest.json"),
                Path("benchmark/diagnostics.json"),
                Path("benchmark/resource_ledger.json"),
            ),
            details=(
                DetailEntry(key="gate", value=gate),
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
            ),
        )
        ledger = {
            "schema_version": "1.0",
            "stage": "benchmark",
            "status": "invalid",
            "resource_status": "EXCEEDED" if gate.startswith("resource") else "WITHIN_LIMIT",
            "gate": gate,
            "wall_clock_seconds": time.perf_counter() - guard.started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
        }
        with store.stage(
            "benchmark",
            terminal_status="INVALID",
            prerequisite_verification="marker_only",
            details={
                "result_status": "invalid",
                "scientific_decision": "NOT_EVALUATED",
                "gate": gate,
            },
        ) as transaction:
            transaction.write_json(
                "selection_manifest.json",
                {"schema_version": "1.0", "status": "invalid", "gate": gate},
            )
            transaction.write_bytes("validation_scores.parquet", empty_scores)
            transaction.write_bytes("holdout_photonic_features.parquet", empty_feature_bytes)
            transaction.write_bytes("exact_predictions.parquet", empty_exact)
            transaction.write_bytes("finite_shot_counts.parquet", empty_shots)
            transaction.write_bytes("finite_shot_predictions.parquet", empty_shot_predictions)
            transaction.write_bytes("models/photonic_models.pkl", model_bytes)
            transaction.write_json(
                "models/manifest.json",
                {
                    "schema_version": "1.0",
                    "status": "invalid",
                    "models_sha256": hashlib.sha256(model_bytes).hexdigest(),
                },
            )
            transaction.write_json("diagnostics.json", {"schema_version": "1.0", "gate": gate})
            transaction.write_json("resource_ledger.json", ledger)
            transaction.write_json("stage_result.json", deterministic_data(result))
        return result

    @staticmethod
    def _publish_baseline_resource_invalid(
        *,
        store: RunStore,
        config: PilotConfig,
        guard: _ResourceGuard,
        gate: str,
        message: str = "baseline training exceeded a configured resource cap",
        resource_status: str = "EXCEEDED",
    ) -> StageResult:
        del config
        empty_predictions = pd.DataFrame(columns=baseline_models.prediction_arrow_schema().names)
        empty_scores = pd.DataFrame(columns=baseline_models.validation_score_arrow_schema().names)
        prediction_bytes = baseline_models.prediction_parquet_bytes(empty_predictions)
        validation_bytes = baseline_models.validation_scores_parquet_bytes(empty_scores)
        model_bytes = pickle.dumps({}, protocol=5)
        result = StageResult(
            stage="train-baselines",
            status=StageStatus.INVALID,
            message=message,
            artifacts=(
                Path("train-baselines/selection_manifest.json"),
                Path("train-baselines/predictions.parquet"),
                Path("train-baselines/validation_scores.parquet"),
                Path("train-baselines/diagnostics.json"),
                Path("train-baselines/models/baseline_models.pkl"),
                Path("train-baselines/models/manifest.json"),
                Path("train-baselines/resource_ledger.json"),
            ),
            details=(
                DetailEntry(key="resource_gate", value=gate),
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
            ),
        )
        ledger = {
            "schema_version": "1.0",
            "stage": "train-baselines",
            "status": StageStatus.INVALID.value,
            "resource_status": resource_status,
            "resource_gate": gate,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_parent_rss_bytes": guard.peak_parent_rss_bytes,
            "peak_child_tree_rss_bytes": guard.peak_child_tree_rss_bytes,
            "peak_combined_rss_bytes": guard.peak_combined_rss_bytes,
            "max_memory_bytes": guard.memory_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "cpu_only": True,
        }
        with store.stage(
            "train-baselines",
            input_paths=(
                Path("build-splits/development_features.parquet"),
                Path("build-splits/feature_manifest.json"),
                Path("build-splits/minmax_scalers.json"),
            ),
            terminal_status="INVALID",
            prerequisite_verification="marker_only",
            details={
                "result_status": StageStatus.INVALID.value,
                "scientific_decision": "NOT_EVALUATED",
                "resource_gate": gate,
            },
        ) as transaction:
            transaction.write_json(
                "selection_manifest.json",
                {"schema_version": "1.0", "status": "invalid", "candidates": []},
            )
            transaction.write_bytes("predictions.parquet", prediction_bytes)
            transaction.write_bytes("validation_scores.parquet", validation_bytes)
            transaction.write_json("diagnostics.json", {"schema_version": "1.0"})
            transaction.write_bytes("models/baseline_models.pkl", model_bytes)
            transaction.write_json(
                "models/manifest.json",
                {
                    "schema_version": "1.0",
                    "status": "invalid",
                    "models": [],
                    "scientific_decision": "NOT_EVALUATED",
                },
            )
            transaction.write_json("stage_result.json", deterministic_data(result))
            transaction.write_json("resource_ledger.json", ledger)
        return result

    @staticmethod
    def _publish_resource_invalid(
        *,
        store: RunStore,
        config: PilotConfig,
        source_path: Path,
        expected_hash: object,
        register_external_input: bool,
        guard: _ResourceGuard,
        gate: str,
    ) -> StageResult:
        """Publish bounded INVALID diagnostics without re-enforcing an expired budget."""

        include_daylight = config.data.columns.daylight is not None
        canonical = empty_canonical_frame(include_daylight=include_daylight)
        canonical_bytes, exclusion_bytes = diagnostic_parquet_bytes(include_daylight)
        guard.sample_without_enforcement()
        elapsed = time.perf_counter() - guard.started
        card = {
            "accepted_input_row_count": 0,
            "canonical_row_count": 0,
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "canonical_sha256": canonical_checksum(canonical),
            "cadence_minutes": config.data.raw_cadence_minutes,
            "end_utc": None,
            "excluded_by_reason": {},
            "excluded_row_count": 0,
            "exclusion_record_count": 0,
            "input_row_count": 0,
            "input_sha256": expected_hash,
            "missing_expected_interval_count": 0,
            "provenance": config.protocol.provenance.value,
            "schema_version": "1.0",
            "scientific_decision": "NOT_EVALUATED",
            "site_counts": {},
            "start_utc": None,
            "status": StageStatus.INVALID.value,
            "synthetic": config.protocol.provenance.value == "synthetic",
        }
        ledger = {
            "elapsed_seconds": elapsed,
            "human_cleaning_hours": None,
            "peak_declared_bytes_observed": guard.peak_declared_bytes,
            "peak_rss_bytes_observed": guard.peak_rss_bytes,
            "resource_status": "EXCEEDED",
            "schema_version": "1.0",
            "status": StageStatus.INVALID.value,
            "timebox_seconds": config.resources.max_wall_clock_seconds,
        }
        result = StageResult(
            stage="validate-data",
            status=StageStatus.INVALID,
            message="data validation exceeded a configured resource cap",
            artifacts=(
                Path("validate-data/canonical.parquet"),
                Path("validate-data/qc_exclusions.parquet"),
                Path("validate-data/data_card.json"),
                Path("validate-data/cleaning_ledger.json"),
            ),
            details=(
                DetailEntry(key="input_sha256", value=expected_hash),
                DetailEntry(key="resource_gate", value=gate),
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
            ),
        )
        with store.stage(
            "validate-data",
            external_input_hashes=(
                {source_path: expected_hash}
                if register_external_input and isinstance(expected_hash, str)
                else None
            ),
            terminal_status="INVALID",
            details={
                "result_status": "invalid",
                "scientific_decision": "NOT_EVALUATED",
            },
        ) as transaction:
            transaction.write_bytes("canonical.parquet", canonical_bytes)
            transaction.write_bytes("qc_exclusions.parquet", exclusion_bytes)
            transaction.write_json("data_card.json", card)
            transaction.write_json("cleaning_ledger.json", ledger)
            transaction.write_json("stage_result.json", deterministic_data(result))
        return result

    def validate_data(self, run_dir: Path) -> StageResult:
        started = time.perf_counter()
        store = RunStore.open(run_dir)
        existing = self._existing_validate_result(store)
        if existing is not None:
            return existing
        config = self._run_config(store)
        source_manifest = self._read_object(store.root / "source_manifest.json")
        source_record = source_manifest.get("input")
        if not isinstance(source_record, dict) or not isinstance(source_record.get("path"), str):
            raise ValueError("source manifest has invalid input record")
        source_path = Path(source_record["path"])
        expected_hash = source_record.get("sha256")
        actual_hash = sha256_file(source_path) if source_path.is_file() else None
        result: DataValidationResult
        status: StageStatus
        message: str
        failure_details: dict[str, object] = {}
        resource_status = "WITHIN_LIMIT"
        memory_limit_bytes = int(config.resources.max_memory_gb * 1024**3)
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=memory_limit_bytes,
        )
        register_external_input = False
        if source_record.get("status") == "MISSING":
            status = StageStatus.BLOCKED
            message = "input dataset is missing; scientific decision not evaluated"
            result = DataValidationResult(
                status=status,
                canonical=empty_canonical_frame(
                    include_daylight=config.data.columns.daylight is not None
                ),
                exclusions=empty_exclusion_frame(),
                input_row_count=0,
                accepted_input_row_count=0,
            )
        elif actual_hash is None:
            status = StageStatus.INVALID
            message = "input dataset checksum cannot be verified because the source is missing"
            failure_details = {
                "actual_sha256": None,
                "expected_sha256": expected_hash,
            }
            result = DataValidationResult(
                status=status,
                canonical=empty_canonical_frame(
                    include_daylight=config.data.columns.daylight is not None
                ),
                exclusions=empty_exclusion_frame(),
                input_row_count=0,
                accepted_input_row_count=0,
            )
        elif not isinstance(expected_hash, str) or actual_hash != expected_hash:
            status = StageStatus.INVALID
            message = "input dataset checksum differs from immutable init snapshot"
            failure_details = {
                "actual_sha256": actual_hash,
                "expected_sha256": expected_hash,
            }
            result = DataValidationResult(
                status=status,
                canonical=empty_canonical_frame(
                    include_daylight=config.data.columns.daylight is not None
                ),
                exclusions=empty_exclusion_frame(),
                input_row_count=0,
                accepted_input_row_count=0,
            )
        else:
            register_external_input = True
            try:
                guard.checkpoint(source_path.stat().st_size)
                result = canonicalize_data(
                    source_path,
                    config.data,
                    config.data.format,
                    resource_checkpoint=guard.checkpoint,
                )
                guard.checkpoint()
            except _ResourceLimitExceeded as error:
                return self._publish_resource_invalid(
                    store=store,
                    config=config,
                    source_path=source_path,
                    expected_hash=expected_hash,
                    register_external_input=register_external_input,
                    guard=guard,
                    gate=error.gate,
                )
            except (DataContractError, OSError, ValueError) as error:
                status = StageStatus.INVALID
                message = f"input dataset violates its configured contract: {error}"
                failure_details = {"error_type": type(error).__name__}
                result = DataValidationResult(
                    status=status,
                    canonical=empty_canonical_frame(
                        include_daylight=config.data.columns.daylight is not None
                    ),
                    exclusions=empty_exclusion_frame(),
                    input_row_count=0,
                    accepted_input_row_count=0,
                )
            else:
                post_read_hash = sha256_file(source_path)
                try:
                    guard.checkpoint()
                except _ResourceLimitExceeded as error:
                    return self._publish_resource_invalid(
                        store=store,
                        config=config,
                        source_path=source_path,
                        expected_hash=expected_hash,
                        register_external_input=register_external_input,
                        guard=guard,
                        gate=error.gate,
                    )
                if post_read_hash != expected_hash:
                    register_external_input = False
                    status = StageStatus.INVALID
                    message = "input dataset changed while it was being validated"
                    failure_details = {
                        "actual_sha256": post_read_hash,
                        "expected_sha256": expected_hash,
                    }
                    result = DataValidationResult(
                        status=status,
                        canonical=empty_canonical_frame(
                            include_daylight=config.data.columns.daylight is not None
                        ),
                        exclusions=empty_exclusion_frame(),
                        input_row_count=0,
                        accepted_input_row_count=0,
                    )
                else:
                    status = result.status
                    message = (
                        "data validation successful"
                        if status is StageStatus.SUCCESS
                        else "no canonical observations survived quality control"
                    )

        canonical = result.canonical
        exclusions = result.exclusions
        site_counts = {
            str(key): int(value) for key, value in sorted(Counter(canonical["site_id"]).items())
        }
        start_utc = None
        end_utc = None
        if not canonical.empty:
            start_utc = canonical["interval_end_utc"].min().isoformat()
            end_utc = canonical["interval_end_utc"].max().isoformat()
        card = {
            "canonical_row_count": len(canonical),
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "canonical_sha256": canonical_checksum(canonical),
            "cadence_minutes": config.data.raw_cadence_minutes,
            "end_utc": end_utc,
            "excluded_by_reason": result.exclusion_counts,
            "excluded_row_count": result.excluded_input_row_count,
            "exclusion_record_count": len(exclusions),
            "input_row_count": result.input_row_count,
            "accepted_input_row_count": result.accepted_input_row_count,
            "missing_expected_interval_count": result.missing_expected_interval_count,
            "input_sha256": expected_hash,
            "provenance": config.protocol.provenance.value,
            "schema_version": "1.0",
            "scientific_decision": "NOT_EVALUATED",
            "site_counts": site_counts,
            "start_utc": start_utc,
            "status": status.value,
            "synthetic": config.protocol.provenance.value == "synthetic",
        }
        details = {
            "scientific_decision": "NOT_EVALUATED",
            "input_sha256": expected_hash,
            **failure_details,
        }
        stage_result = StageResult(
            stage="validate-data",
            status=status,
            message=message,
            artifacts=(
                Path("validate-data/canonical.parquet"),
                Path("validate-data/qc_exclusions.parquet"),
                Path("validate-data/data_card.json"),
                Path("validate-data/cleaning_ledger.json"),
            ),
            details=tuple(DetailEntry(key=key, value=value) for key, value in details.items()),
        )
        terminal = cast(Literal["SUCCESS", "BLOCKED", "INVALID"], status.value.upper())
        try:
            canonical_bytes = dataframe_parquet_bytes(canonical, kind="canonical")
            guard.checkpoint(len(canonical_bytes))
            exclusion_bytes = dataframe_parquet_bytes(exclusions, kind="exclusions")
            guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
            with store.stage(
                "validate-data",
                external_input_hashes=(
                    {source_path: expected_hash}
                    if register_external_input and isinstance(expected_hash, str)
                    else None
                ),
                terminal_status=terminal,
                details={
                    "result_status": status.value,
                    "scientific_decision": "NOT_EVALUATED",
                },
                commit_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_bytes("canonical.parquet", canonical_bytes)
                guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
                transaction.write_bytes("qc_exclusions.parquet", exclusion_bytes)
                guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
                transaction.write_json("data_card.json", card)
                guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
                transaction.write_json("stage_result.json", deterministic_data(stage_result))
                guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
                elapsed = time.perf_counter() - started
                ledger = {
                    "elapsed_seconds": elapsed,
                    "human_cleaning_hours": None,
                    "peak_declared_bytes_observed": guard.peak_declared_bytes,
                    "peak_rss_bytes_observed": guard.peak_rss_bytes,
                    "resource_status": resource_status,
                    "schema_version": "1.0",
                    "status": status.value,
                    "timebox_seconds": config.resources.max_wall_clock_seconds,
                }
                transaction.write_json("cleaning_ledger.json", ledger)
                guard.checkpoint(len(canonical_bytes) + len(exclusion_bytes))
        except _ResourceLimitExceeded as error:
            return self._publish_resource_invalid(
                store=store,
                config=config,
                source_path=source_path,
                expected_hash=expected_hash,
                register_external_input=register_external_input,
                guard=guard,
                gate=error.gate,
            )
        return stage_result

    @staticmethod
    def _split_artifacts() -> tuple[Path, ...]:
        return (
            Path("build-splits/feature_matrix.parquet"),
            Path("build-splits/development_features.parquet"),
            Path("build-splits/holdout_features.parquet"),
            Path("build-splits/exclusions.parquet"),
            Path("build-splits/feature_manifest.json"),
            Path("build-splits/minmax_scalers.json"),
            Path("build-splits/clip_rates.json"),
            Path("build-splits/resource_ledger.json"),
        )

    @staticmethod
    def _split_ledger(
        *,
        guard: _ResourceGuard,
        status: StageStatus,
        resource_status: str,
    ) -> dict[str, object]:
        guard.sample_without_enforcement()
        return {
            "elapsed_seconds": time.perf_counter() - guard.started,
            "peak_declared_bytes_observed": guard.peak_declared_bytes,
            "peak_rss_bytes_observed": guard.peak_rss_bytes,
            "resource_status": resource_status,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "scientific_decision": "NOT_EVALUATED",
            "status": status.value,
            "timebox_seconds": guard.wall_seconds,
        }

    def _publish_split_invalid(
        self,
        *,
        store: RunStore,
        config: PilotConfig,
        canonical_path: Path,
        guard: _ResourceGuard,
        message: str,
        failure_details: dict[str, object],
        resource_status: str,
    ) -> StageResult:
        """Publish small, schema-stable INVALID outputs without an expired guard."""

        feature_bytes, exclusion_bytes = diagnostic_split_parquet_bytes(config)
        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "status": StageStatus.INVALID.value,
            "scientific_decision": "NOT_EVALUATED",
            "split_counts": {"train": 0, "validation": 0, "test": 0},
            "exclusion_counts": {},
            "partitions": {
                "development": {
                    "path": "build-splits/development_features.parquet",
                    "splits": ["train", "validation"],
                    "row_count": 0,
                    "schema_sha256": feature_schema_sha256(config),
                    "sample_id_order_sha256": hashlib.sha256(b"[]").hexdigest(),
                    "order_policy": list(PARTITION_ORDER_COLUMNS),
                },
                "holdout": {
                    "path": "build-splits/holdout_features.parquet",
                    "splits": ["test"],
                    "row_count": 0,
                    "schema_sha256": feature_schema_sha256(config),
                    "sample_id_order_sha256": hashlib.sha256(b"[]").hexdigest(),
                    "order_policy": list(PARTITION_ORDER_COLUMNS),
                },
                "combined": {
                    "row_count": 0,
                    "schema_sha256": feature_schema_sha256(config),
                    "membership_policy": COMBINED_MEMBERSHIP_POLICY,
                    "sample_id_membership_sha256": hashlib.sha256(b"[]").hexdigest(),
                },
            },
            "diagnostic": failure_details,
        }
        details = {
            "scientific_decision": "NOT_EVALUATED",
            **failure_details,
        }
        result = StageResult(
            stage="build-splits",
            status=StageStatus.INVALID,
            message=message,
            artifacts=self._split_artifacts(),
            details=tuple(DetailEntry(key=key, value=value) for key, value in details.items()),
        )
        ledger = self._split_ledger(
            guard=guard,
            status=StageStatus.INVALID,
            resource_status=resource_status,
        )
        with store.stage(
            "build-splits",
            input_paths=(Path("validate-data/canonical.parquet"),),
            terminal_status="INVALID",
            details={
                "result_status": StageStatus.INVALID.value,
                "scientific_decision": "NOT_EVALUATED",
            },
        ) as transaction:
            transaction.write_bytes("feature_matrix.parquet", feature_bytes)
            transaction.write_bytes("development_features.parquet", feature_bytes)
            transaction.write_bytes("holdout_features.parquet", feature_bytes)
            transaction.write_bytes("exclusions.parquet", exclusion_bytes)
            transaction.write_json("feature_manifest.json", manifest)
            transaction.write_json("minmax_scalers.json", {})
            transaction.write_json("clip_rates.json", empty_clip_metrics())
            transaction.write_json("stage_result.json", deterministic_data(result))
            transaction.write_json("resource_ledger.json", ledger)
        return result

    def build_splits(self, run_dir: Path) -> StageResult:
        started = time.perf_counter()
        store = RunStore.open(run_dir)
        existing = self._existing_build_splits_result(store)
        if existing is not None:
            return existing
        store.verify_terminal_stage("validate-data")
        config = self._run_config(store)
        canonical_path = store.root / "validate-data/canonical.parquet"
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        try:
            guard.checkpoint(canonical_path.stat().st_size)
            canonical = pd.read_parquet(canonical_path)
            declared = int(canonical.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
            build = build_feature_splits(
                canonical,
                config,
                resource_checkpoint=guard.checkpoint,
            )
        except _ResourceLimitExceeded as error:
            return self._publish_split_invalid(
                store=store,
                config=config,
                canonical_path=canonical_path,
                guard=guard,
                message="split construction exceeded a configured resource cap",
                failure_details={"resource_gate": error.gate},
                resource_status="EXCEEDED",
            )
        except (SplitContractError, OSError, TypeError, ValueError) as error:
            return self._publish_split_invalid(
                store=store,
                config=config,
                canonical_path=canonical_path,
                guard=guard,
                message=f"canonical data violates the split contract: {error}",
                failure_details={"error_type": type(error).__name__},
                resource_status="WITHIN_LIMIT",
            )

        try:
            feature_bytes = feature_dataframe_parquet_bytes(build.features, config)
            guard.checkpoint(declared + len(feature_bytes))
            development_bytes = feature_dataframe_parquet_bytes(
                build.features.loc[build.features["split"].isin(("train", "validation"))],
                config,
            )
            holdout_bytes = feature_dataframe_parquet_bytes(
                build.features.loc[build.features["split"] == "test"], config
            )
            guard.checkpoint(
                declared + len(feature_bytes) + len(development_bytes) + len(holdout_bytes)
            )
            exclusion_bytes = split_exclusion_dataframe_parquet_bytes(build.exclusions)
            guard.checkpoint(
                declared
                + len(feature_bytes)
                + len(development_bytes)
                + len(holdout_bytes)
                + len(exclusion_bytes)
            )
        except _ResourceLimitExceeded as error:
            return self._publish_split_invalid(
                store=store,
                config=config,
                canonical_path=canonical_path,
                guard=guard,
                message="split serialization exceeded a configured resource cap",
                failure_details={"resource_gate": error.gate},
                resource_status="EXCEEDED",
            )
        except (OSError, TypeError, ValueError) as error:
            return self._publish_split_invalid(
                store=store,
                config=config,
                canonical_path=canonical_path,
                guard=guard,
                message=f"split outputs violate their serialization contract: {error}",
                failure_details={"error_type": type(error).__name__},
                resource_status="WITHIN_LIMIT",
            )

        status = build.status
        message = (
            "feature splits built successfully"
            if status is StageStatus.SUCCESS
            else "one or more chronological splits contain no eligible samples"
        )
        result = StageResult(
            stage="build-splits",
            status=status,
            message=message,
            artifacts=self._split_artifacts(),
            details=(
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
                DetailEntry(key="split_counts", value=build.feature_manifest["split_counts"]),
            ),
        )
        ledger = self._split_ledger(
            guard=guard,
            status=status,
            resource_status="WITHIN_LIMIT",
        )
        terminal = cast(Literal["SUCCESS", "INVALID"], status.value.upper())
        try:
            with store.stage(
                "build-splits",
                input_paths=(Path("validate-data/canonical.parquet"),),
                terminal_status=terminal,
                details={
                    "result_status": status.value,
                    "scientific_decision": "NOT_EVALUATED",
                },
                commit_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_bytes("feature_matrix.parquet", feature_bytes)
                transaction.write_bytes("development_features.parquet", development_bytes)
                transaction.write_bytes("holdout_features.parquet", holdout_bytes)
                output_bytes = (
                    len(feature_bytes)
                    + len(development_bytes)
                    + len(holdout_bytes)
                    + len(exclusion_bytes)
                )
                guard.checkpoint(declared + output_bytes)
                transaction.write_bytes("exclusions.parquet", exclusion_bytes)
                guard.checkpoint(declared + output_bytes)
                transaction.write_json("feature_manifest.json", build.feature_manifest)
                guard.checkpoint(declared + output_bytes)
                transaction.write_json("minmax_scalers.json", build.scalers)
                guard.checkpoint(declared + output_bytes)
                transaction.write_json("clip_rates.json", build.clip_rates)
                guard.checkpoint(declared + output_bytes)
                transaction.write_json("stage_result.json", deterministic_data(result))
                guard.checkpoint(declared + output_bytes)
                transaction.write_json("resource_ledger.json", ledger)
                guard.checkpoint(declared + output_bytes)
        except _ResourceLimitExceeded as error:
            return self._publish_split_invalid(
                store=store,
                config=config,
                canonical_path=canonical_path,
                guard=guard,
                message="split publication exceeded a configured resource cap",
                failure_details={"resource_gate": error.gate},
                resource_status="EXCEEDED",
            )
        return result

    def train_baselines(self, run_dir: Path) -> StageResult:
        started = time.perf_counter()
        store = RunStore.open(run_dir)
        existing = self._existing_train_baselines_result(store)
        if existing is not None:
            return existing
        store.verify_terminal_stage("validate-data")
        config = self._run_config(store)
        development_path = store.root / "build-splits/development_features.parquet"
        holdout_path = store.root / "build-splits/holdout_features.parquet"
        manifest_path = store.root / "build-splits/feature_manifest.json"
        split_stage_manifest = self._read_object(store.root / "build-splits/stage_manifest.json")
        if (
            split_stage_manifest.get("schema_version") != "1.0"
            or split_stage_manifest.get("prerequisite_manifest_hashes")
            != {"validate-data": sha256_file(store.root / "validate-data/stage_manifest.json")}
            or split_stage_manifest.get("stage") != "build-splits"
            or split_stage_manifest.get("status") != "SUCCESS"
            or split_stage_manifest.get("config_hash") != store.config_hash
        ):
            raise ValueError("build-splits has an invalid SUCCESS marker")
        split_outputs = split_stage_manifest.get("outputs")
        if not isinstance(split_outputs, list):
            raise ValueError("build-splits marker has invalid outputs")
        expected_output_hashes = {
            str(record.get("path")): str(record.get("sha256"))
            for record in split_outputs
            if isinstance(record, dict)
        }
        development_relative = "build-splits/development_features.parquet"
        holdout_relative = "build-splits/holdout_features.parquet"
        for required in (
            development_relative,
            holdout_relative,
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ):
            if required not in expected_output_hashes:
                raise ValueError(f"build-splits marker is missing partition output: {required}")
        if (
            not development_path.is_file()
            or sha256_file(development_path) != expected_output_hashes[development_relative]
        ):
            raise ValueError("development partition checksum mismatch")
        for relative_path in (
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ):
            path = store.root / relative_path
            if not path.is_file() or sha256_file(path) != expected_output_hashes[relative_path]:
                raise ValueError(f"split metadata checksum mismatch: {relative_path}")
        if not holdout_path.is_file():
            raise ValueError("holdout partition is missing")
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        try:
            guard.checkpoint(development_path.stat().st_size)
            development = pds.dataset(development_path, format="parquet").to_table().to_pandas()
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        feature_manifest = self._read_object(manifest_path)
        expected_profiles = {
            "matched4_columns": list(MATCHED4_COLUMNS),
            "matched6_columns": list(MATCHED6_COLUMNS),
        }
        for key, expected in expected_profiles.items():
            if feature_manifest.get(key) != expected:
                return self._publish_baseline_resource_invalid(
                    store=store,
                    config=config,
                    guard=guard,
                    gate="data_or_split_contract",
                    message=(f"feature manifest {key} does not match the frozen baseline contract"),
                    resource_status="WITHIN_LIMIT",
                )
        expected_xgb = [
            "power_lag_0m_kw",
            "power_lag_5m_kw",
            "power_lag_10m_kw",
            "power_lag_15m_kw",
            "power_lag_30m_kw",
            "power_lag_60m_kw",
            "ghi_w_m2",
            "air_temp_c",
        ]
        if config.daylight.source.value == "solar_elevation":
            expected_xgb.extend(("solar_elevation_deg", "solar_azimuth_deg"))
        if feature_manifest.get("xgb_columns") != expected_xgb:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="data_or_split_contract",
                message=(
                    "feature manifest xgb_columns does not match the frozen baseline contract"
                ),
                resource_status="WITHIN_LIMIT",
            )
        partitions = feature_manifest.get("partitions")
        if not isinstance(partitions, dict):
            raise baseline_models.BaselineContractError(
                "feature manifest has no immutable partition contract"
            )
        development_contract = partitions.get("development")
        holdout_contract = partitions.get("holdout")
        combined_contract = partitions.get("combined")
        if not isinstance(development_contract, dict):
            raise baseline_models.BaselineContractError(
                "feature manifest development partition record is invalid"
            )
        if not isinstance(holdout_contract, dict):
            raise baseline_models.BaselineContractError(
                "feature manifest holdout partition record is invalid"
            )
        if not isinstance(combined_contract, dict):
            raise baseline_models.BaselineContractError(
                "feature manifest combined partition record is invalid"
            )
        development_order = (
            development.sort_values(list(PARTITION_ORDER_COLUMNS), kind="stable")["sample_id"]
            .astype(str)
            .tolist()
        )
        if (
            development_contract.get("row_count") != len(development)
            or development_contract.get("schema_sha256") != feature_schema_sha256(config)
            or development_contract.get("path") != development_relative
            or development_contract.get("splits") != ["train", "validation"]
            or development_contract.get("order_policy") != list(PARTITION_ORDER_COLUMNS)
            or set(development["split"].astype(str)) != {"train", "validation"}
            or development["sample_id"].duplicated().any()
            or development["sample_id"].astype(str).tolist() != development_order
        ):
            raise baseline_models.BaselineContractError(
                "development partition violates its manifest contract"
            )
        development_sample_payload = json.dumps(
            development["sample_id"].astype(str).tolist(), separators=(",", ":")
        )
        development_sample_hash = hashlib.sha256(
            development_sample_payload.encode("utf-8")
        ).hexdigest()
        declared_development_hash = development_contract.get("sample_id_order_sha256")
        if declared_development_hash != development_sample_hash:
            raise baseline_models.BaselineContractError(
                "development partition sample order hash mismatch"
            )

        holdout_open_count = 0
        holdout_declared = 0

        def load_holdout(selection_freeze_sha256: str) -> pd.DataFrame:
            nonlocal holdout_declared, holdout_open_count
            if len(selection_freeze_sha256) != 64:
                raise RuntimeError("holdout cannot open before selection is frozen")
            if holdout_open_count != 0:
                raise RuntimeError("holdout loader may be invoked exactly once")
            holdout_open_count += 1
            if sha256_file(holdout_path) != expected_output_hashes[holdout_relative]:
                raise ValueError("holdout partition checksum mismatch")
            holdout = pds.dataset(holdout_path, format="parquet").to_table().to_pandas()
            holdout_declared = int(holdout.memory_usage(index=True, deep=True).sum())
            if holdout_contract.get("row_count") != len(holdout):
                raise baseline_models.BaselineContractError(
                    "holdout partition violates its manifest row count"
                )
            if (
                holdout_contract.get("schema_sha256") != feature_schema_sha256(config)
                or holdout_contract.get("path") != holdout_relative
                or holdout_contract.get("splits") != ["test"]
                or holdout_contract.get("order_policy") != list(PARTITION_ORDER_COLUMNS)
                or combined_contract.get("schema_sha256") != feature_schema_sha256(config)
                or combined_contract.get("membership_policy") != COMBINED_MEMBERSHIP_POLICY
            ):
                raise baseline_models.BaselineContractError(
                    "feature partition schema hash mismatch"
                )
            if (
                set(holdout["split"].astype(str)) != {"test"}
                or holdout["sample_id"].duplicated().any()
                or holdout["sample_id"].astype(str).tolist()
                != holdout.sort_values(list(PARTITION_ORDER_COLUMNS), kind="stable")["sample_id"]
                .astype(str)
                .tolist()
            ):
                raise baseline_models.BaselineContractError(
                    "holdout partition violates its split or uniqueness contract"
                )
            if combined_contract.get("row_count") != len(development) + len(holdout):
                raise baseline_models.BaselineContractError(
                    "feature partitions do not reproduce the combined row count"
                )
            holdout_sample_payload = json.dumps(
                holdout["sample_id"].astype(str).tolist(), separators=(",", ":")
            )
            holdout_sample_hash = hashlib.sha256(holdout_sample_payload.encode("utf-8")).hexdigest()
            declared_holdout_hash = holdout_contract.get("sample_id_order_sha256")
            if declared_holdout_hash != holdout_sample_hash:
                raise baseline_models.BaselineContractError(
                    "holdout partition sample order hash mismatch"
                )
            development_ids = development["sample_id"].astype(str).tolist()
            holdout_ids = holdout["sample_id"].astype(str).tolist()
            if set(development_ids) & set(holdout_ids):
                raise baseline_models.BaselineContractError(
                    "development and holdout partition memberships overlap"
                )
            combined_sample_payload = json.dumps(
                sorted([*development_ids, *holdout_ids]), separators=(",", ":")
            )
            declared_combined_hash = combined_contract.get("sample_id_membership_sha256")
            if (
                hashlib.sha256(combined_sample_payload.encode("utf-8")).hexdigest()
                != declared_combined_hash
            ):
                raise baseline_models.BaselineContractError(
                    "feature partitions do not reproduce the combined sample membership hash"
                )
            guard.checkpoint(declared + holdout_declared)
            return holdout

        try:
            fit_executor = baseline_models.DeadlineProcessFitExecutor(
                deadline_monotonic=(
                    time.monotonic()
                    + max(
                        0.0,
                        config.resources.max_wall_clock_seconds - (time.perf_counter() - started),
                    )
                ),
                resource_checkpoint=guard.checkpoint,
                private_directory=store.root,
                max_memory_bytes=guard.memory_bytes,
                telemetry_callback=guard.record_fit_rss,
            )
            training = baseline_models.train_baseline_models(
                development,
                config,
                holdout_loader=load_holdout,
                resource_checkpoint=guard.checkpoint,
                fit_executor=fit_executor,
            )
            if holdout_open_count != 1:
                raise RuntimeError("baseline trainer did not open holdout exactly once")
            guard.checkpoint(declared)
            prediction_bytes = baseline_models.prediction_parquet_bytes(training.predictions)
            validation_frame = pd.DataFrame.from_records(training.validation_scores)
            validation_bytes = baseline_models.validation_scores_parquet_bytes(validation_frame)
            model_bytes = pickle.dumps(training.models, protocol=5)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        except baseline_models.FitDeadlineExceeded:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="fit_wall_clock_deadline",
            )
        except baseline_models.FitMemoryExceeded:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="fit_memory_rss",
            )
        except baseline_models.BaselineContractError as error:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="data_or_split_contract",
                message=f"baseline input violates the data contract: {error}",
                resource_status="WITHIN_LIMIT",
            )
        restored: object = pickle.loads(model_bytes)
        if not isinstance(restored, dict) or set(restored) != set(training.models):
            raise RuntimeError("serialized baseline model bundle failed round-trip validation")
        selected_object = training.selection_manifest.get("selected")
        if not isinstance(selected_object, dict):
            raise RuntimeError("baseline selection manifest lost selected configurations")
        serialization_probe = development.loc[development["split"] == "train"].iloc[:3].copy()
        try:
            original_probe_predictions = baseline_models.predict_test_models(
                serialization_probe,
                training.models,
                tuple(config.evaluation.quantiles),
                selected_object,
                executor=fit_executor,
            )
            restored_predictions = baseline_models.predict_test_models(
                serialization_probe,
                restored,
                tuple(config.evaluation.quantiles),
                selected_object,
                executor=fit_executor,
            )
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        except baseline_models.FitDeadlineExceeded:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="fit_wall_clock_deadline",
            )
        except baseline_models.FitMemoryExceeded:
            return self._publish_baseline_resource_invalid(
                store=store,
                config=config,
                guard=guard,
                gate="fit_memory_rss",
            )
        prediction_columns = baseline_models.quantile_columns(tuple(config.evaluation.quantiles))
        if not (
            original_probe_predictions.loc[:, prediction_columns].to_numpy(dtype=float)
            == restored_predictions.loc[:, prediction_columns].to_numpy(dtype=float)
        ).all():
            raise RuntimeError("serialized baseline predictions changed after round-trip")
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        model_manifest = {
            "schema_version": "1.0",
            "format": "python_pickle_protocol_5",
            "warning": "load only artifacts produced by this trusted pilot run",
            "models": sorted(training.models),
            "models_sha256": model_sha256,
            "serialization_roundtrip_verified": True,
            "serialization_probe": "first_three_train_rows",
            "scientific_decision": "NOT_EVALUATED",
        }
        total_bytes = (
            declared
            + holdout_declared
            + len(prediction_bytes)
            + len(validation_bytes)
            + len(model_bytes)
        )
        try:
            guard.checkpoint(total_bytes)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        artifacts = (
            Path("train-baselines/selection_manifest.json"),
            Path("train-baselines/predictions.parquet"),
            Path("train-baselines/validation_scores.parquet"),
            Path("train-baselines/diagnostics.json"),
            Path("train-baselines/models/baseline_models.pkl"),
            Path("train-baselines/models/manifest.json"),
            Path("train-baselines/resource_ledger.json"),
        )
        result = StageResult(
            stage="train-baselines",
            status=StageStatus.SUCCESS,
            message="classical probabilistic baselines trained successfully",
            artifacts=artifacts,
            details=(
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
                DetailEntry(key="test_prediction_passes", value=1),
            ),
        )
        guard.sample_without_enforcement()
        ledger = {
            "schema_version": "1.0",
            "stage": "train-baselines",
            "status": StageStatus.SUCCESS.value,
            "resource_status": "WITHIN_LIMIT",
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_parent_rss_bytes": guard.peak_parent_rss_bytes,
            "peak_child_tree_rss_bytes": guard.peak_child_tree_rss_bytes,
            "peak_combined_rss_bytes": guard.peak_combined_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": config.resources.max_wall_clock_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
        }
        try:
            with store.stage(
                "train-baselines",
                input_paths=(
                    Path("build-splits/development_features.parquet"),
                    Path("build-splits/holdout_features.parquet"),
                    Path("build-splits/feature_manifest.json"),
                    Path("build-splits/minmax_scalers.json"),
                ),
                details={
                    "result_status": StageStatus.SUCCESS.value,
                    "scientific_decision": "NOT_EVALUATED",
                },
                commit_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_json("selection_manifest.json", training.selection_manifest)
                transaction.write_bytes("predictions.parquet", prediction_bytes)
                transaction.write_bytes("validation_scores.parquet", validation_bytes)
                transaction.write_json("diagnostics.json", training.diagnostics)
                transaction.write_bytes("models/baseline_models.pkl", model_bytes)
                transaction.write_json("models/manifest.json", model_manifest)
                transaction.write_json("stage_result.json", deterministic_data(result))
                transaction.write_json("resource_ledger.json", ledger)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        return result

    @staticmethod
    def _photonic_allowlist() -> dict[str, tuple[str, ...]]:
        return {
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        }

    @staticmethod
    def _publish_photonic_invalid(
        *,
        store: RunStore,
        guard: _ResourceGuard,
        gate: str,
        message: str,
        verified_input_hashes: Mapping[str, str] | None = None,
        prerequisite_manifest_hash: str | None = None,
    ) -> StageResult:
        empty_bytes = photonic_features.feature_parquet_bytes(
            pd.DataFrame(columns=photonic_features.feature_arrow_schema().names)
        )
        result = StageResult(
            stage="build-photonic-features",
            status=StageStatus.INVALID,
            message=message,
            artifacts=(
                Path("build-photonic-features/development_photonic_features.parquet"),
                Path("build-photonic-features/state_spaces.json"),
                Path("build-photonic-features/circuit_manifest.json"),
                Path("build-photonic-features/feature_manifest.json"),
                Path("build-photonic-features/resource_ledger.json"),
            ),
            details=(
                DetailEntry(key="resource_gate", value=gate),
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
            ),
        )
        ledger = {
            "schema_version": "1.0",
            "stage": "build-photonic-features",
            "status": "invalid",
            "resource_status": "EXCEEDED" if gate != "data_or_split_contract" else "WITHIN_LIMIT",
            "resource_gate": gate,
            "wall_clock_seconds": time.perf_counter() - guard.started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
            "scientific_decision": "NOT_EVALUATED",
            "verified_input_hashes": dict(verified_input_hashes or {}),
            "prerequisite_manifest_hash": prerequisite_manifest_hash,
        }
        with store.stage(
            "build-photonic-features",
            prerequisite_verification="marker_only",
            terminal_status="INVALID",
            details={
                "result_status": "invalid",
                "scientific_decision": "NOT_EVALUATED",
                "resource_gate": gate,
                "verified_input_hashes": dict(verified_input_hashes or {}),
                "prerequisite_manifest_hash": prerequisite_manifest_hash,
            },
        ) as transaction:
            transaction.write_bytes("development_photonic_features.parquet", empty_bytes)
            transaction.write_json("state_spaces.json", {"schema_version": "1.0", "spaces": []})
            transaction.write_json(
                "circuit_manifest.json",
                {"schema_version": "1.0", "status": "invalid", "circuits": []},
            )
            transaction.write_json(
                "feature_manifest.json",
                {
                    "schema_version": "1.0",
                    "status": "invalid",
                    "scientific_decision": "NOT_EVALUATED",
                    "row_count": 0,
                },
            )
            transaction.write_json("resource_ledger.json", ledger)
            transaction.write_json("stage_result.json", deterministic_data(result))
        return result

    def build_photonic_features(self, run_dir: Path) -> StageResult:
        started = time.perf_counter()
        store = RunStore.open(run_dir)
        existing = self._existing_photonic_result(store)
        if existing is not None:
            return existing
        config = self._run_config(store)
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        required_inputs = (
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        )
        verified_input_hashes: dict[str, str] = {}
        split_marker_hash: str | None = None
        try:
            split_marker_hash = store.verify_selected_stage_outputs("build-splits", required_inputs)
            split_stage_marker = self._read_object(store.root / "build-splits/stage_manifest.json")
            split_outputs = split_stage_marker.get("outputs")
            if not isinstance(split_outputs, list):
                raise photonic_features.PhotonicContractError(
                    "build-splits marker has invalid output records"
                )
            declared_output_hashes = {
                str(record.get("path")): str(record.get("sha256"))
                for record in split_outputs
                if isinstance(record, dict)
            }
            verified_input_hashes = {path: declared_output_hashes[path] for path in required_inputs}
            development_path = store.root / required_inputs[0]
            guard.checkpoint(development_path.stat().st_size)
            development = pds.dataset(development_path, format="parquet").to_table().to_pandas()
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
            split_manifest = self._read_object(store.root / required_inputs[1])
            scalers = self._read_object(store.root / required_inputs[2])
            expected_profiles = {
                "matched4_columns": list(MATCHED4_COLUMNS),
                "matched6_columns": list(MATCHED6_COLUMNS),
            }
            for key, expected in expected_profiles.items():
                if split_manifest.get(key) != expected:
                    raise photonic_features.PhotonicContractError(
                        f"feature manifest {key} violates the photonic contract"
                    )
            partitions = split_manifest.get("partitions")
            partition = partitions.get("development") if isinstance(partitions, dict) else None
            if not isinstance(partition, dict):
                raise photonic_features.PhotonicContractError(
                    "feature manifest has no development partition contract"
                )
            ordered_ids = (
                development.sort_values(list(PARTITION_ORDER_COLUMNS), kind="stable")["sample_id"]
                .astype(str)
                .tolist()
            )
            sample_hash = hashlib.sha256(
                json.dumps(
                    development["sample_id"].astype(str).tolist(), separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if (
                partition.get("path") != required_inputs[0]
                or partition.get("splits") != ["train", "validation"]
                or partition.get("row_count") != len(development)
                or partition.get("schema_sha256") != feature_schema_sha256(config)
                or partition.get("order_policy") != list(PARTITION_ORDER_COLUMNS)
                or partition.get("sample_id_order_sha256") != sample_hash
                or development["sample_id"].astype(str).tolist() != ordered_ids
            ):
                raise photonic_features.PhotonicContractError(
                    "development partition violates its immutable manifest"
                )
            for profile, columns in (
                ("matched4", MATCHED4_COLUMNS),
                ("matched6", MATCHED6_COLUMNS),
            ):
                record = scalers.get(profile)
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != FEATURE_SCHEMA_VERSION
                    or record.get("feature_order") != list(columns)
                    or record.get("fitted_split") != "train"
                ):
                    raise photonic_features.PhotonicContractError(
                        f"{profile} train-fitted scaler order is invalid"
                    )
            extraction = photonic_features.build_development_photonic_features(
                development,
                modes=tuple(config.photonics.modes),
                photons=config.photonics.primary_photons,
                phase_scales=tuple(config.photonics.phase_scales),
                circuit_seeds=tuple(config.seeds.circuit_seeds),
                resource_checkpoint=guard.checkpoint,
            )
            guard.checkpoint(declared)
            feature_bytes = photonic_features.feature_parquet_bytes(extraction.features)
            declared += len(feature_bytes)
            guard.checkpoint(declared)
            circuit_payloads: dict[str, bytes] = {}
            circuit_records: list[dict[str, object]] = []
            for key, circuit in sorted(extraction.circuits.items()):
                guard.checkpoint(declared)
                payload = photonic_features.circuit_npz_bytes(circuit)
                declared += len(payload)
                guard.checkpoint(declared)
                guard.checkpoint(declared)
                restored = photonic_features.load_circuit_npz(payload)
                guard.checkpoint(declared)
                if photonic_features.circuit_checksum(
                    restored
                ) != photonic_features.circuit_checksum(circuit):
                    raise RuntimeError("circuit NPZ checksum round-trip failed")
                circuit_payloads[f"circuits/{key}.npz"] = payload
                circuit_records.append(
                    {
                        "key": key,
                        "path": f"build-photonic-features/circuits/{key}.npz",
                        "circuit_seed": circuit.circuit_seed,
                        "modes": circuit.modes,
                        "photons": circuit.photons,
                        "v1_sha256": photonic_features.matrix_checksum(circuit.v1),
                        "v2_sha256": photonic_features.matrix_checksum(circuit.v2),
                        "circuit_sha256": photonic_features.circuit_checksum(circuit),
                        "npz_sha256": hashlib.sha256(payload).hexdigest(),
                        "state_order_sha256": circuit.state_order_sha256,
                    }
                )
            guard.checkpoint(declared)
        except _ResourceLimitExceeded as error:
            return self._publish_photonic_invalid(
                store=store,
                guard=guard,
                gate=error.gate,
                message="photonic feature extraction exceeded a configured resource cap",
                verified_input_hashes=verified_input_hashes,
                prerequisite_manifest_hash=split_marker_hash,
            )
        except photonic_features.PhotonicContractError as error:
            return self._publish_photonic_invalid(
                store=store,
                guard=guard,
                gate="data_or_split_contract",
                message=f"photonic input violates the data contract: {error}",
                verified_input_hashes=verified_input_hashes,
                prerequisite_manifest_hash=split_marker_hash,
            )

        state_spaces = {
            "schema_version": "1.0",
            "ordering": "ascending_lexicographic_occupation_tuple",
            "spaces": [
                {
                    "modes": modes,
                    "photons": config.photonics.primary_photons,
                    "dimension": photonic_features.fock_dimension(
                        modes, config.photonics.primary_photons
                    ),
                    "states": [
                        list(state)
                        for state in photonic_features.enumerate_fock_states(
                            modes, config.photonics.primary_photons
                        )
                    ],
                    "state_order_sha256": photonic_features.state_order_checksum(
                        photonic_features.enumerate_fock_states(
                            modes, config.photonics.primary_photons
                        )
                    ),
                }
                for modes in config.photonics.modes
            ],
        }
        if split_marker_hash is None or set(verified_input_hashes) != set(required_inputs):
            raise RuntimeError("verified photonic input metadata is incomplete")
        input_hashes = verified_input_hashes
        circuit_manifest = {
            "schema_version": "1.0",
            "status": "success",
            "backend": {"name": "SLOS", "class": "SLOSExqaliburBackend"},
            "versions": {
                "perceval-quandela": version("perceval-quandela"),
                "exqalibur": version("exqalibur"),
            },
            "construction": "V1 -> diag(exp(i*c*x)) -> V2",
            "matrix_convention": "V2 @ phase @ V1",
            "haar_generator": "numpy.Generator.normal+QR-phase-normalization",
            "matrix_checksum_contract": {
                "schema_version": "1.0",
                "dtype": "<c16",
                "shape": "uint64 dimensions",
                "order": "C",
                "aggregate": "ordered V1 and V2 matrix checksums",
            },
            "circuits": circuit_records,
        }
        feature_manifest = {
            "schema_version": photonic_features.PHOTONIC_FEATURE_SCHEMA_VERSION,
            "status": "success",
            "partition": "development_only",
            "holdout_opened": False,
            "exact": True,
            "ideal": True,
            "sampling": False,
            "loss_model": False,
            "qpu": False,
            "controls": ["indistinguishable", "distinguishable"],
            "modes": list(config.photonics.modes),
            "photons": config.photonics.primary_photons,
            "phase_scales": list(config.photonics.phase_scales),
            "circuit_seeds": list(config.seeds.circuit_seeds),
            "development_sample_count": len(development),
            "row_count": len(extraction.features),
            "expected_row_count": len(development)
            * len(config.photonics.modes)
            * len(config.photonics.phase_scales)
            * len(config.seeds.circuit_seeds)
            * 2,
            "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
            "split_marker_sha256": split_marker_hash,
            "input_hashes": input_hashes,
            "scaler_sha256": input_hashes["build-splits/minmax_scalers.json"],
            "scaled_feature_order": {
                "4": [f"matched4_scaled_{index}" for index in range(4)],
                "6": [f"matched6_scaled_{index}" for index in range(6)],
            },
            "probability_tolerance": photonic_features.PROBABILITY_TOLERANCE,
            "scientific_decision": "NOT_EVALUATED",
        }
        if feature_manifest["row_count"] != feature_manifest["expected_row_count"]:
            raise RuntimeError("photonic feature candidate matrix is incomplete")
        guard.checkpoint(declared)
        ledger = {
            "schema_version": "1.0",
            "stage": "build-photonic-features",
            "status": "success",
            "resource_status": "WITHIN_LIMIT",
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
            "configurations": list(extraction.configuration_ledger),
        }
        result = StageResult(
            stage="build-photonic-features",
            status=StageStatus.SUCCESS,
            message="exact development photonic features built successfully",
            artifacts=(
                Path("build-photonic-features/development_photonic_features.parquet"),
                Path("build-photonic-features/state_spaces.json"),
                Path("build-photonic-features/circuit_manifest.json"),
                Path("build-photonic-features/feature_manifest.json"),
                Path("build-photonic-features/resource_ledger.json"),
            ),
            details=(
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
                DetailEntry(key="holdout_opened", value=False),
            ),
        )
        try:
            with store.stage(
                "build-photonic-features",
                input_paths=required_inputs,
                prerequisite_verification="selected_outputs",
                prerequisite_output_allowlist=self._photonic_allowlist(),
                details={
                    "result_status": "success",
                    "scientific_decision": "NOT_EVALUATED",
                    "holdout_opened": False,
                },
                commit_guard=guard.checkpoint,
                private_write_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_bytes("development_photonic_features.parquet", feature_bytes)
                transaction.write_json("state_spaces.json", state_spaces)
                transaction.write_json("circuit_manifest.json", circuit_manifest)
                for path, payload in sorted(circuit_payloads.items()):
                    transaction.write_bytes(path, payload)
                transaction.write_json("feature_manifest.json", feature_manifest)
                guard.checkpoint(declared)
                ledger.update(
                    {
                        "wall_clock_seconds": time.perf_counter() - started,
                        "peak_rss_bytes": guard.peak_rss_bytes,
                        "peak_declared_bytes": guard.peak_declared_bytes,
                    }
                )
                transaction.write_json("resource_ledger.json", ledger)
                transaction.write_json("stage_result.json", deterministic_data(result))
        except _ResourceLimitExceeded as error:
            return self._publish_photonic_invalid(
                store=store,
                guard=guard,
                gate=error.gate,
                message="photonic feature publication exceeded a configured resource cap",
                verified_input_hashes=verified_input_hashes,
                prerequisite_manifest_hash=split_marker_hash,
            )
        return result

    def benchmark(self, run_dir: Path) -> StageResult:
        started = time.perf_counter()
        store = RunStore.open(run_dir)
        existing = self._existing_benchmark_result(store)
        if existing is not None:
            return existing
        config = self._run_config(store)
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        photonic_allowlist = self._photonic_allowlist()
        development_relative = "build-photonic-features/development_photonic_features.parquet"
        circuit_manifest_relative = "build-photonic-features/circuit_manifest.json"
        feature_manifest_relative = "build-photonic-features/feature_manifest.json"
        state_spaces_relative = "build-photonic-features/state_spaces.json"
        development_reference_relative = "build-splits/development_features.parquet"
        try:
            # Holdout-safe boundary: this verifies Task 7 and only its fixed development
            # prerequisites. It cannot hash/materialize the Task 5 holdout partition.
            photonic_marker_hash = store.verify_terminal_stage(
                "build-photonic-features",
                accepted_statuses=frozenset({"SUCCESS"}),
                prerequisite_output_allowlist=photonic_allowlist,
            )
            task7_manifest = self._read_object(
                store.root / "build-photonic-features/stage_manifest.json"
            )
            task7_outputs = task7_manifest.get("outputs")
            if not isinstance(task7_outputs, list):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "photonic stage output manifest is invalid"
                )
            task7_hashes = {
                str(record.get("path")): str(record.get("sha256"))
                for record in task7_outputs
                if isinstance(record, dict)
            }
            for relative in (
                development_relative,
                circuit_manifest_relative,
                feature_manifest_relative,
                state_spaces_relative,
            ):
                if relative not in task7_hashes:
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        f"photonic stage did not declare {relative}"
                    )
            development_path = store.root / development_relative
            guard.checkpoint(development_path.stat().st_size)
            development = pds.dataset(development_path, format="parquet").to_table().to_pandas()
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
            task7_inputs = task7_manifest.get("input_hashes")
            if not isinstance(task7_inputs, dict) or task7_inputs.get(
                development_reference_relative
            ) != sha256_file(store.root / development_reference_relative):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "Task 5 development reference is not bound to Task 7"
                )
            development_reference = (
                pds.dataset(store.root / development_reference_relative, format="parquet")
                .to_table()
                .to_pandas()
            )
            declared += int(development_reference.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
            task7_feature_manifest = self._read_object(store.root / feature_manifest_relative)
            if (
                task7_feature_manifest.get("holdout_opened") is not False
                or task7_feature_manifest.get("photons") != 2
                or task7_feature_manifest.get("circuit_seeds") != list(config.seeds.circuit_seeds)
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "development photonic manifest violates the G3 firewall"
                )
            raw_state_spaces = self._read_object(store.root / state_spaces_relative).get("spaces")
            if not isinstance(raw_state_spaces, list):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "trusted state-space manifest is invalid"
                )
            trusted_state_spaces: dict[tuple[int, int], tuple[int, str]] = {}
            for record in raw_state_spaces:
                if not isinstance(record, dict):
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "trusted state-space record is invalid"
                    )
                key = (int(record.get("modes", -1)), int(record.get("photons", -1)))
                value = (
                    int(record.get("dimension", -1)),
                    str(record.get("state_order_sha256", "")),
                )
                if key in trusted_state_spaces:
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "trusted state-space record is duplicated"
                    )
                trusted_state_spaces[key] = value
            circuit_manifest = self._read_object(store.root / circuit_manifest_relative)
            circuit_records = circuit_manifest.get("circuits")
            if not isinstance(circuit_records, list):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "circuit manifest is invalid"
                )
            trusted_circuits: dict[tuple[int, int], photonic_features.CircuitDefinition] = {}
            trusted_circuit_paths: list[str] = []
            for record in circuit_records:
                if not isinstance(record, dict):
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "circuit manifest record is invalid"
                    )
                relative = str(record.get("path"))
                if relative not in task7_hashes:
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "circuit archive is not a verified Task 7 output"
                    )
                payload = (store.root / relative).read_bytes()
                declared += len(payload)
                guard.checkpoint(declared)
                if hashlib.sha256(payload).hexdigest() != record.get("npz_sha256"):
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "trusted circuit archive checksum mismatch"
                    )
                circuit = photonic_features.load_circuit_npz(payload)
                key = (circuit.modes, circuit.circuit_seed)
                if (
                    key in trusted_circuits
                    or circuit.photons != 2
                    or photonic_features.circuit_checksum(circuit) != record.get("circuit_sha256")
                ):
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        "trusted circuit definition is invalid or duplicated"
                    )
                trusted_circuits[key] = circuit
                trusted_circuit_paths.append(relative)
            fit_executor = baseline_models.DeadlineProcessFitExecutor(
                deadline_monotonic=(
                    time.monotonic()
                    + max(
                        0.0,
                        config.resources.max_wall_clock_seconds - (time.perf_counter() - started),
                    )
                ),
                resource_checkpoint=guard.checkpoint,
                private_directory=store.root,
                max_memory_bytes=guard.memory_bytes,
                telemetry_callback=guard.record_fit_rss,
            )
            selection = photonic_benchmark.select_photonic_models(
                development,
                config,
                development_reference=development_reference,
                trusted_circuits=trusted_circuits,
                trusted_state_spaces=trusted_state_spaces,
                fit_executor=fit_executor,
                resource_checkpoint=guard.checkpoint,
            )
            freeze_hash = selection.selection_manifest["selection_freeze_sha256"]
            if not isinstance(freeze_hash, str) or len(freeze_hash) != 64:
                raise RuntimeError("photonic selection did not produce a freeze hash")

            # All holdout-bound verification is deliberately below the immutable freeze.
            guard.checkpoint(declared)
            baseline_marker_hash = store.verify_terminal_stage(
                "train-baselines", accepted_statuses=frozenset({"SUCCESS"})
            )
            baseline_predictions_relative = "train-baselines/predictions.parquet"
            baseline_predictions_path = store.root / baseline_predictions_relative
            baseline_predictions_sha256 = sha256_file(baseline_predictions_path)

            split_marker_hash = store.verify_terminal_stage(
                "build-splits", accepted_statuses=frozenset({"SUCCESS"})
            )
            split_manifest = self._read_object(store.root / "build-splits/feature_manifest.json")
            split_stage_manifest = self._read_object(
                store.root / "build-splits/stage_manifest.json"
            )
            raw_split_outputs = split_stage_manifest.get("outputs")
            if not isinstance(raw_split_outputs, list):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "split stage output manifest is invalid"
                )
            split_hashes = {
                str(record.get("path")): str(record.get("sha256"))
                for record in raw_split_outputs
                if isinstance(record, dict)
            }
            holdout_relative = "build-splits/holdout_features.parquet"
            holdout_path = store.root / holdout_relative
            guard.checkpoint(declared)
            if split_hashes.get(holdout_relative) != sha256_file(holdout_path):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "holdout checksum changed after selection freeze"
                )
            partitions = split_manifest.get("partitions")
            holdout_contract = partitions.get("holdout") if isinstance(partitions, dict) else None
            if not isinstance(holdout_contract, dict):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "holdout partition contract is missing"
                )
            # Exactly one dataframe materialization of the test partition.
            guard.checkpoint(declared)
            holdout = pds.dataset(holdout_path, format="parquet").to_table().to_pandas()
            holdout_declared = int(holdout.memory_usage(index=True, deep=True).sum())
            declared += holdout_declared
            guard.checkpoint(declared)
            ordered_ids = (
                holdout.sort_values(list(PARTITION_ORDER_COLUMNS), kind="stable")["sample_id"]
                .astype(str)
                .tolist()
            )
            if (
                holdout_contract.get("path") != holdout_relative
                or holdout_contract.get("row_count") != len(holdout)
                or holdout_contract.get("schema_sha256") != feature_schema_sha256(config)
                or holdout_contract.get("splits") != ["test"]
                or holdout["sample_id"].astype(str).tolist() != ordered_ids
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "holdout partition violates its immutable contract"
                )
            development_ids = set(development["sample_id"].astype(str))
            if development_ids & set(holdout["sample_id"].astype(str)):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "development and holdout sample IDs overlap"
                )

            selected_modes = int(selection.selected["modes"])
            circuits = {
                seed: trusted_circuits[(selected_modes, seed)]
                for seed in config.seeds.circuit_seeds
            }

            validated_exact_features = photonic_benchmark.build_holdout_photonic_features(
                holdout,
                selected=selection.selected,
                circuits=circuits,
                trusted_state_space=trusted_state_spaces[(selected_modes, 2)],
                resource_checkpoint=guard.checkpoint,
            )
            exact_features = validated_exact_features.frame
            declared += int(exact_features.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(declared)
            expected_exact_feature_rows = len(holdout) * len(config.seeds.circuit_seeds) * 2
            if len(exact_features) != expected_exact_feature_rows:
                raise RuntimeError("exact holdout photonic matrix is incomplete")
            exact_predictions, exact_diagnostics = photonic_benchmark.predict_exact(
                validated_exact_features,
                selection.selected_models,
                tuple(config.evaluation.quantiles),
                fit_executor=fit_executor,
                resource_checkpoint=guard.checkpoint,
            )
            if len(exact_predictions) != expected_exact_feature_rows:
                raise RuntimeError("exact holdout predictions are incomplete")
            finite_counts = photonic_benchmark.finite_shot_screen(
                exact_features,
                tuple(config.photonics.raw_attempt_budgets),
                SeedManager(config.seeds.master_seed),
            )
            expected_finite_rows = (
                len(holdout)
                * len(config.seeds.circuit_seeds)
                * len(config.photonics.raw_attempt_budgets)
            )
            if len(finite_counts) != expected_finite_rows:
                raise RuntimeError("finite-shot count matrix is incomplete")
            finite_predictions = photonic_benchmark.predict_finite_shot(
                finite_counts,
                selection.selected_models,
                tuple(config.evaluation.quantiles),
                fit_executor=fit_executor,
            )
            if len(finite_predictions) != expected_finite_rows:
                raise RuntimeError("finite-shot predictions are incomplete")

            validation_bytes = photonic_benchmark.parquet_bytes(
                selection.validation_scores,
                photonic_benchmark.validation_score_arrow_schema(),
            )
            feature_bytes = photonic_features.feature_parquet_bytes(exact_features)
            exact_prediction_bytes = photonic_benchmark.parquet_bytes(
                exact_predictions, photonic_benchmark.exact_prediction_arrow_schema()
            )
            finite_count_bytes = photonic_benchmark.parquet_bytes(
                finite_counts, photonic_benchmark.finite_shot_count_arrow_schema()
            )
            finite_prediction_bytes = photonic_benchmark.parquet_bytes(
                finite_predictions,
                photonic_benchmark.finite_shot_prediction_arrow_schema(),
            )
            model_bytes = photonic_benchmark.model_pickle_bytes(selection.selected_models)
            restored = pickle.loads(model_bytes)
            if not isinstance(restored, dict) or set(restored) != set(selection.selected_models):
                raise RuntimeError("photonic model serialization round-trip failed")
            for key, model in restored.items():
                original = selection.selected_models[key]
                if (
                    model.circuit_sha256 != original.circuit_sha256
                    or model.alpha != original.alpha
                    or model.modes != original.modes
                ):
                    raise RuntimeError("photonic model metadata changed after serialization")
            output_bytes = sum(
                map(
                    len,
                    (
                        validation_bytes,
                        feature_bytes,
                        exact_prediction_bytes,
                        finite_count_bytes,
                        finite_prediction_bytes,
                        model_bytes,
                    ),
                )
            )
            guard.checkpoint(declared + output_bytes)
        except _ResourceLimitExceeded as error:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate=f"resource_{error.gate}",
                message="photonic benchmark exceeded a configured resource cap",
            )
        except baseline_models.FitDeadlineExceeded:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate="resource_fit_wall_clock_deadline",
                message="photonic benchmark fit exceeded its hard deadline",
            )
        except baseline_models.FitMemoryExceeded:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate="resource_fit_memory_rss",
                message="photonic benchmark fit exceeded aggregate RSS",
            )
        except photonic_benchmark.PhotonicBenchmarkContractError as error:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate="data_or_artifact_contract",
                message=f"photonic benchmark input is invalid: {error}",
            )

        selection_manifest = dict(selection.selection_manifest)
        selection_manifest.update(
            {
                "holdout_audit": {
                    "opened_after_selection_freeze_sha256": freeze_hash,
                    "materialization_count": 1,
                    "prediction_passes": 1,
                    "row_count": len(holdout),
                    "sample_id_order_sha256": hashlib.sha256(
                        json.dumps(
                            holdout["sample_id"].astype(str).tolist(), separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                },
                "prerequisite_manifest_hashes_after_freeze": {
                    "build_photonic_features": photonic_marker_hash,
                    "build_splits": split_marker_hash,
                    "train_baselines": baseline_marker_hash,
                },
                "baseline_predictions": {
                    "path": baseline_predictions_relative,
                    "sha256": baseline_predictions_sha256,
                    "policy": "immutable_reuse_no_retrain_no_repredict",
                },
                "scientific_decision": "NOT_EVALUATED",
            }
        )
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        model_manifest = {
            "schema_version": "1.0",
            "format": "python_pickle_protocol_5",
            "warning": "load only artifacts produced by this trusted pilot run",
            "models_sha256": model_sha256,
            "model_keys": [
                f"{control}:seed={seed}" for control, seed in sorted(selection.selected_models)
            ],
            "serialization_roundtrip_verified": True,
            "holdout_reinference_for_serialization": False,
            "selection_freeze_sha256": freeze_hash,
        }
        diagnostics = {
            **selection.diagnostics,
            "exact_holdout": exact_diagnostics,
            "exact_prediction_rows": len(exact_predictions),
            "finite_shot_rows": len(finite_predictions),
            "finite_shot": {
                "budgets": list(config.photonics.raw_attempt_budgets),
                "ideal_shot_noise_only": True,
                "loss_modeled": False,
                "used_for_selection": False,
                "used_for_go_stop": False,
            },
        }
        guard.sample_without_enforcement()
        ledger = {
            "schema_version": "1.0",
            "stage": "benchmark",
            "status": "success",
            "resource_status": "WITHIN_LIMIT",
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_parent_rss_bytes": guard.peak_parent_rss_bytes,
            "peak_child_tree_rss_bytes": guard.peak_child_tree_rss_bytes,
            "peak_combined_rss_bytes": guard.peak_combined_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
        }
        artifacts = (
            Path("benchmark/selection_manifest.json"),
            Path("benchmark/validation_scores.parquet"),
            Path("benchmark/holdout_photonic_features.parquet"),
            Path("benchmark/exact_predictions.parquet"),
            Path("benchmark/finite_shot_counts.parquet"),
            Path("benchmark/finite_shot_predictions.parquet"),
            Path("benchmark/models/photonic_models.pkl"),
            Path("benchmark/models/manifest.json"),
            Path("benchmark/diagnostics.json"),
            Path("benchmark/resource_ledger.json"),
        )
        result = StageResult(
            stage="benchmark",
            status=StageStatus.SUCCESS,
            message="photonic G3 benchmark completed successfully",
            artifacts=artifacts,
            details=(
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
                DetailEntry(key="selection_freeze_sha256", value=freeze_hash),
                DetailEntry(key="test_prediction_passes", value=1),
            ),
        )
        input_paths = (
            development_relative,
            circuit_manifest_relative,
            feature_manifest_relative,
            state_spaces_relative,
            development_reference_relative,
            *trusted_circuit_paths,
            holdout_relative,
            "build-splits/feature_manifest.json",
            baseline_predictions_relative,
        )
        try:
            with store.stage(
                "benchmark",
                input_paths=input_paths,
                details={
                    "result_status": "success",
                    "scientific_decision": "NOT_EVALUATED",
                    "selection_freeze_sha256": freeze_hash,
                    "holdout_materialization_count": 1,
                },
                commit_guard=guard.checkpoint,
                private_write_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_json("selection_manifest.json", selection_manifest)
                transaction.write_bytes("validation_scores.parquet", validation_bytes)
                transaction.write_bytes("holdout_photonic_features.parquet", feature_bytes)
                transaction.write_bytes("exact_predictions.parquet", exact_prediction_bytes)
                transaction.write_bytes("finite_shot_counts.parquet", finite_count_bytes)
                transaction.write_bytes("finite_shot_predictions.parquet", finite_prediction_bytes)
                transaction.write_bytes("models/photonic_models.pkl", model_bytes)
                transaction.write_json("models/manifest.json", model_manifest)
                transaction.write_json("diagnostics.json", diagnostics)
                transaction.write_json("resource_ledger.json", ledger)
                transaction.write_json("stage_result.json", deterministic_data(result))
        except _ResourceLimitExceeded as error:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate=f"resource_{error.gate}",
                message="photonic benchmark publication exceeded a configured resource cap",
            )
        return result

    def report(self, run_dir: Path) -> None:
        self._unimplemented("report")

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None:
        self._unimplemented("run-pilot")


def init_run(config: Path) -> Path:
    """Initialize a run from a config and return its new immutable directory."""

    return DefaultPilotServices().init_run(config)


__all__ = ["DefaultPilotServices", "init_run"]
