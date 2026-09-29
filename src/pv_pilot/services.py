"""Application services for the staged pilot command line."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import psutil

from . import baselines as baseline_models
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
from .splits import (
    FEATURE_SCHEMA_VERSION,
    MATCHED4_COLUMNS,
    MATCHED6_COLUMNS,
    SplitContractError,
    build_feature_splits,
    diagnostic_split_parquet_bytes,
    empty_clip_metrics,
    feature_dataframe_parquet_bytes,
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

    def checkpoint(self, declared_bytes: int = 0) -> None:
        self.peak_declared_bytes = max(self.peak_declared_bytes, declared_bytes)
        rss = psutil.Process().memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        if time.perf_counter() - self.started > self.wall_seconds:
            raise _ResourceLimitExceeded("wall_clock")
        if rss > self.memory_bytes:
            raise _ResourceLimitExceeded("memory_rss")
        if declared_bytes > self.memory_bytes:
            raise _ResourceLimitExceeded("dataframe_or_output_bytes")

    def sample_without_enforcement(self) -> None:
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, psutil.Process().memory_info().rss
        )


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
        store.verify_terminal_stage(
            "validate-data", accepted_statuses=frozenset({str(status)})
        )
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
        store.verify_terminal_stage(
            "build-splits", accepted_statuses=frozenset({str(status)})
        )
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
        store.verify_terminal_stage(
            "train-baselines", accepted_statuses=frozenset({str(status)})
        )
        result_path = store.root / "train-baselines/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        message = (
            "train-baselines already successful"
            if status == "SUCCESS"
            else "train-baselines already invalid"
        )
        return result.model_copy(update={"message": message})

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
        empty_scores = pd.DataFrame(
            columns=baseline_models.validation_score_arrow_schema().names
        )
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
            "peak_declared_bytes": guard.peak_declared_bytes,
            "cpu_only": True,
        }
        with store.stage(
            "train-baselines",
            input_paths=(
                Path("build-splits/feature_matrix.parquet"),
                Path("build-splits/feature_manifest.json"),
                Path("build-splits/minmax_scalers.json"),
            ),
            terminal_status="INVALID",
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
            str(key): int(value)
            for key, value in sorted(Counter(canonical["site_id"]).items())
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
            details=tuple(
                DetailEntry(key=key, value=value) for key, value in details.items()
            ),
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
            exclusion_bytes = split_exclusion_dataframe_parquet_bytes(build.exclusions)
            guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
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
                DetailEntry(
                    key="split_counts", value=build.feature_manifest["split_counts"]
                ),
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
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_bytes("exclusions.parquet", exclusion_bytes)
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_json("feature_manifest.json", build.feature_manifest)
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_json("minmax_scalers.json", build.scalers)
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_json("clip_rates.json", build.clip_rates)
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_json("stage_result.json", deterministic_data(result))
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
                transaction.write_json("resource_ledger.json", ledger)
                guard.checkpoint(declared + len(feature_bytes) + len(exclusion_bytes))
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
        store.verify_terminal_stage("build-splits")
        config = self._run_config(store)
        feature_path = store.root / "build-splits/feature_matrix.parquet"
        manifest_path = store.root / "build-splits/feature_manifest.json"
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        try:
            guard.checkpoint(feature_path.stat().st_size)
            features = pd.read_parquet(feature_path)
            declared = int(features.memory_usage(index=True, deep=True).sum())
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
                    message=(
                        f"feature manifest {key} does not match the frozen baseline contract"
                    ),
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

        try:
            training = baseline_models.train_baseline_models(features, config)
            guard.checkpoint(declared)
            prediction_bytes = baseline_models.prediction_parquet_bytes(training.predictions)
            validation_frame = pd.DataFrame.from_records(training.validation_scores)
            validation_bytes = baseline_models.validation_scores_parquet_bytes(validation_frame)
            model_bytes = pickle.dumps(training.models, protocol=5)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
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
        serialization_probe = features.loc[features["split"] == "train"].iloc[:3].copy()
        original_probe_predictions = baseline_models.predict_test_models(
            serialization_probe,
            training.models,
            tuple(config.evaluation.quantiles),
            selected_object,
        )
        restored_predictions = baseline_models.predict_test_models(
            serialization_probe,
            restored,
            tuple(config.evaluation.quantiles),
            selected_object,
        )
        prediction_columns = baseline_models.quantile_columns(
            tuple(config.evaluation.quantiles)
        )
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
        total_bytes = declared + len(prediction_bytes) + len(validation_bytes) + len(model_bytes)
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
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": config.resources.max_wall_clock_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
        }
        try:
            with store.stage(
                "train-baselines",
                input_paths=(
                    Path("build-splits/feature_matrix.parquet"),
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

    def build_photonic_features(self, run_dir: Path) -> None:
        self._unimplemented("build-photonic-features")

    def benchmark(self, run_dir: Path) -> None:
        self._unimplemented("benchmark")

    def report(self, run_dir: Path) -> None:
        self._unimplemented("report")

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None:
        self._unimplemented("run-pilot")


def init_run(config: Path) -> Path:
    """Initialize a run from a config and return its new immutable directory."""

    return DefaultPilotServices().init_run(config)


__all__ = ["DefaultPilotServices", "init_run"]
