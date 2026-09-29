"""Application services for the staged pilot command line."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import psutil

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

    def build_splits(self, run_dir: Path) -> None:
        self._unimplemented("build-splits")

    def train_baselines(self, run_dir: Path) -> None:
        self._unimplemented("train-baselines")

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
