"""Application services for the staged pilot command line."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

from .artifacts import RunStore, sha256_file
from .config import PilotConfig, load_config
from .data import (
    CANONICAL_SCHEMA_VERSION,
    DataContractError,
    DataValidationResult,
    canonical_checksum,
    canonicalize_data,
    empty_canonical_frame,
    empty_exclusion_frame,
)
from .data import dataframe_parquet_bytes as dataframe_parquet_bytes
from .domain import DetailEntry, StageResult, StageStatus, deterministic_data


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
        result_path = store.root / "validate-data/stage_result.json"
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if status == "SUCCESS":
            store.verify_terminal_stage("validate-data")
            return result.model_copy(update={"message": "validate-data already successful"})
        if status not in {"BLOCKED", "INVALID"}:
            raise ValueError("validate-data has an invalid terminal marker")
        store.verify_terminal_stage(
            "validate-data", accepted_statuses=frozenset({str(status)})
        )
        return result.model_copy(update={"message": f"validate-data already {status.lower()}"})

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
                result = canonicalize_data(source_path, config.data, config.data.format)
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

        elapsed = time.perf_counter() - started
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
            "input_sha256": expected_hash,
            "provenance": config.protocol.provenance.value,
            "schema_version": "1.0",
            "scientific_decision": "NOT_EVALUATED",
            "site_counts": site_counts,
            "start_utc": start_utc,
            "status": status.value,
            "synthetic": config.protocol.provenance.value == "synthetic",
        }
        ledger = {
            "elapsed_seconds": elapsed,
            "human_cleaning_hours": None,
            "schema_version": "1.0",
            "status": status.value,
            "timebox_seconds": config.resources.max_wall_clock_seconds,
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
        with store.stage(
            "validate-data",
            external_input_hashes=(
                {source_path: expected_hash}
                if register_external_input and isinstance(expected_hash, str)
                else None
            ),
            terminal_status=terminal,
            details={"result_status": status.value, "scientific_decision": "NOT_EVALUATED"},
        ) as transaction:
            transaction.write_bytes("canonical.parquet", dataframe_parquet_bytes(canonical))
            transaction.write_bytes("qc_exclusions.parquet", dataframe_parquet_bytes(exclusions))
            transaction.write_json("data_card.json", card)
            transaction.write_json("cleaning_ledger.json", ledger)
            transaction.write_json("stage_result.json", deterministic_data(stage_result))
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
