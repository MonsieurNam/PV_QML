"""Application services for the staged pilot command line."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pickle
import shutil
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from . import baselines as baseline_models
from . import benchmark as photonic_benchmark
from . import photonics as photonic_features
from . import three_photon as optional_photonics
from .artifacts import RunStore, authenticated_snapshot, sha256_file
from .config import HUMAN_CLEANING_TIMEBOX_HOURS, PilotConfig, load_config
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
from .domain import (
    CompletionKind,
    DataProvenance,
    DetailEntry,
    StageResult,
    StageStatus,
    deterministic_data,
)
from .evaluation import EvaluationContractError
from .report_worker import (
    close_descriptor_transfer,
    duplicate_descriptor_for_worker,
    run_report_worker,
)
from .reporting import ReportContractError, report_parquet_schemas
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
    sample_contiguity_policy,
    split_exclusion_dataframe_parquet_bytes,
)

_ReportProcessExecutor = baseline_models.DeadlineProcessFitExecutor
_REPORT_STATUS_DECISIONS = {
    StageStatus.BLOCKED: frozenset({"not_evaluated"}),
    StageStatus.SUCCESS: frozenset({"go", "stop", "inconclusive"}),
    StageStatus.INVALID: frozenset({"not_evaluated"}),
}
_REPORT_DECISION_CONCLUSIONS = {
    (StageStatus.BLOCKED, "not_evaluated"): "not_evaluated",
    (StageStatus.SUCCESS, "go"): "go",
    (StageStatus.SUCCESS, "stop"): "skill_below_threshold",
    (StageStatus.SUCCESS, "inconclusive"): "inconclusive",
    (StageStatus.INVALID, "not_evaluated"): "resource_gate_failed",
}


class _ResourceLimitExceeded(RuntimeError):
    def __init__(self, gate: str) -> None:
        super().__init__(f"configured resource cap exceeded: {gate}")
        self.gate = gate


class _HoldoutAccessIndeterminate(RuntimeError):
    """A prior process may have physically opened the holdout without caching it."""


@dataclass(frozen=True, slots=True)
class _MaterializedHoldout:
    """One physical holdout read plus the exact authenticated source snapshot."""

    frame: pd.DataFrame
    source_bytes: bytes


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

    @staticmethod
    def _after_benchmark_holdout_materialized() -> None:
        """Crash-injection seam immediately after the sole physical source access."""

    @staticmethod
    def _after_benchmark_holdout_cache_complete() -> None:
        """Crash-injection seam after cache and COMPLETE are durable."""

    @staticmethod
    def _safe_private_bytes(path: Path, *, root: Path, max_bytes: int) -> bytes:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError("benchmark holdout journal escapes the run directory")
        path_stat = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
            or path_stat.st_size > max_bytes
        ):
            raise ValueError("benchmark holdout journal file is unsafe")
        return authenticated_snapshot(path, root=root)

    @classmethod
    def _safe_private_object(cls, path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
        try:
            captured = cls._safe_private_bytes(path, root=root, max_bytes=64 * 1024)
            payload: object = json.loads(captured.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("benchmark holdout journal JSON is invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("benchmark holdout journal JSON must be an object")
        return payload, hashlib.sha256(captured).hexdigest()

    @staticmethod
    def _holdout_sample_order_sha256(frame: pd.DataFrame) -> str:
        payload = json.dumps(frame["sample_id"].astype(str).tolist(), separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _materialize_benchmark_holdout_source(
        self,
        path: Path,
        *,
        expected_sha256: str,
        guard: _ResourceGuard,
        declared_bytes: int,
    ) -> _MaterializedHoldout:
        """Read, checksum, and decode the immutable source through one file open."""

        del self
        payload = authenticated_snapshot(
            path,
            expected_sha256=expected_sha256,
            checkpoint=guard.checkpoint,
            live_bytes=declared_bytes,
        )
        live_bytes = declared_bytes + len(payload)
        guard.checkpoint(live_bytes)
        frame = pd.read_parquet(io.BytesIO(payload))
        guard.checkpoint(live_bytes + int(frame.memory_usage(index=True, deep=True).sum()))
        return _MaterializedHoldout(frame=frame, source_bytes=payload)

    def _load_or_materialize_benchmark_holdout(
        self,
        *,
        store: RunStore,
        config: PilotConfig,
        source_relative: str,
        source_sha256: str,
        split_marker_sha256: str,
        selection_freeze_sha256: str,
        guard: _ResourceGuard,
        declared_bytes: int,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
        access_relative = ".benchmark-holdout-access"
        access_root = store.root / access_relative
        if access_root.exists() or access_root.is_symlink():
            is_junction = getattr(access_root, "is_junction", lambda: False)()
            if (
                access_root.is_symlink()
                or is_junction
                or not access_root.is_dir()
                or not access_root.resolve().is_relative_to(store.root)
            ):
                raise ValueError("benchmark holdout access directory is unsafe")
        expected_identity = {
            "schema_version": "1.0",
            "run_id": store.root.name,
            "run_config_sha256": store.config_hash,
            "stage": "benchmark",
            "source_path": source_relative,
            "source_sha256": source_sha256,
            "source_marker_sha256": split_marker_sha256,
            "selection_freeze_sha256": selection_freeze_sha256,
        }
        intent_path = access_root / "intent.json"
        cache_path = access_root / "cache.parquet"
        complete_path = access_root / "complete.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent, intent_sha256 = self._safe_private_object(intent_path, root=store.root)
            if any(intent.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("benchmark holdout intent identity changed")
            access_id = intent.get("access_id")
            if not isinstance(access_id, str) or len(access_id) != 32:
                raise ValueError("benchmark holdout intent access ID is invalid")
            if intent.get("transaction_id") != access_id:
                raise ValueError("benchmark holdout transaction ID is invalid")
            if intent.get("state") != "INTENT":
                raise ValueError("benchmark holdout intent state is invalid")
            if not complete_path.exists() or complete_path.is_symlink():
                raise _HoldoutAccessIndeterminate(
                    "holdout INTENT exists without durable COMPLETE; a new run is required"
                )
            complete, complete_sha256 = self._safe_private_object(complete_path, root=store.root)
            expected_complete_identity = {
                **expected_identity,
                "access_id": access_id,
                "transaction_id": access_id,
                "state": "COMPLETE",
            }
            if any(complete.get(key) != value for key, value in expected_complete_identity.items()):
                raise ValueError("benchmark holdout COMPLETE identity changed")
            cache_bytes = self._safe_private_bytes(
                cache_path, root=store.root, max_bytes=guard.memory_bytes
            )
            guard.checkpoint(declared_bytes + len(cache_bytes))
            cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
            if cache_sha256 != source_sha256:
                raise ValueError(
                    "benchmark holdout cache is not anchored to the trusted source checksum"
                )
            if complete.get(
                "cache_kind"
            ) != "exact_source_snapshot" or cache_sha256 != complete.get("cache_sha256"):
                raise ValueError("benchmark holdout cache checksum mismatch")
            holdout = pd.read_parquet(io.BytesIO(cache_bytes))
            guard.checkpoint(
                declared_bytes
                + len(cache_bytes)
                + int(holdout.memory_usage(index=True, deep=True).sum())
            )
            if (
                complete.get("row_count") != len(holdout)
                or complete.get("schema_sha256") != feature_schema_sha256(config)
                or complete.get("sample_id_order_sha256")
                != self._holdout_sample_order_sha256(holdout)
            ):
                raise ValueError("benchmark holdout cache semantic contract mismatch")
            audit = dict(complete)
            audit["source_materializations"] = 1
            audit["resume_source_reopened"] = False
            return (
                holdout,
                audit,
                {
                    f"{access_relative}/intent.json": intent_sha256,
                    f"{access_relative}/cache.parquet": source_sha256,
                    f"{access_relative}/complete.json": complete_sha256,
                },
            )

        if cache_path.exists() or complete_path.exists():
            raise ValueError("benchmark holdout cache exists without INTENT")
        access_id = uuid4().hex
        intent = {
            **expected_identity,
            "access_id": access_id,
            "transaction_id": access_id,
            "state": "INTENT",
        }
        guard.checkpoint(declared_bytes)
        intent_bytes = self._benchmark_json_bytes(intent)
        intent_sha256 = hashlib.sha256(intent_bytes).hexdigest()
        store.write_bytes(f"{access_relative}/intent.json", intent_bytes)
        guard.checkpoint(declared_bytes)
        materialized = self._materialize_benchmark_holdout_source(
            store.root / source_relative,
            expected_sha256=source_sha256,
            guard=guard,
            declared_bytes=declared_bytes,
        )
        holdout = materialized.frame
        self._after_benchmark_holdout_materialized()
        holdout_memory = int(holdout.memory_usage(index=True, deep=True).sum())
        holdout_bytes = materialized.source_bytes
        guard.checkpoint(declared_bytes + holdout_memory + len(holdout_bytes))
        cache = store.write_bytes(f"{access_relative}/cache.parquet", holdout_bytes)
        cache_sha256 = sha256_file(cache)
        if cache_sha256 != source_sha256:
            raise ValueError("durable holdout snapshot changed while being published")
        complete = {
            **expected_identity,
            "access_id": access_id,
            "transaction_id": access_id,
            "state": "COMPLETE",
            "cache_path": f"{access_relative}/cache.parquet",
            "cache_kind": "exact_source_snapshot",
            "cache_sha256": cache_sha256,
            "row_count": len(holdout),
            "schema_sha256": feature_schema_sha256(config),
            "sample_id_order_sha256": self._holdout_sample_order_sha256(holdout),
        }
        guard.checkpoint(declared_bytes + holdout_memory + len(holdout_bytes))
        complete_bytes = self._benchmark_json_bytes(complete)
        complete_sha256 = hashlib.sha256(complete_bytes).hexdigest()
        store.write_bytes(f"{access_relative}/complete.json", complete_bytes)
        guard.checkpoint(declared_bytes + holdout_memory + len(holdout_bytes))
        self._after_benchmark_holdout_cache_complete()
        audit = dict(complete)
        audit["source_materializations"] = 1
        audit["resume_source_reopened"] = False
        return (
            holdout,
            audit,
            {
                f"{access_relative}/intent.json": intent_sha256,
                f"{access_relative}/cache.parquet": source_sha256,
                f"{access_relative}/complete.json": complete_sha256,
            },
        )

    def _serialize_benchmark_payload(
        self,
        label: str,
        producer: Callable[[], bytes],
        *,
        guard: _ResourceGuard,
        live_bytes: int,
    ) -> tuple[bytes, int]:
        """Bound one large serialization and carry conservative live-byte accounting."""

        del self, label
        guard.checkpoint(live_bytes)
        payload = producer()
        updated_live_bytes = live_bytes + len(payload)
        guard.checkpoint(updated_live_bytes)
        return payload, updated_live_bytes

    @staticmethod
    def _benchmark_json_bytes(value: object) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _verify_optional_model_roundtrip(
        payload: bytes,
        originals: dict[tuple[str, int], photonic_benchmark.PhotonicSeedModel],
        guard: _ResourceGuard,
        live_bytes: int,
    ) -> None:
        """Validate the trusted producer's pickle before its bytes become an artifact."""
        guard.checkpoint(live_bytes + len(payload))
        try:
            restored = pickle.loads(payload)
            if not isinstance(restored, dict) or set(restored) != set(originals):
                raise ValueError("model inventory changed")
            for key, original in originals.items():
                model = restored[key]
                if not isinstance(model, photonic_benchmark.PhotonicSeedModel):
                    raise ValueError("model type changed")
                if (
                    not isinstance(model.scaler, StandardScaler)
                    or not isinstance(model.heads, baseline_models.QuantileHeads)
                ):
                    raise ValueError("model scaler or heads type changed")
                for field in (
                    "control", "circuit_seed", "modes", "photons", "phase_scale",
                    "alpha", "circuit_sha256",
                ):
                    if getattr(model, field) != getattr(original, field):
                        raise ValueError(f"model {field} changed")
                if model.heads.quantiles != original.heads.quantiles:
                    raise ValueError("model quantiles changed")
                for field in ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"):
                    if not np.array_equal(
                        getattr(model.scaler, field), getattr(original.scaler, field)
                    ):
                        raise ValueError(f"model scaler {field} changed")
                if len(model.heads.estimators) != len(original.heads.estimators):
                    raise ValueError("model heads changed")
                for head, source in zip(
                    model.heads.estimators, original.heads.estimators, strict=True
                ):
                    if pickle.dumps(head, protocol=5) != pickle.dumps(source, protocol=5):
                        raise ValueError("model head coefficients changed")
            if pickle.dumps(originals, protocol=5) != payload:
                raise ValueError("model persistence is nondeterministic")
        except (
            pickle.UnpicklingError, EOFError, TypeError, AttributeError, ValueError, KeyError
        ) as error:
            raise photonic_benchmark.PhotonicBenchmarkContractError(
                f"optional model serialization round-trip failed: {error}"
            ) from error
        guard.checkpoint(live_bytes + len(payload))

    @classmethod
    def _run_config(cls, store: RunStore) -> PilotConfig:
        payload = cls._object_from_captured(store.read_authenticated("config.json"), "config.json")
        return cls._config_from_snapshot(payload, store.config_hash)

    @staticmethod
    def _config_from_snapshot(payload: Mapping[str, Any], expected_config_hash: str) -> PilotConfig:
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
        if config.config_hash() != expected_config_hash:
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
            details = manifest.get("details")
            raw_allowlist = (
                details.get("prerequisite_output_allowlist") if isinstance(details, dict) else None
            )
            if not isinstance(raw_allowlist, dict) or not all(
                isinstance(key, str)
                and isinstance(paths, list)
                and all(isinstance(path, str) for path in paths)
                for key, paths in raw_allowlist.items()
            ):
                raise ValueError("benchmark selected-output audit is missing")
            store.verify_terminal_stage(
                "benchmark",
                accepted_statuses=frozenset({"SUCCESS"}),
                prerequisite_output_allowlist=cast(dict[str, list[str]], raw_allowlist),
            )
        else:
            store.verify_terminal_diagnostic_stage("benchmark")
        result = StageResult.model_validate_json(
            (store.root / "benchmark/stage_result.json").read_text(encoding="utf-8")
        )
        return result.model_copy(update={"message": f"benchmark already {status.lower()}"})

    @classmethod
    def _existing_report_result(cls, store: RunStore) -> StageResult | None:
        marker = store.root / "report/stage_manifest.json"
        if not marker.is_file():
            return None
        manifest = cls._read_object(marker)
        status = manifest.get("status")
        if status not in {"SUCCESS", "BLOCKED", "INVALID"}:
            raise ValueError("report has an invalid terminal marker")
        if status == "INVALID":
            store.verify_terminal_diagnostic_stage("report")
        else:
            details = manifest.get("details")
            raw_allowlist = (
                details.get("prerequisite_output_allowlist") if isinstance(details, dict) else None
            )
            if not isinstance(raw_allowlist, dict) or not all(
                isinstance(key, str)
                and isinstance(paths, list)
                and all(isinstance(path, str) for path in paths)
                for key, paths in raw_allowlist.items()
            ):
                raise ValueError("report selected-output audit is missing")
            store.verify_terminal_stage(
                "report",
                accepted_statuses=frozenset({str(status)}),
                prerequisite_output_allowlist=cast(dict[str, list[str]], raw_allowlist),
            )
        result = StageResult.model_validate_json(
            (store.root / "report/stage_result.json").read_text(encoding="utf-8")
        )
        return result.model_copy(update={"message": f"report already {status.lower()}"})

    @staticmethod
    def _report_benchmark_allowlist() -> dict[str, tuple[str, ...]]:
        return {
            "benchmark": (
                "benchmark/selection_manifest.json",
                "benchmark/baseline_predictions.parquet",
                "benchmark/exact_predictions.parquet",
                "benchmark/finite_shot_predictions.parquet",
                "benchmark/diagnostics.json",
                "benchmark/models/manifest.json",
                "benchmark/resource_ledger.json",
                "benchmark/stage_result.json",
            )
        }

    @classmethod
    def _read_declared_stage_artifact(cls, store: RunStore, stage: str, relative_path: str) -> Path:
        marker = cls._read_object(store.root / stage / "stage_manifest.json")
        if (
            marker.get("schema_version") != "1.0"
            or marker.get("stage") != stage
            or marker.get("status") != "SUCCESS"
            or marker.get("config_hash") != store.config_hash
        ):
            raise ValueError(f"{stage} has an invalid SUCCESS marker")
        outputs = marker.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"{stage} has an invalid output inventory")
        matches = [
            record
            for record in outputs
            if isinstance(record, dict) and record.get("path") == relative_path
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
            raise ValueError(f"{stage} output inventory is missing {relative_path}")
        raw_target = store.root / relative_path
        target = raw_target.resolve(strict=False)
        if (
            not target.is_relative_to(store.root)
            or raw_target.is_symlink()
            or not target.is_file()
            or sha256_file(target) != matches[0]["sha256"]
        ):
            raise ValueError(f"stage output checksum mismatch: {relative_path}")
        return target

    @classmethod
    def _declared_stage_artifact_sha256(
        cls, store: RunStore, stage: str, relative_path: str
    ) -> str:
        marker = cls._read_object(store.root / stage / "stage_manifest.json")
        if (
            marker.get("schema_version") != "1.0"
            or marker.get("stage") != stage
            or marker.get("status") != "SUCCESS"
            or marker.get("config_hash") != store.config_hash
        ):
            raise ValueError(f"{stage} has an invalid SUCCESS marker")
        outputs = marker.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"{stage} has an invalid output inventory")
        matches = [
            record.get("sha256")
            for record in outputs
            if isinstance(record, dict) and record.get("path") == relative_path
        ]
        if len(matches) != 1 or not isinstance(matches[0], str):
            raise ValueError(f"{stage} output inventory is missing {relative_path}")
        return matches[0]

    @staticmethod
    def _capture_report_artifact(
        store: RunStore,
        relative_path: str,
        expected_sha256: str | None,
        guard: _ResourceGuard,
        live_bytes: int,
    ) -> tuple[bytes, int]:
        """Capture a trusted regular file once and bind parsing to those bytes."""

        payload = authenticated_snapshot(
            store.root / relative_path,
            expected_sha256=expected_sha256,
            checkpoint=guard.checkpoint,
            live_bytes=live_bytes,
            root=store.root,
        )
        return payload, live_bytes + len(payload)

    @staticmethod
    def _object_from_captured(payload: bytes, label: str) -> dict[str, Any]:
        try:
            value: object = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"captured JSON artifact is invalid: {label}") from error
        if not isinstance(value, dict):
            raise ValueError(f"captured JSON artifact must be an object: {label}")
        return value

    @classmethod
    def _capture_report_provenance(
        cls,
        store: RunStore,
        guard: _ResourceGuard,
    ) -> tuple[
        PilotConfig,
        dict[str, bytes],
        dict[str, str],
        dict[str, dict[str, Any]],
        frozenset[str],
        int,
    ]:
        """Capture base and stage provenance once, then validate the frozen lineage."""

        captured: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        live_bytes = 0

        def capture(relative: str, expected: str | None = None) -> dict[str, Any]:
            nonlocal live_bytes
            payload, live_bytes = cls._capture_report_artifact(
                store, relative, expected, guard, live_bytes
            )
            captured[relative] = payload
            digests[relative] = hashlib.sha256(payload).hexdigest()
            return cls._object_from_captured(payload, relative)

        run_manifest = capture("run_manifest.json")
        if (
            run_manifest.get("schema_version") != "1.0"
            or run_manifest.get("stage") != "init"
            or run_manifest.get("status") != "SUCCESS"
            or run_manifest.get("config_hash") != store.config_hash
            or run_manifest.get("input_hashes") != {}
            or run_manifest.get("prerequisite_manifest_hashes") != {}
        ):
            raise ValueError("captured run provenance manifest is invalid")
        raw_base_outputs = run_manifest.get("outputs")
        if not isinstance(raw_base_outputs, list):
            raise ValueError("captured run provenance outputs are invalid")
        base_outputs: dict[str, str] = {}
        for record in raw_base_outputs:
            if not isinstance(record, dict):
                raise ValueError("captured run provenance output is invalid")
            path = record.get("path")
            digest = record.get("sha256")
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or path in base_outputs
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("captured run provenance output is invalid")
            base_outputs[path] = digest
        expected_base_outputs = {
            "config.json",
            "seed_manifest.json",
            "environment_manifest.json",
            "source_manifest.json",
        }
        if set(base_outputs) != expected_base_outputs:
            raise ValueError("captured run provenance inventory is invalid")
        if base_outputs["config.json"] != store.config_hash:
            raise ValueError("captured config checksum is not anchored to the run")

        config_payload = capture("config.json", base_outputs["config.json"])
        config = cls._config_from_snapshot(config_payload, store.config_hash)
        for relative in ("environment_manifest.json", "source_manifest.json"):
            payload = capture(relative, base_outputs[relative])
            if (
                payload.get("schema_version") != "1.0"
                or payload.get("config_hash") != store.config_hash
            ):
                raise ValueError(f"captured base provenance is invalid: {relative}")

        stages = (
            "validate-data",
            "build-splits",
            "train-baselines",
            "build-photonic-features",
            "benchmark",
        )
        stage_manifests: dict[str, dict[str, Any]] = {}
        stage_hashes: dict[str, str] = {}
        for stage in stages:
            relative = f"{stage}/stage_manifest.json"
            stage_manifests[stage] = capture(relative)
            stage_hashes[stage] = digests[relative]
        inventory = store.report_input_inventory_from_captured_manifests(
            stage_manifests, stage_hashes, captured_config=config
        )
        return config, captured, digests, stage_manifests, inventory, live_bytes

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
        empty_baseline = baseline_models.prediction_parquet_bytes(
            pd.DataFrame(columns=baseline_models.prediction_arrow_schema().names)
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
                Path("benchmark/baseline_predictions.parquet"),
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
            transaction.write_bytes("baseline_predictions.parquet", empty_baseline)
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
        empty_scores = pd.DataFrame(columns=baseline_models.validation_score_arrow_schema().names)
        validation_bytes = baseline_models.validation_scores_parquet_bytes(empty_scores)
        model_bytes = pickle.dumps({}, protocol=5)
        result = StageResult(
            stage="train-baselines",
            status=StageStatus.INVALID,
            message=message,
            artifacts=(
                Path("train-baselines/selection_manifest.json"),
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
    def _human_cleaning_ledger(config: PilotConfig) -> dict[str, object]:
        """Record declared engineer-hours without deriving them from process runtime."""

        cleaning = config.data.cleaning
        hours = None if cleaning is None else cleaning.human_cleaning_hours
        return {
            "human_cleaning_hours": hours,
            "human_cleaning_hours_source": (
                "NOT_RECORDED" if cleaning is None else "CONFIG_DECLARED"
            ),
            "human_cleaning_timebox_hours": HUMAN_CLEANING_TIMEBOX_HOURS,
            "human_cleaning_timebox_status": (
                "NOT_RECORDED"
                if hours is None
                else "TIMEBOX_REACHED"
                if hours >= HUMAN_CLEANING_TIMEBOX_HOURS
                else "WITHIN_TIMEBOX"
            ),
            "qc_policy_locked": None if cleaning is None else cleaning.qc_policy_locked,
        }

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
            **DefaultPilotServices._human_cleaning_ledger(config),
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
        source_manifest = self._object_from_captured(
            store.read_authenticated("source_manifest.json"), "source_manifest.json"
        )
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
                source_payload = authenticated_snapshot(
                    source_path, expected_sha256=expected_hash, checkpoint=guard.checkpoint
                )
                result = canonicalize_data(
                    source_payload,
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
                    **self._human_cleaning_ledger(config),
                    "peak_declared_bytes_observed": guard.peak_declared_bytes,
                    "peak_rss_bytes_observed": guard.peak_rss_bytes,
                    "resource_status": resource_status,
                    "schema_version": "1.0",
                    "stage": "validate-data",
                    "status": status.value,
                    "timebox_seconds": config.resources.max_wall_clock_seconds,
                    "wall_clock_seconds": elapsed,
                    "peak_rss_bytes": guard.peak_rss_bytes,
                    "peak_declared_bytes": guard.peak_declared_bytes,
                    "max_wall_clock_seconds": config.resources.max_wall_clock_seconds,
                    "max_memory_bytes": guard.memory_bytes,
                    "cpu_only": True,
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
            "stage": "build-splits",
            "scientific_decision": "NOT_EVALUATED",
            "status": status.value,
            "timebox_seconds": guard.wall_seconds,
            "wall_clock_seconds": time.perf_counter() - guard.started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
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
            "contiguity_policy": sample_contiguity_policy(),
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
        config = self._run_config(store)
        canonical_path = store.root / "validate-data/canonical.parquet"
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        try:
            validate_marker_sha256 = store.verify_terminal_stage("validate-data")
            marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="validate-data/stage_manifest.json",
                expected_sha256=validate_marker_sha256,
                guard=guard,
                live_bytes=0,
            )
            marker = self._object_from_captured(marker_payload, "validate-data/stage_manifest.json")
            raw_outputs = marker.get("outputs")
            if not isinstance(raw_outputs, list):
                raise ValueError("validate-data marker has invalid outputs")
            canonical_relative = "validate-data/canonical.parquet"
            matches = [
                record.get("sha256")
                for record in raw_outputs
                if isinstance(record, dict) and record.get("path") == canonical_relative
            ]
            if len(matches) != 1 or not isinstance(matches[0], str):
                raise ValueError("validate-data marker does not bind canonical data")
            canonical_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path=canonical_relative,
                expected_sha256=matches[0],
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            canonical = pd.read_parquet(io.BytesIO(canonical_payload))
            declared = int(canonical.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(captured_live_bytes + declared)
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
                preverified_input_hashes={canonical_relative: matches[0]},
                preverified_prerequisite_manifest_hashes={"validate-data": validate_marker_sha256},
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
        config = self._run_config(store)
        development_relative = "build-splits/development_features.parquet"
        manifest_relative = "build-splits/feature_manifest.json"
        scalers_relative = "build-splits/minmax_scalers.json"
        required_inputs = (development_relative, manifest_relative, scalers_relative)
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        split_marker_hash = store._verify_stage_selected_outputs("build-splits", required_inputs)
        try:
            marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="build-splits/stage_manifest.json",
                expected_sha256=split_marker_hash,
                guard=guard,
                live_bytes=0,
            )
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        split_stage_manifest = self._object_from_captured(
            marker_payload, "build-splits/stage_manifest.json"
        )
        if (
            split_stage_manifest.get("schema_version") != "1.0"
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
        for required in required_inputs:
            if required not in expected_output_hashes:
                raise ValueError(f"build-splits marker is missing partition output: {required}")
        try:
            captured_inputs: dict[str, bytes] = {}
            for relative_path in required_inputs:
                payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative_path,
                    expected_sha256=expected_output_hashes[relative_path],
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                captured_inputs[relative_path] = payload
            development = pd.read_parquet(io.BytesIO(captured_inputs[development_relative]))
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(captured_live_bytes + declared)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        feature_manifest = self._object_from_captured(
            captured_inputs[manifest_relative], manifest_relative
        )
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
        if not isinstance(development_contract, dict):
            raise baseline_models.BaselineContractError(
                "feature manifest development partition record is invalid"
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
                resource_checkpoint=guard.checkpoint,
                fit_executor=fit_executor,
            )
            guard.checkpoint(declared)
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
            "selection_freeze_sha256": training.selection_manifest["selection_freeze_sha256"],
        }
        total_bytes = declared + len(validation_bytes) + len(model_bytes)
        try:
            guard.checkpoint(total_bytes)
        except _ResourceLimitExceeded as error:
            return self._publish_baseline_resource_invalid(
                store=store, config=config, guard=guard, gate=error.gate
            )
        artifacts = (
            Path("train-baselines/selection_manifest.json"),
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
                DetailEntry(key="test_prediction_passes", value=0),
                DetailEntry(key="holdout_opened", value=False),
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
                    Path(development_relative),
                    Path(manifest_relative),
                    Path(scalers_relative),
                ),
                preverified_input_hashes={
                    relative: expected_output_hashes[relative] for relative in required_inputs
                },
                preverified_prerequisite_manifest_hashes={"build-splits": split_marker_hash},
                details={
                    "result_status": StageStatus.SUCCESS.value,
                    "scientific_decision": "NOT_EVALUATED",
                },
                commit_guard=guard.checkpoint,
            ) as transaction:
                transaction.write_json("selection_manifest.json", training.selection_manifest)
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
            marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="build-splits/stage_manifest.json",
                expected_sha256=split_marker_hash,
                guard=guard,
                live_bytes=0,
            )
            split_stage_marker = self._object_from_captured(
                marker_payload, "build-splits/stage_manifest.json"
            )
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
            captured_inputs: dict[str, bytes] = {}
            for relative_path in required_inputs:
                payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative_path,
                    expected_sha256=verified_input_hashes[relative_path],
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                captured_inputs[relative_path] = payload
            development = pd.read_parquet(io.BytesIO(captured_inputs[required_inputs[0]]))
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(captured_live_bytes + declared)
            split_manifest = self._object_from_captured(
                captured_inputs[required_inputs[1]], required_inputs[1]
            )
            scalers = self._object_from_captured(
                captured_inputs[required_inputs[2]], required_inputs[2]
            )
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
        optional_profile: optional_photonics.ProfileBundle | None = None
        if config.photonics.enable_three_photon_profile:
            profile_started = time.perf_counter()
            optional_profile = optional_photonics.build_profile(
                development, config, input_hashes=input_hashes
            )
            # The optional measured budget is separate from the exact primary resource gate.
            guard.started += time.perf_counter() - profile_started
        try:
            with store.stage(
                "build-photonic-features",
                input_paths=required_inputs,
                prerequisite_verification="selected_outputs",
                prerequisite_output_allowlist=self._photonic_allowlist(),
                preverified_input_hashes=input_hashes,
                preverified_prerequisite_manifest_hashes={"build-splits": split_marker_hash},
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
                if optional_profile is not None:
                    transaction.write_bytes(
                        "three-photon/development_features.parquet", optional_profile.feature_bytes
                    )
                    transaction.write_json(
                        "three-photon/state_spaces.json", optional_profile.state_spaces
                    )
                    transaction.write_json(
                        "three-photon/circuit_manifest.json", optional_profile.circuit_manifest
                    )
                    transaction.write_json("three-photon/resource_gate.json", optional_profile.gate)
                    for path, payload in sorted(optional_profile.circuit_payloads.items()):
                        transaction.write_bytes(path, payload)
                guard.checkpoint(declared)
                ledger.update(
                    {
                        "wall_clock_seconds": time.perf_counter() - guard.started,
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

    def _capture_three_photon_selection(
        self,
        *,
        store: RunStore,
        config: PilotConfig,
        task7_hashes: dict[str, str],
        task7_inputs: dict[str, str],
        development_reference: pd.DataFrame,
        guard: _ResourceGuard,
        fit_executor: baseline_models.FitExecutor,
        captured_input_hashes: dict[str, str],
    ) -> tuple[
        dict[str, Any],
        photonic_benchmark.PhotonicSelectionResult | None,
        dict[tuple[int, int], photonic_features.CircuitDefinition],
        dict[tuple[int, int], tuple[int, str]],
        tuple[str, ...],
    ]:
        if not config.photonics.enable_three_photon_profile:
            return {"profile_status": "DISABLED", "benchmark_status": "DISABLED"}, None, {}, {}, ()
        payloads: dict[str, bytes] = {}
        live = 0
        for path in optional_photonics.PROFILE_FIXED_PATHS:
            expected = task7_hashes.get(path)
            if expected is None:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "enabled three-photon profile artifact is missing"
                )
            payload, live = self._capture_report_artifact(store, path, expected, guard, live)
            payloads[path] = payload
            captured_input_hashes[path] = expected
        gate_path, feature_path, state_path, circuit_path = optional_photonics.PROFILE_FIXED_PATHS
        def optional_object(path: str) -> dict[str, Any]:
            try:
                return self._object_from_captured(payloads[path], path)
            except ValueError as error:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    f"optional JSON artifact is invalid: {path}"
                ) from error

        gate = optional_object(gate_path)
        summary = {
            "label": "diagnostic_3photon",
            "profile_status": gate.get("status"),
            "benchmark_status": "DISABLED",
            "used_for_go_stop": False,
            "profile_gate_sha256": task7_hashes[gate_path],
            "resource_gate": gate,
        }
        circuits: dict[tuple[int, int], photonic_features.CircuitDefinition] = {}
        circuit_manifest = optional_object(circuit_path)
        state_manifest = optional_object(state_path)
        if gate.get("status") == "FAIL":
            if (
                config.photonics.enable_three_photon_benchmark
                or gate.get("schema_version") != "1.0"
                or gate.get("photons") != 3
                or gate.get("config_sha256") != config.config_hash()
                or gate.get("input_hashes") != task7_inputs
                or gate.get("holdout_opened") is not False
                or gate.get("partition") != "development_only"
                or gate.get("failure_gate") not in {
                    "wall_clock", "memory_rss", "dataframe_or_output_bytes",
                    "preflight_projected_wall_clock",
                }
                or gate.get("row_count") != 0
                or "feature_sha256" in gate
                or circuit_manifest != {"schema_version": "1.0", "circuits": []}
                or state_manifest != {"schema_version": "1.0", "spaces": []}
                or payloads[feature_path] != photonic_features.feature_parquet_bytes(pd.DataFrame())
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "failed optional profile evidence is invalid"
                )
            return summary, None, {}, {}, tuple(optional_photonics.PROFILE_FIXED_PATHS)
        records = circuit_manifest.get("circuits")
        if not isinstance(records, list):
            raise photonic_benchmark.PhotonicBenchmarkContractError("optional circuit list invalid")
        paths = list(optional_photonics.PROFILE_FIXED_PATHS)
        expected_circuit_keys = {
            (modes, seed) for modes in config.photonics.modes
            for seed in config.seeds.circuit_seeds
        }

        def valid_digest(value: object) -> bool:
            return (
                isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        for record in records:
            if not isinstance(record, dict):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional circuit record invalid"
                )
            optional_path = record.get("path")
            modes, photons, seed = (
                record.get(name) for name in ("modes", "photons", "circuit_seed")
            )
            if (
                type(modes) is not int or type(photons) is not int or type(seed) is not int
                or (modes, seed) not in expected_circuit_keys
                or photons != 3
                or not isinstance(optional_path, str)
                or optional_path != (
                    f"{optional_photonics.PROFILE_PREFIX}/circuits/m{modes}-n3-seed{seed}.npz"
                )
                or optional_path not in task7_hashes
                or optional_path in paths
                or not valid_digest(record.get("npz_sha256"))
                or not valid_digest(record.get("circuit_sha256"))
                or not valid_digest(record.get("state_order_sha256"))
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional circuit record invalid"
                )
            payload, live = self._capture_report_artifact(
                store, optional_path, task7_hashes[optional_path], guard, live
            )
            captured_input_hashes[optional_path] = task7_hashes[optional_path]
            paths.append(optional_path)
            try:
                circuit = photonic_features.load_circuit_npz(payload)
            except photonic_features.PhotonicContractError as error:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional circuit archive invalid"
                ) from error
            if (
                circuit.photons != 3
                or hashlib.sha256(payload).hexdigest() != record["npz_sha256"]
                or photonic_features.circuit_checksum(circuit) != record["circuit_sha256"]
                or circuit.state_order_sha256 != record["state_order_sha256"]
                or circuit.modes != modes
                or circuit.circuit_seed != seed
                or (circuit.modes, circuit.circuit_seed) in circuits
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError("optional circuit invalid")
            circuits[(circuit.modes, circuit.circuit_seed)] = circuit
        if set(circuits) != expected_circuit_keys:
            raise photonic_benchmark.PhotonicBenchmarkContractError(
                "optional circuit grid incomplete"
            )
        spaces = state_manifest.get("spaces")
        if not isinstance(spaces, list):
            raise photonic_benchmark.PhotonicBenchmarkContractError("optional state list invalid")
        states: dict[tuple[int, int], tuple[int, str]] = {}
        for record in spaces:
            if not isinstance(record, dict):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional state record invalid"
                )
            modes, photons, dimension = (
                record.get(name) for name in ("modes", "photons", "dimension")
            )
            if (
                type(modes) is not int or type(photons) is not int
                or type(dimension) is not int or modes not in config.photonics.modes
                or photons != 3 or (modes, photons) in states
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional state record invalid"
                )
            canonical_states = photonic_features.enumerate_fock_states(modes, 3)
            checksum = photonic_features.state_order_checksum(canonical_states)
            if (
                dimension != len(canonical_states)
                or record.get("states") != [list(state) for state in canonical_states]
                or record.get("state_order_sha256") != checksum
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "optional state order invalid"
                )
            states[(modes, photons)] = (dimension, checksum)
        if set(states) != {(modes, 3) for modes in config.photonics.modes}:
            raise photonic_benchmark.PhotonicBenchmarkContractError(
                "optional state grid incomplete"
            )
        selection = None
        if config.photonics.enable_three_photon_benchmark:
            try:
                optional_photonics.require_profile_pass(
                    gate, config, payloads[feature_path], task7_inputs
                )
            except photonic_features.PhotonicContractError as error:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "three-photon benchmark requires authenticated G2 profile PASS"
                ) from error
            selection = photonic_benchmark.select_photonic_models(
                pd.read_parquet(io.BytesIO(payloads[feature_path])),
                config,
                diagnostic_photons=3,
                authenticated_profile_gate_sha256=task7_hashes[gate_path],
                development_reference=development_reference,
                trusted_circuits=circuits,
                trusted_state_spaces=states,
                fit_executor=fit_executor,
                resource_checkpoint=guard.checkpoint,
            )
            summary.update(
                benchmark_status="SELECTED",
                selection_freeze_sha256=selection.selection_manifest["selection_freeze_sha256"],
            )
        return summary, selection, circuits, states, tuple(paths)

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
            captured_input_hashes: dict[str, str] = {}
            captured_live_bytes = 0
            task7_marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="build-photonic-features/stage_manifest.json",
                expected_sha256=photonic_marker_hash,
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            task7_manifest = self._object_from_captured(
                task7_marker_payload, "build-photonic-features/stage_manifest.json"
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
            task7_payloads: dict[str, bytes] = {}
            for relative in (
                development_relative,
                circuit_manifest_relative,
                feature_manifest_relative,
                state_spaces_relative,
            ):
                payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative,
                    expected_sha256=task7_hashes[relative],
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                task7_payloads[relative] = payload
                captured_input_hashes[relative] = task7_hashes[relative]
            development = pd.read_parquet(io.BytesIO(task7_payloads[development_relative]))
            declared = int(development.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(captured_live_bytes + declared)
            task7_inputs = task7_manifest.get("input_hashes")
            development_reference_sha256 = (
                task7_inputs.get(development_reference_relative)
                if isinstance(task7_inputs, dict)
                else None
            )
            if not isinstance(development_reference_sha256, str):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "Task 5 development reference is not bound to Task 7"
                )
            development_reference_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path=development_reference_relative,
                expected_sha256=development_reference_sha256,
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            captured_input_hashes[development_reference_relative] = development_reference_sha256
            development_reference = pd.read_parquet(io.BytesIO(development_reference_payload))
            declared += int(development_reference.memory_usage(index=True, deep=True).sum())
            guard.checkpoint(captured_live_bytes + declared)
            task7_feature_manifest = self._object_from_captured(
                task7_payloads[feature_manifest_relative], feature_manifest_relative
            )
            if (
                task7_feature_manifest.get("holdout_opened") is not False
                or task7_feature_manifest.get("photons") != 2
                or task7_feature_manifest.get("circuit_seeds") != list(config.seeds.circuit_seeds)
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "development photonic manifest violates the G3 firewall"
                )
            raw_state_spaces = self._object_from_captured(
                task7_payloads[state_spaces_relative], state_spaces_relative
            ).get("spaces")
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
            circuit_manifest = self._object_from_captured(
                task7_payloads[circuit_manifest_relative], circuit_manifest_relative
            )
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
                payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative,
                    expected_sha256=task7_hashes[relative],
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                captured_input_hashes[relative] = task7_hashes[relative]
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
            (
                optional_summary,
                optional_selection,
                optional_circuits,
                optional_states,
                optional_paths,
            ) = self._capture_three_photon_selection(
                store=store,
                config=config,
                task7_hashes=task7_hashes,
                task7_inputs=cast(dict[str, str], task7_inputs),
                development_reference=development_reference,
                guard=guard,
                fit_executor=fit_executor,
                captured_input_hashes=captured_input_hashes,
            )
            access_freeze_hash = freeze_hash
            if optional_selection is not None:
                access_freeze_hash = hashlib.sha256(
                    self._benchmark_json_bytes(
                        {
                            "primary_2photon": freeze_hash,
                            "diagnostic_3photon": optional_selection.selection_manifest[
                                "selection_freeze_sha256"
                            ],
                        }
                    )
                ).hexdigest()

            # All holdout-bound verification is deliberately below the immutable freeze.
            guard.checkpoint(declared)
            baseline_selection_relative = "train-baselines/selection_manifest.json"
            baseline_models_relative = "train-baselines/models/baseline_models.pkl"
            baseline_model_manifest_relative = "train-baselines/models/manifest.json"
            baseline_inputs = (
                baseline_selection_relative,
                baseline_models_relative,
                baseline_model_manifest_relative,
            )
            baseline_marker_hash = store._verify_stage_selected_outputs(
                "train-baselines", baseline_inputs
            )
            baseline_marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="train-baselines/stage_manifest.json",
                expected_sha256=baseline_marker_hash,
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            baseline_marker = self._object_from_captured(
                baseline_marker_payload, "train-baselines/stage_manifest.json"
            )
            raw_baseline_outputs = baseline_marker.get("outputs")
            if not isinstance(raw_baseline_outputs, list):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "baseline stage output manifest is invalid"
                )
            baseline_hashes = {
                str(record.get("path")): str(record.get("sha256"))
                for record in raw_baseline_outputs
                if isinstance(record, dict)
            }
            baseline_payloads: dict[str, bytes] = {}
            for relative in baseline_inputs:
                if relative not in baseline_hashes:
                    raise photonic_benchmark.PhotonicBenchmarkContractError(
                        f"baseline stage did not declare {relative}"
                    )
                payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative,
                    expected_sha256=baseline_hashes[relative],
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                baseline_payloads[relative] = payload
                captured_input_hashes[relative] = baseline_hashes[relative]
            baseline_selection = self._object_from_captured(
                baseline_payloads[baseline_selection_relative], baseline_selection_relative
            )
            baseline_model_bytes = baseline_payloads[baseline_models_relative]
            baseline_model_manifest = self._object_from_captured(
                baseline_payloads[baseline_model_manifest_relative],
                baseline_model_manifest_relative,
            )
            if baseline_model_manifest.get("models_sha256") != hashlib.sha256(
                baseline_model_bytes
            ).hexdigest() or baseline_model_manifest.get(
                "selection_freeze_sha256"
            ) != baseline_selection.get("selection_freeze_sha256"):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "baseline model lineage is inconsistent"
                )
            baseline_model_bundles = pickle.loads(baseline_model_bytes)
            baseline_selected = baseline_selection.get("selected")
            if not isinstance(baseline_model_bundles, dict) or not isinstance(
                baseline_selected, dict
            ):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "baseline model bundle is invalid"
                )

            raw_prerequisite_hashes = task7_manifest.get("prerequisite_manifest_hashes")
            split_marker_hash = (
                raw_prerequisite_hashes.get("build-splits")
                if isinstance(raw_prerequisite_hashes, dict)
                else None
            )
            if not isinstance(split_marker_hash, str) or len(split_marker_hash) != 64:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "Task 7 does not bind a valid split marker"
                )
            split_marker_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path="build-splits/stage_manifest.json",
                expected_sha256=split_marker_hash,
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            split_stage_manifest = self._object_from_captured(
                split_marker_payload, "build-splits/stage_manifest.json"
            )
            captured_input_hashes["build-splits/stage_manifest.json"] = split_marker_hash
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
            holdout_sha256 = split_hashes.get(holdout_relative)
            if not isinstance(holdout_sha256, str) or len(holdout_sha256) != 64:
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "split marker does not bind the physical holdout"
                )
            split_feature_relative = "build-splits/feature_manifest.json"
            split_feature_sha256 = split_hashes.get(split_feature_relative)
            if not isinstance(split_feature_sha256, str):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "split marker does not bind its feature manifest"
                )
            split_feature_payload, captured_live_bytes = self._capture_report_artifact(
                store=store,
                relative_path=split_feature_relative,
                expected_sha256=split_feature_sha256,
                guard=guard,
                live_bytes=captured_live_bytes,
            )
            split_manifest = self._object_from_captured(
                split_feature_payload, split_feature_relative
            )
            captured_input_hashes[split_feature_relative] = split_feature_sha256
            partitions = split_manifest.get("partitions")
            holdout_contract = partitions.get("holdout") if isinstance(partitions, dict) else None
            if not isinstance(holdout_contract, dict):
                raise photonic_benchmark.PhotonicBenchmarkContractError(
                    "holdout partition contract is missing"
                )
            holdout, holdout_access_audit, holdout_access_hashes = (
                self._load_or_materialize_benchmark_holdout(
                    store=store,
                    config=config,
                    source_relative=holdout_relative,
                    source_sha256=holdout_sha256,
                    split_marker_sha256=split_marker_hash,
                    selection_freeze_sha256=access_freeze_hash,
                    guard=guard,
                    declared_bytes=declared,
                )
            )
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

            baseline_predictions = baseline_models.predict_test_models(
                holdout,
                baseline_model_bundles,
                tuple(config.evaluation.quantiles),
                baseline_selected,
                executor=fit_executor,
            )
            baseline_prediction_bytes = baseline_models.prediction_parquet_bytes(
                baseline_predictions
            )
            baseline_predictions_sha256 = hashlib.sha256(baseline_prediction_bytes).hexdigest()

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
            optional_payloads: dict[str, bytes] = {}
            if optional_selection is not None:
                optional_modes = int(optional_selection.selected["modes"])
                optional_exact = photonic_benchmark.build_holdout_photonic_features(
                    holdout,
                    selected=optional_selection.selected,
                    circuits={
                        seed: optional_circuits[(optional_modes, seed)]
                        for seed in config.seeds.circuit_seeds
                    },
                    trusted_state_space=optional_states[(optional_modes, 3)],
                    resource_checkpoint=guard.checkpoint,
                )
                diagnostic_predictions, diagnostic_details = photonic_benchmark.predict_exact(
                    optional_exact,
                    optional_selection.selected_models,
                    tuple(config.evaluation.quantiles),
                    fit_executor=fit_executor,
                    resource_checkpoint=guard.checkpoint,
                )
                if len(diagnostic_predictions) != expected_exact_feature_rows:
                    raise RuntimeError("optional exact predictions are incomplete")
                optional_summary.update(
                    benchmark_status="EVALUATED",
                    used_for_go_stop=False,
                    joint_selection_freeze_sha256=access_freeze_hash,
                    prediction_rows=len(diagnostic_predictions),
                )
                diagnostic_scores = baseline_models.normalized_wis(
                    diagnostic_predictions["target_ac_power_kw"].to_numpy(),
                    diagnostic_predictions.loc[
                        :,
                        list(baseline_models.quantile_columns(tuple(config.evaluation.quantiles))),
                    ].to_numpy(),
                    diagnostic_predictions["capacity_kw"].to_numpy(),
                    tuple(config.evaluation.quantiles),
                )
                metric_frame = diagnostic_predictions.loc[:, ["control", "circuit_seed"]].copy()
                metric_frame["normalized_wis"] = diagnostic_scores
                metrics = {
                    "schema_version": "1.0",
                    "label": "diagnostic_3photon",
                    "used_for_go_stop": False,
                    "scientific_decision": "NOT_EVALUATED",
                    "synthetic": config.protocol.provenance is DataProvenance.SYNTHETIC,
                    "seed_policy": "equal_weight_mean_no_best_seed",
                    "seed_scores": metric_frame.groupby(
                        ["control", "circuit_seed"], as_index=False
                    )["normalized_wis"]
                    .agg(normalized_wis="mean", row_count="size")
                    .to_dict(orient="records"),
                    "exact_holdout": diagnostic_details,
                    "selection_freeze_sha256": optional_summary["selection_freeze_sha256"],
                    "joint_selection_freeze_sha256": access_freeze_hash,
                }
                optional_feature_frame = optional_exact.frame
                optional_frame_bytes = {
                    "validation_scores": int(
                        optional_selection.validation_scores.memory_usage(
                            index=True, deep=True
                        ).sum()
                    ),
                    "holdout_features": int(
                        optional_feature_frame.memory_usage(index=True, deep=True).sum()
                    ),
                    "exact_predictions": int(
                        diagnostic_predictions.memory_usage(index=True, deep=True).sum()
                    ),
                }
                optional_live_bytes = (
                    declared + sum(optional_frame_bytes.values()) + len(baseline_prediction_bytes)
                )
                guard.checkpoint(optional_live_bytes)

                def optional_serialize(name: str, producer: Callable[[], bytes]) -> bytes:
                    nonlocal optional_live_bytes
                    guard.checkpoint(optional_live_bytes + 2 * optional_frame_bytes.get(name, 0))
                    payload, optional_live_bytes = self._serialize_benchmark_payload(
                        f"diagnostic_3photon/{name}", producer,
                        guard=guard, live_bytes=optional_live_bytes,
                    )
                    return payload

                optional_payloads["selection_manifest.json"] = optional_serialize(
                    "selection_manifest",
                    lambda: self._benchmark_json_bytes(optional_selection.selection_manifest),
                )
                optional_payloads["validation_scores.parquet"] = optional_serialize(
                    "validation_scores",
                    lambda: photonic_benchmark.parquet_bytes(
                        optional_selection.validation_scores,
                        photonic_benchmark.validation_score_arrow_schema(),
                    ),
                )
                optional_payloads["holdout_features.parquet"] = optional_serialize(
                    "holdout_features",
                    lambda: photonic_features.feature_parquet_bytes(optional_feature_frame),
                )
                optional_payloads["exact_predictions.parquet"] = optional_serialize(
                    "exact_predictions",
                    lambda: photonic_benchmark.parquet_bytes(
                        diagnostic_predictions, photonic_benchmark.exact_prediction_arrow_schema()
                    ),
                )
                optional_model_bytes = optional_serialize(
                    "models",
                    lambda: photonic_benchmark.model_pickle_bytes(
                        optional_selection.selected_models
                    ),
                )
                self._verify_optional_model_roundtrip(
                    optional_model_bytes, optional_selection.selected_models,
                    guard, optional_live_bytes,
                )
                optional_payloads["models.pkl"] = optional_model_bytes
                metrics["model_inventory"] = {
                    "models_sha256": hashlib.sha256(optional_model_bytes).hexdigest(),
                    "model_keys": [
                        f"{control}:seed={seed}"
                        for control, seed in sorted(optional_selection.selected_models)
                    ],
                    "models": [
                        {
                            "control": control,
                            "circuit_seed": seed,
                            "modes": model.modes,
                            "photons": model.photons,
                            "phase_scale": model.phase_scale,
                            "alpha": model.alpha,
                            "circuit_sha256": model.circuit_sha256,
                            "quantiles": list(model.heads.quantiles),
                        }
                        for (control, seed), model in sorted(
                            optional_selection.selected_models.items()
                        )
                    ],
                    "serialization_roundtrip_verified": True,
                    "holdout_reinference_for_serialization": False,
                }
                optional_payloads["metrics.json"] = optional_serialize(
                    "metrics", lambda: self._benchmark_json_bytes(metrics)
                )
                declared += sum(optional_frame_bytes.values()) + sum(
                    map(len, optional_payloads.values())
                )
                guard.checkpoint(declared)
            finite_counts = photonic_benchmark.finite_shot_screen(
                exact_features,
                tuple(config.photonics.raw_attempt_budgets),
                SeedManager(config.seeds.master_seed),
                resource_checkpoint=guard.checkpoint,
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

            dataframe_live_bytes = sum(
                int(frame.memory_usage(index=True, deep=True).sum())
                for frame in (
                    selection.validation_scores,
                    baseline_predictions,
                    exact_predictions,
                    finite_counts,
                    finite_predictions,
                )
            )
            serialization_live_bytes = (
                declared + dataframe_live_bytes + len(baseline_prediction_bytes)
            )
            guard.checkpoint(serialization_live_bytes)
            validation_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "validation_scores",
                lambda: photonic_benchmark.parquet_bytes(
                    selection.validation_scores,
                    photonic_benchmark.validation_score_arrow_schema(),
                ),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            feature_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "holdout_features",
                lambda: photonic_features.feature_parquet_bytes(exact_features),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            exact_prediction_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "exact_predictions",
                lambda: photonic_benchmark.parquet_bytes(
                    exact_predictions,
                    photonic_benchmark.exact_prediction_arrow_schema(),
                ),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            finite_count_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "finite_shot_counts",
                lambda: photonic_benchmark.parquet_bytes(
                    finite_counts, photonic_benchmark.finite_shot_count_arrow_schema()
                ),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            finite_prediction_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "finite_shot_predictions",
                lambda: photonic_benchmark.parquet_bytes(
                    finite_predictions,
                    photonic_benchmark.finite_shot_prediction_arrow_schema(),
                ),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            model_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "photonic_models",
                lambda: photonic_benchmark.model_pickle_bytes(selection.selected_models),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            guard.checkpoint(serialization_live_bytes)
            restored = pickle.loads(model_bytes)
            guard.checkpoint(serialization_live_bytes)
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
            guard.checkpoint(serialization_live_bytes)
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
        except _HoldoutAccessIndeterminate as error:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate="holdout_access_indeterminate",
                message=f"photonic benchmark cannot safely reopen the holdout: {error}",
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
                    **holdout_access_audit,
                    "opened_after_selection_freeze_sha256": access_freeze_hash,
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
                    "path": "benchmark/baseline_predictions.parquet",
                    "sha256": baseline_predictions_sha256,
                    "policy": "same_authenticated_holdout_after_photonic_freeze",
                },
                "scientific_decision": "NOT_EVALUATED",
            }
        )
        if optional_selection is not None:
            selection_manifest["holdout_audit"]["selection_freezes"] = {
                "primary_2photon": freeze_hash,
                "diagnostic_3photon": optional_selection.selection_manifest[
                    "selection_freeze_sha256"
                ],
            }
            selection_manifest["baseline_predictions"][
                "opened_after_selection_freeze_sha256"
            ] = access_freeze_hash
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
        if config.photonics.enable_three_photon_profile:
            diagnostics["diagnostic_3photon"] = optional_summary
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
            Path("benchmark/baseline_predictions.parquet"),
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
            *trusted_circuit_paths,
            *optional_paths,
            development_reference_relative,
            "build-splits/feature_manifest.json",
            "build-splits/stage_manifest.json",
            *holdout_access_hashes,
            *baseline_inputs,
        )
        benchmark_prerequisite_allowlist = {
            "train-baselines": baseline_inputs,
            "build-photonic-features": (
                development_relative,
                circuit_manifest_relative,
                feature_manifest_relative,
                state_spaces_relative,
                *trusted_circuit_paths,
                *optional_paths,
            ),
        }
        try:
            for relative, expected_sha256 in holdout_access_hashes.items():
                _payload, captured_live_bytes = self._capture_report_artifact(
                    store=store,
                    relative_path=relative,
                    expected_sha256=expected_sha256,
                    guard=guard,
                    live_bytes=captured_live_bytes,
                )
                captured_input_hashes[relative] = expected_sha256
            if set(captured_input_hashes) != set(input_paths):
                raise RuntimeError("benchmark authenticated input snapshot is incomplete")
            selection_manifest_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "selection_manifest_json",
                lambda: self._benchmark_json_bytes(selection_manifest),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            model_manifest_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "model_manifest_json",
                lambda: self._benchmark_json_bytes(model_manifest),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            diagnostics_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "diagnostics_json",
                lambda: self._benchmark_json_bytes(diagnostics),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            ledger_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "resource_ledger_json",
                lambda: self._benchmark_json_bytes(ledger),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )
            result_bytes, serialization_live_bytes = self._serialize_benchmark_payload(
                "stage_result_json",
                lambda: self._benchmark_json_bytes(deterministic_data(result)),
                guard=guard,
                live_bytes=serialization_live_bytes,
            )

            def publication_checkpoint() -> None:
                guard.checkpoint(serialization_live_bytes)

            with store.stage(
                "benchmark",
                input_paths=input_paths,
                details={
                    "result_status": "success",
                    "scientific_decision": "NOT_EVALUATED",
                    "selection_freeze_sha256": freeze_hash,
                    "holdout_materialization_count": 1,
                },
                prerequisite_verification="selected_outputs",
                prerequisite_output_allowlist=benchmark_prerequisite_allowlist,
                preverified_input_hashes=captured_input_hashes,
                preverified_prerequisite_manifest_hashes={
                    "train-baselines": baseline_marker_hash,
                    "build-photonic-features": photonic_marker_hash,
                },
                commit_guard=publication_checkpoint,
                private_write_guard=publication_checkpoint,
            ) as transaction:
                transaction.write_bytes("selection_manifest.json", selection_manifest_bytes)
                transaction.write_bytes("validation_scores.parquet", validation_bytes)
                transaction.write_bytes("holdout_photonic_features.parquet", feature_bytes)
                transaction.write_bytes("baseline_predictions.parquet", baseline_prediction_bytes)
                transaction.write_bytes("exact_predictions.parquet", exact_prediction_bytes)
                transaction.write_bytes("finite_shot_counts.parquet", finite_count_bytes)
                transaction.write_bytes("finite_shot_predictions.parquet", finite_prediction_bytes)
                transaction.write_bytes("models/photonic_models.pkl", model_bytes)
                transaction.write_bytes("models/manifest.json", model_manifest_bytes)
                transaction.write_bytes("diagnostics.json", diagnostics_bytes)
                for name, payload in optional_payloads.items():
                    transaction.write_bytes(f"diagnostic_3photon/{name}", payload)
                transaction.write_bytes("resource_ledger.json", ledger_bytes)
                transaction.write_bytes("stage_result.json", result_bytes)
        except _ResourceLimitExceeded as error:
            return self._publish_benchmark_invalid(
                store=store,
                guard=guard,
                gate=f"resource_{error.gate}",
                message="photonic benchmark publication exceeded a configured resource cap",
            )
        return result

    @staticmethod
    def _open_report_worker_file(path: Path, payload: bytes | None = None) -> int:
        parent_before = path.parent.lstat()
        if (
            path.exists()
            or path.is_symlink()
            or path.is_junction()
            or path.parent.is_symlink()
            or path.parent.is_junction()
            or not stat.S_ISDIR(parent_before.st_mode)
            or parent_before.st_nlink != 1
        ):
            raise ReportContractError("report worker snapshot path is unsafe")
        if os.name == "nt":
            import _winapi
            import msvcrt

            handle = _winapi.CreateFile(
                str(path),
                _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | WRITE | DELETE
                0,
                1,  # CREATE_NEW
                0x80,  # FILE_ATTRIBUTE_NORMAL
                0,
            )
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR | getattr(os, "O_BINARY", 0))
            except BaseException:
                with suppress(OSError):
                    _winapi.CloseHandle(handle)
                raise
        else:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ReportContractError("report worker file is not regular single-link")
            if payload is not None:
                view = memoryview(payload)
                written = 0
                while written < len(view):
                    count = os.write(descriptor, view[written : written + 1024 * 1024])
                    if count <= 0:
                        raise OSError("short write")
                    written += count
                os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _close_report_worker_files(descriptors: Mapping[str, int]) -> None:
        for descriptor in descriptors.values():
            with suppress(OSError):
                os.close(descriptor)

    @staticmethod
    def _cleanup_abandoned_report_workers(store: RunStore) -> None:
        root = store.root.resolve()
        for candidate in store.root.glob(".report-worker-*"):
            try:
                candidate_stat = candidate.lstat()
            except FileNotFoundError:
                continue
            if (
                candidate.is_symlink()
                or candidate.is_junction()
                or not stat.S_ISDIR(candidate_stat.st_mode)
                or candidate.resolve(strict=False).parent != root
            ):
                raise ReportContractError("abandoned report worker path is unsafe")
            suffix = candidate.name.removeprefix(".report-worker-")
            pid_text = suffix.split("-", 1)[0]
            if pid_text.isdigit() and psutil.pid_exists(int(pid_text)):
                continue
            shutil.rmtree(candidate)

    @staticmethod
    def _validated_report_decision_evidence(
        *,
        captured_claims: Mapping[str, bytes],
        worker_status: StageStatus,
        worker_decision: str,
        worker_conclusion: str,
        expected_science_eligible: bool,
        expected_data_provenance: str,
        expected_approved_by: str | None,
        expected_approved_at: str | None,
    ) -> tuple[StageStatus, str, str]:
        def object_from(name: str) -> dict[str, Any]:
            try:
                value: object = json.loads(captured_claims[name].decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ReportContractError(f"report decision evidence is invalid: {name}") from error
            if not isinstance(value, dict):
                raise ReportContractError(f"report decision evidence must be an object: {name}")
            return value

        decision = object_from("decision.json")
        scores = object_from("scores.json")
        manifest = object_from("report_manifest.json")
        manifest_scores = manifest.get("scores")
        if not isinstance(manifest_scores, dict) or manifest_scores != scores:
            raise ReportContractError("report decision evidence is inconsistent")
        try:
            csv_text = captured_claims["scores.csv"].decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            markdown = captured_claims["report.md"].decode("utf-8")
        except (KeyError, UnicodeDecodeError, csv.Error) as error:
            raise ReportContractError("report decision evidence is invalid") from error
        if len(rows) != 1:
            raise ReportContractError("report decision evidence CSV is invalid")
        row = rows[0]
        raw_status = decision.get("stage_status")
        raw_decision = decision.get("scientific_decision")
        raw_conclusion = decision.get("conclusion")
        try:
            stage_status = StageStatus(str(raw_status))
        except ValueError as error:
            raise ReportContractError("report decision evidence status is invalid") from error
        expected_status = StageStatus.SUCCESS if expected_science_eligible else StageStatus.BLOCKED
        expected_decisions = (
            {"go", "stop", "inconclusive"} if expected_science_eligible else {"not_evaluated"}
        )
        rationale = f"Decision rationale: `{raw_conclusion}`."
        if (
            stage_status not in _REPORT_STATUS_DECISIONS
            or raw_decision not in _REPORT_STATUS_DECISIONS[stage_status]
            or _REPORT_DECISION_CONCLUSIONS.get((stage_status, str(raw_decision))) != raw_conclusion
            or stage_status is not expected_status
            or raw_decision not in expected_decisions
            or (stage_status is StageStatus.INVALID and raw_conclusion != "resource_gate_failed")
            or not isinstance(raw_conclusion, str)
            or not raw_conclusion
            or decision.get("schema_version") != "1.0"
            or decision.get("science_eligible") is not expected_science_eligible
            or scores.get("science_eligible") is not expected_science_eligible
            or row.get("science_eligible") != str(expected_science_eligible)
            or decision.get("decision_basis")
            != "exact_primary_system_score_calibration_resource_gates"
            or decision.get("finite_shot_used") is not False
            or scores.get("stage_status") != stage_status.value
            or scores.get("decision") != raw_decision
            or scores.get("conclusion") != raw_conclusion
            or manifest.get("scientific_decision") != raw_decision
            or manifest.get("conclusion") != raw_conclusion
            or manifest.get("data_provenance") != expected_data_provenance
            or manifest.get("protocol_approved_by") != expected_approved_by
            or manifest.get("protocol_approved_at") != expected_approved_at
            or row.get("stage_status") != stage_status.value
            or row.get("decision") != raw_decision
            or row.get("conclusion") != raw_conclusion
            or markdown.count(
                f"**Status:** {stage_status.value.upper()} / {str(raw_decision).upper()}"
            )
            != 1
            or markdown.count("Decision rationale:") != 1
            or markdown.count(rationale) != 1
            or worker_status is not stage_status
            or worker_decision != raw_decision
            or worker_conclusion != raw_conclusion
        ):
            raise ReportContractError("report decision evidence is inconsistent")
        return stage_status, str(raw_decision), raw_conclusion

    @staticmethod
    def _validate_report_worker_outputs(
        *,
        output_descriptors: Mapping[str, int],
        worker_result: object,
        guard: _ResourceGuard,
        live_bytes: int,
        expected_science_eligible: bool,
        expected_data_provenance: str,
        expected_approved_by: str | None,
        expected_approved_at: str | None,
    ) -> tuple[
        StageStatus,
        str,
        str,
        dict[str, tuple[bytes, str, int]],
        int,
    ]:
        if not isinstance(worker_result, dict) or worker_result.get("schema_version") != "1.0":
            raise ReportContractError("report worker result is invalid")
        try:
            worker_manifest_size = len(
                json.dumps(
                    worker_result,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            raise ReportContractError("report worker result is not bounded JSON") from error
        if worker_manifest_size > 1024 * 1024:
            raise ReportContractError("report worker result exceeds its size limit")
        worker_status = worker_result.get("status")
        if worker_status == "contract_error":
            message = worker_result.get("message")
            if not isinstance(message, str) or not message or len(message) > 1000:
                raise ReportContractError("report worker contract error is malformed")
            raise ReportContractError(message)
        if worker_status != "success":
            raise ReportContractError("report worker status is invalid")
        try:
            stage_status = StageStatus(str(worker_result.get("stage_status")))
        except ValueError as error:
            raise ReportContractError("report worker stage status is invalid") from error
        if stage_status not in {StageStatus.SUCCESS, StageStatus.BLOCKED}:
            raise ReportContractError("report worker stage status is not publishable")
        decision = worker_result.get("scientific_decision")
        conclusion = worker_result.get("conclusion")
        if (
            not isinstance(decision, str)
            or decision not in {"go", "stop", "inconclusive", "not_evaluated"}
            or not isinstance(conclusion, str)
            or not conclusion
            or len(conclusion) > 1000
        ):
            raise ReportContractError("report worker decision evidence is invalid")
        expected_names = {
            "report.md",
            "timestamp_scores.parquet",
            "evaluated_predictions.parquet",
            "bootstrap_replicates.parquet",
            "scores.json",
            "scores.csv",
            "decision.json",
            "report_manifest.json",
        }
        raw_outputs = worker_result.get("outputs")
        if not isinstance(raw_outputs, dict) or set(raw_outputs) != expected_names:
            raise ReportContractError("report worker output inventory is invalid")
        if set(output_descriptors) != expected_names:
            raise ReportContractError("report worker output handle inventory is invalid")
        schemas = report_parquet_schemas()
        claim_names = {
            "decision.json",
            "scores.json",
            "report_manifest.json",
            "scores.csv",
            "report.md",
        }
        records: dict[str, tuple[int, str, int, tuple[int, int], int]] = {}
        output_bytes = 0
        for name in sorted(expected_names):
            record = raw_outputs[name]
            if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
                raise ReportContractError(f"report worker output record is invalid: {name}")
            declared_digest = record.get("sha256")
            declared_size = record.get("size")
            if (
                not isinstance(declared_digest, str)
                or len(declared_digest) != 64
                or any(character not in "0123456789abcdef" for character in declared_digest)
                or isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                raise ReportContractError(f"report worker output metadata is invalid: {name}")
            try:
                descriptor = output_descriptors[name]
                before = os.fstat(descriptor)
            except OSError as error:
                raise ReportContractError(
                    f"report worker output is missing or inaccessible: {name}"
                ) from error
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != declared_size
            ):
                raise ReportContractError(f"report worker output is unsafe: {name}")
            output_bytes += declared_size
            records[name] = (
                descriptor,
                declared_digest,
                declared_size,
                (before.st_dev, before.st_ino),
                before.st_mtime_ns,
            )
        capture_overhead = 4096 * len(expected_names)
        guard.checkpoint(live_bytes + 2 * output_bytes + capture_overhead)

        captured_claims: dict[str, bytes] = {}
        verified: dict[str, tuple[bytes, str, int]] = {}
        captured_bytes = 0
        for name in sorted(expected_names):
            descriptor, declared_digest, declared_size, identity, initial_mtime_ns = records[name]
            buffer = bytearray(declared_size)
            offset = 0
            os.lseek(descriptor, 0, os.SEEK_SET)
            while offset < declared_size:
                chunk_size = min(1024 * 1024, declared_size - offset)
                guard.checkpoint(
                    live_bytes + captured_bytes + declared_size + chunk_size + capture_overhead
                )
                chunk = os.read(descriptor, chunk_size)
                if not chunk:
                    raise ReportContractError(f"report worker output ended during capture: {name}")
                buffer[offset : offset + len(chunk)] = chunk
                offset += len(chunk)
            if os.read(descriptor, 1):
                raise ReportContractError(
                    f"report worker output exceeded its declared size: {name}"
                )
            after_capture = os.fstat(descriptor)
            if (
                after_capture.st_size != declared_size
                or after_capture.st_mtime_ns != initial_mtime_ns
                or (after_capture.st_dev, after_capture.st_ino) != identity
            ):
                raise ReportContractError(f"report worker output changed: {name}")
            guard.checkpoint(live_bytes + captured_bytes + 2 * declared_size + capture_overhead)
            payload = bytes(buffer)
            del buffer
            captured_bytes += declared_size
            guard.checkpoint(live_bytes + captured_bytes + capture_overhead)
            digest = hashlib.sha256()
            try:
                digest.update(payload)
                if name in schemas:
                    actual_schema = pq.ParquetFile(
                        pa.BufferReader(payload)
                    ).schema_arrow.remove_metadata()
                    if not actual_schema.equals(
                        schemas[name].remove_metadata(), check_metadata=False
                    ):
                        raise ReportContractError(
                            f"report worker Parquet schema is invalid: {name}"
                        )
            except (OSError, pa.ArrowException) as error:
                raise ReportContractError(
                    f"report worker output cannot be verified: {name}"
                ) from error
            actual_digest = digest.hexdigest()
            if actual_digest != declared_digest:
                raise ReportContractError(f"report worker output checksum is invalid: {name}")
            if name in claim_names:
                if declared_size > 8 * 1024 * 1024:
                    raise ReportContractError(
                        f"report decision evidence exceeds its size limit: {name}"
                    )
                captured_claims[name] = payload
            verified[name] = (payload, actual_digest, declared_size)
        stage_status, decision, conclusion = (
            DefaultPilotServices._validated_report_decision_evidence(
                captured_claims=captured_claims,
                worker_status=stage_status,
                worker_decision=decision,
                worker_conclusion=conclusion,
                expected_science_eligible=expected_science_eligible,
                expected_data_provenance=expected_data_provenance,
                expected_approved_by=expected_approved_by,
                expected_approved_at=expected_approved_at,
            )
        )
        return stage_status, decision, conclusion, verified, output_bytes

    @staticmethod
    def _publish_report_invalid(
        *,
        store: RunStore,
        guard: _ResourceGuard,
        gate: str,
        message: str,
    ) -> StageResult:
        result = StageResult(
            stage="report",
            status=StageStatus.INVALID,
            message=message,
            artifacts=(
                Path("report/report.md"),
                Path("report/scores.json"),
                Path("report/scores.csv"),
                Path("report/decision.json"),
                Path("report/report_manifest.json"),
                Path("report/resource_ledger.json"),
            ),
            details=(
                DetailEntry(key="resource_gate", value=gate),
                DetailEntry(key="scientific_decision", value="NOT_EVALUATED"),
            ),
        )
        diagnostic = {
            "schema_version": "1.0",
            "stage_status": "invalid",
            "scientific_decision": "not_evaluated",
            "conclusion": "resource_gate_failed",
            "science_eligible": False,
            "decision_basis": "resource_gate",
            "finite_shot_used": False,
            "gate": gate,
        }
        if (
            diagnostic["scientific_decision"] not in _REPORT_STATUS_DECISIONS[StageStatus.INVALID]
            or diagnostic["conclusion"] != "resource_gate_failed"
        ):
            raise ReportContractError("invalid report decision evidence is inconsistent")
        ledger = {
            "schema_version": "1.0",
            "stage": "report",
            "status": "invalid",
            "resource_status": "EXCEEDED",
            "resource_gate": gate,
            "wall_clock_seconds": time.perf_counter() - guard.started,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "peak_declared_bytes": guard.peak_declared_bytes,
            "max_wall_clock_seconds": guard.wall_seconds,
            "max_memory_bytes": guard.memory_bytes,
            "cpu_only": True,
        }
        with store.stage(
            "report",
            terminal_status="INVALID",
            prerequisite_verification="marker_only",
            details={
                "result_status": "invalid",
                "scientific_decision": "NOT_EVALUATED",
                "resource_gate": gate,
            },
        ) as transaction:
            transaction.write_text(
                "report.md",
                "# Photonic PV Pilot — Report unavailable\n\n"
                f"**Status:** INVALID / NOT_EVALUATED\n\n{message}\n\n"
                "Decision rationale: `resource_gate_failed`.\n",
            )
            transaction.write_json("scores.json", diagnostic)
            transaction.write_text(
                "scores.csv",
                "stage_status,scientific_decision,conclusion,science_eligible,gate\n"
                f"invalid,not_evaluated,resource_gate_failed,False,{gate}\n",
            )
            transaction.write_json("decision.json", diagnostic)
            transaction.write_json(
                "report_manifest.json",
                {
                    **diagnostic,
                    "pipeline_scope": "G0-G3",
                    "limitations": ["report_resource_gate_failed"],
                },
            )
            transaction.write_json("resource_ledger.json", ledger)
            transaction.write_json("stage_result.json", deterministic_data(result))
        return result

    def report(self, run_dir: Path) -> StageResult:
        """Score frozen G3 artifacts and publish a read-only decision report."""

        started = time.perf_counter()
        store = RunStore.open(run_dir)
        try:
            self._cleanup_abandoned_report_workers(store)
        except (OSError, ReportContractError) as error:
            raise ValueError(f"report worker staging is invalid: {error}") from error
        existing = self._existing_report_result(store)
        if existing is not None:
            return existing
        config = self._run_config(store)
        guard = _ResourceGuard(
            started=started,
            wall_seconds=config.resources.max_wall_clock_seconds,
            memory_bytes=int(config.resources.max_memory_gb * 1024**3),
        )
        worker_root: Path | None = None
        worker_descriptors: dict[str, int] = {}
        worker_transfers: list[object] = []
        try:
            (
                config,
                captured,
                input_hashes,
                stage_manifests,
                inventory,
                total_file_bytes,
            ) = self._capture_report_provenance(store, guard)
            required: dict[str, tuple[str, ...]] = {
                "validate-data": (
                    "validate-data/data_card.json",
                    "validate-data/cleaning_ledger.json",
                    "validate-data/stage_result.json",
                ),
                "build-splits": (
                    "build-splits/feature_manifest.json",
                    "build-splits/minmax_scalers.json",
                    "build-splits/clip_rates.json",
                    "build-splits/resource_ledger.json",
                    "build-splits/stage_result.json",
                ),
                "train-baselines": (
                    "train-baselines/selection_manifest.json",
                    "train-baselines/diagnostics.json",
                    "train-baselines/models/baseline_models.pkl",
                    "train-baselines/models/manifest.json",
                    "train-baselines/resource_ledger.json",
                    "train-baselines/stage_result.json",
                ),
                "build-photonic-features": (
                    "build-photonic-features/development_photonic_features.parquet",
                    "build-photonic-features/state_spaces.json",
                    "build-photonic-features/circuit_manifest.json",
                    "build-photonic-features/feature_manifest.json",
                    "build-photonic-features/resource_ledger.json",
                    "build-photonic-features/stage_result.json",
                ),
                "benchmark": (
                    "benchmark/selection_manifest.json",
                    "benchmark/baseline_predictions.parquet",
                    "benchmark/exact_predictions.parquet",
                    "benchmark/finite_shot_predictions.parquet",
                    "benchmark/diagnostics.json",
                    "benchmark/models/photonic_models.pkl",
                    "benchmark/models/manifest.json",
                    "benchmark/resource_ledger.json",
                    "benchmark/stage_result.json",
                ),
            }
            required_fixed = {path for paths in required.values() for path in paths}
            if not required_fixed <= set(inventory):
                raise ReportContractError("report fixed input inventory is incomplete")
            output_hashes_by_stage = {
                stage: RunStore._exact_output_map(marker, stage)
                for stage, marker in stage_manifests.items()
            }
            for relative in sorted(inventory):
                if relative in captured:
                    continue
                guard.checkpoint(total_file_bytes)
                stage = relative.split("/", 1)[0]
                expected = output_hashes_by_stage.get(stage, {}).get(relative)
                if expected is None:
                    raise ReportContractError(
                        f"captured stage marker does not declare report input: {relative}"
                    )
                payload, total_file_bytes = self._capture_report_artifact(
                    store, relative, expected, guard, total_file_bytes
                )
                captured[relative] = payload
                input_hashes[relative] = hashlib.sha256(payload).hexdigest()

            def captured_object(relative: str) -> dict[str, Any]:
                return self._object_from_captured(captured[relative], relative)

            circuit_manifest = captured_object("build-photonic-features/circuit_manifest.json")
            raw_circuits = circuit_manifest.get("circuits")
            if not isinstance(raw_circuits, list):
                raise ReportContractError("circuit manifest has no circuit inventory")
            declared_circuits: set[str] = set()
            for item in raw_circuits:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ReportContractError("circuit inventory contains an invalid path")
                path = str(item["path"])
                if (
                    path in declared_circuits
                    or path not in captured
                    or input_hashes[path] != item.get("npz_sha256")
                ):
                    raise ReportContractError(
                        "circuit NPZ checksum/path does not match the exact inventory"
                    )
                declared_circuits.add(path)
            captured_circuits = {
                path for path in captured if path.startswith("build-photonic-features/circuits/")
            }
            if declared_circuits != captured_circuits:
                raise ReportContractError("circuit manifest inventory is missing or extra")
            selection_manifest = captured_object("benchmark/selection_manifest.json")
            baseline_selection = captured_object("train-baselines/selection_manifest.json")
            baseline_model_manifest = captured_object("train-baselines/models/manifest.json")
            benchmark_model_manifest = captured_object("benchmark/models/manifest.json")
            split_feature_manifest = captured_object("build-splits/feature_manifest.json")
            photonic_feature_manifest = captured_object(
                "build-photonic-features/feature_manifest.json"
            )
            state_spaces = captured_object("build-photonic-features/state_spaces.json")
            benchmark_diagnostics = captured_object("benchmark/diagnostics.json")
            data_card = captured_object("validate-data/data_card.json")
            if (
                data_card.get("provenance") != config.protocol.provenance.value
                or bool(data_card.get("synthetic"))
                != (config.protocol.provenance.value == "synthetic")
                or data_card.get("status") != "success"
            ):
                raise ReportContractError("data-card provenance does not match the frozen config")
            ledger_paths = {
                "validate-data": "validate-data/cleaning_ledger.json",
                "build-splits": "build-splits/resource_ledger.json",
                "train-baselines": "train-baselines/resource_ledger.json",
                "build-photonic-features": "build-photonic-features/resource_ledger.json",
                "benchmark": "benchmark/resource_ledger.json",
            }
            resource_ledgers = {
                stage: captured_object(path) for stage, path in ledger_paths.items()
            }
            environment_manifest = captured_object("environment_manifest.json")
            provenance = {
                "run_id": store.root.name,
                "config_sha256": store.config_hash,
                "canonical_data_sha256": data_card.get("canonical_sha256"),
                "environment_manifest_sha256": input_hashes["environment_manifest.json"],
                "source_manifest_sha256": input_hashes["source_manifest.json"],
                "run_manifest_sha256": input_hashes["run_manifest.json"],
                "code": environment_manifest.get("git"),
                "stage_manifest_sha256": {
                    stage: input_hashes[f"{stage}/stage_manifest.json"]
                    for stage in (
                        "validate-data",
                        "build-splits",
                        "train-baselines",
                        "build-photonic-features",
                        "benchmark",
                    )
                },
            }
            staging_parent = store.root.resolve()
            worker_root = Path(
                tempfile.mkdtemp(prefix=f".report-worker-{os.getpid()}-", dir=staging_parent)
            )
            if (
                worker_root.resolve().parent != staging_parent
                or worker_root.is_symlink()
                or worker_root.is_junction()
            ):
                raise ReportContractError("report worker staging escaped the run directory")
            input_root = worker_root / "inputs"
            output_root = worker_root / "outputs"
            control_root = worker_root / "control"
            input_root.mkdir()
            output_root.mkdir()
            control_root.mkdir()
            snapshots = {
                "exact_predictions.parquet": captured["benchmark/exact_predictions.parquet"],
                "baseline_predictions.parquet": captured["benchmark/baseline_predictions.parquet"],
                "finite_shot_predictions.parquet": captured[
                    "benchmark/finite_shot_predictions.parquet"
                ],
            }
            input_descriptors: dict[str, int] = {}
            for name, payload in snapshots.items():
                guard.checkpoint(total_file_bytes)
                descriptor = self._open_report_worker_file(input_root / name, payload)
                input_descriptors[name] = descriptor
                worker_descriptors[f"input:{name}"] = descriptor
            output_names = {
                "report.md",
                "timestamp_scores.parquet",
                "evaluated_predictions.parquet",
                "bootstrap_replicates.parquet",
                "scores.json",
                "scores.csv",
                "decision.json",
                "report_manifest.json",
            }
            output_descriptors: dict[str, int] = {}
            for name in sorted(output_names):
                descriptor = self._open_report_worker_file(output_root / name)
                output_descriptors[name] = descriptor
                worker_descriptors[f"output:{name}"] = descriptor
            versions = {
                "protocol": config.protocol.version,
                "canonical_data_schema": data_card.get("canonical_schema_version"),
                "split_feature_schema": split_feature_manifest.get("schema_version"),
                "photonic_feature_schema": photonic_feature_manifest.get("schema_version"),
                "benchmark_schema": selection_manifest.get("schema_version"),
                "prediction_fit_phase": "train_plus_validation_refit",
            }
            input_transfers: dict[str, object] = {}
            output_transfers: dict[str, object] = {}
            try:
                for name, descriptor in input_descriptors.items():
                    transfer = duplicate_descriptor_for_worker(descriptor, writable=False)
                    worker_transfers.append(transfer)
                    input_transfers[name] = transfer
                for name, descriptor in output_descriptors.items():
                    transfer = duplicate_descriptor_for_worker(descriptor, writable=True)
                    worker_transfers.append(transfer)
                    output_transfers[name] = transfer
                worker_request = {
                    "input_handles": input_transfers,
                    "output_handles": output_transfers,
                    "snapshot_sha256": {
                        "exact_predictions.parquet": input_hashes[
                            "benchmark/exact_predictions.parquet"
                        ],
                        "baseline_predictions.parquet": input_hashes[
                            "benchmark/baseline_predictions.parquet"
                        ],
                        "finite_shot_predictions.parquet": input_hashes[
                            "benchmark/finite_shot_predictions.parquet"
                        ],
                    },
                    "config": config.model_dump(mode="json"),
                    "selection_manifest": selection_manifest,
                    "circuit_manifest": circuit_manifest,
                    "state_spaces": state_spaces,
                    "benchmark_diagnostics": benchmark_diagnostics,
                    "benchmark_model_manifest": benchmark_model_manifest,
                    "resource_ledgers": resource_ledgers,
                    "provenance": provenance,
                    "input_hashes": input_hashes,
                    "versions": versions,
                    "baseline_selection": baseline_selection,
                    "baseline_model_manifest": baseline_model_manifest,
                    "optional_artifacts": {
                        path: captured[path]
                        for path in sorted(captured)
                        if path in optional_photonics.PROFILE_FIXED_PATHS
                        or path.startswith(f"{optional_photonics.PROFILE_PREFIX}/circuits/")
                        or path in (
                            optional_photonics.DIAGNOSTIC_PATHS[0],
                            optional_photonics.DIAGNOSTIC_PATHS[3],
                            optional_photonics.DIAGNOSTIC_PATHS[4],
                            optional_photonics.DIAGNOSTIC_PATHS[-1],
                        )
                    },
                    "profile_input_hashes": stage_manifests[
                        "build-photonic-features"
                    ]["input_hashes"],
                }
                executor = _ReportProcessExecutor(
                    deadline_monotonic=time.monotonic()
                    + max(
                        0.0,
                        guard.wall_seconds - (time.perf_counter() - guard.started),
                    ),
                    resource_checkpoint=guard.checkpoint,
                    private_directory=control_root,
                    max_memory_bytes=guard.memory_bytes,
                    telemetry_callback=guard.record_fit_rss,
                )
                worker_result = executor.run(
                    "report-decode-evaluate-serialize",
                    run_report_worker,
                    worker_request,
                    declared_bytes=total_file_bytes,
                )
            finally:
                for transfer in worker_transfers:
                    close_descriptor_transfer(transfer)
                worker_transfers.clear()
            (
                report_status,
                report_decision,
                report_conclusion,
                worker_outputs,
                worker_output_bytes,
            ) = self._validate_report_worker_outputs(
                output_descriptors=output_descriptors,
                worker_result=worker_result,
                guard=guard,
                live_bytes=total_file_bytes,
                expected_science_eligible=config.science_decision_allowed,
                expected_data_provenance=config.protocol.provenance.value,
                expected_approved_by=config.protocol.approved_by,
                expected_approved_at=(
                    config.protocol.approved_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    if config.protocol.approved_at is not None
                    else None
                ),
            )
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            publication_live_bytes = total_file_bytes + worker_output_bytes
            guard.checkpoint(publication_live_bytes)
        except _ResourceLimitExceeded as error:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            return self._publish_report_invalid(
                store=store,
                guard=guard,
                gate=f"resource_{error.gate}",
                message="report generation exceeded a configured resource cap",
            )
        except baseline_models.FitMemoryExceeded:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            return self._publish_report_invalid(
                store=store,
                guard=guard,
                gate="report_worker_memory_rss",
                message="report worker exceeded the configured aggregate RSS cap",
            )
        except baseline_models.FitDeadlineExceeded:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            return self._publish_report_invalid(
                store=store,
                guard=guard,
                gate="report_worker_wall_clock",
                message="report worker exceeded the configured wall-clock cap",
            )
        except RuntimeError as error:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            return self._publish_report_invalid(
                store=store,
                guard=guard,
                gate="report_worker_failed",
                message=f"report worker failed: {str(error)[:500]}",
            )
        except (EvaluationContractError, ReportContractError, OSError) as error:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            raise ValueError(f"report artifact contract is invalid: {error}") from error

        terminal_status = cast(
            Literal["SUCCESS", "BLOCKED", "INVALID"], report_status.value.upper()
        )
        artifacts = (
            Path("report/report.md"),
            Path("report/timestamp_scores.parquet"),
            Path("report/evaluated_predictions.parquet"),
            Path("report/bootstrap_replicates.parquet"),
            Path("report/scores.json"),
            Path("report/scores.csv"),
            Path("report/decision.json"),
            Path("report/report_manifest.json"),
            Path("report/resource_ledger.json"),
        )
        result = StageResult(
            stage="report",
            status=report_status,
            message=(
                "pilot report produced; scientific decision is not evaluated"
                if report_status is StageStatus.BLOCKED
                else f"pilot report produced: {report_decision}"
            ),
            artifacts=artifacts,
            details=(
                DetailEntry(key="scientific_decision", value=report_decision),
                DetailEntry(key="conclusion", value=report_conclusion),
            ),
        )
        guard.sample_without_enforcement()
        ledger = {
            "schema_version": "1.0",
            "stage": "report",
            "status": report_status.value,
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

        def publication_checkpoint() -> None:
            guard.checkpoint(publication_live_bytes)

        try:
            with store.stage(
                "report",
                input_paths=tuple(sorted(captured)),
                terminal_status=terminal_status,
                prerequisite_verification="selected_outputs",
                prerequisite_output_allowlist=self._report_benchmark_allowlist(),
                details={
                    "result_status": report_status.value,
                    "scientific_decision": report_decision,
                    "conclusion": report_conclusion,
                    "read_only_consumer": True,
                },
                commit_guard=publication_checkpoint,
                private_write_guard=publication_checkpoint,
                preverified_input_hashes=input_hashes,
            ) as transaction:
                for name in sorted(worker_outputs):
                    payload, _digest, _size = worker_outputs[name]
                    transaction.write_bytes(name, payload)
                transaction.write_json("resource_ledger.json", ledger)
                transaction.write_json("stage_result.json", deterministic_data(result))
        except _ResourceLimitExceeded as error:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
            return self._publish_report_invalid(
                store=store,
                guard=guard,
                gate=f"resource_{error.gate}",
                message="report publication exceeded a configured resource cap",
            )
        finally:
            self._close_report_worker_files(worker_descriptors)
            worker_descriptors.clear()
            if worker_root is not None:
                shutil.rmtree(worker_root, ignore_errors=True)
        return result

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> StageResult:
        """Initialize or resume the exact immutable G0-G3 stage sequence."""

        if (config is None) == (run_dir is None):
            raise ValueError("provide exactly one of config or run_dir")
        active_run = self.init_run(config) if config is not None else cast(Path, run_dir)
        stages: tuple[Callable[[Path], StageResult], ...] = (
            self.validate_data,
            self.build_splits,
            self.train_baselines,
            self.build_photonic_features,
            self.benchmark,
            self.report,
        )
        final: StageResult | None = None
        successful_stages: list[str] = []

        def with_run_path(result: StageResult) -> StageResult:
            message = result.message or result.stage
            details = (
                *(item for item in result.details if item.key != "run_dir"),
                DetailEntry(key="run_dir", value=active_run.as_posix()),
            )
            return result.model_copy(
                update={
                    "message": f"{message} [run: {active_run}]",
                    "details": details,
                }
            )

        for stage in stages:
            final = stage(active_run).model_copy(update={"completion_kind": None})
            if final.status is not StageStatus.SUCCESS:
                details = {item.key: item.value for item in final.details}
                expected_synthetic_terminal = (
                    final.stage == "report"
                    and final.status is StageStatus.BLOCKED
                    and tuple(successful_stages)
                    == (
                        "validate-data",
                        "build-splits",
                        "train-baselines",
                        "build-photonic-features",
                        "benchmark",
                    )
                    and details.get("scientific_decision") == "not_evaluated"
                    and details.get("conclusion") == "not_evaluated"
                )
                if expected_synthetic_terminal:
                    store = RunStore.open(active_run)
                    run_config = self._run_config(store)
                    if run_config.protocol.provenance is DataProvenance.SYNTHETIC:
                        final = final.model_copy(
                            update={"completion_kind": CompletionKind.SYNTHETIC_SOFTWARE_ACCEPTANCE}
                        )
                return with_run_path(final)
            successful_stages.append(final.stage)
        if final is None:
            raise RuntimeError("pipeline has no stages")
        return with_run_path(final)


def init_run(config: Path) -> Path:
    """Initialize a run from a config and return its new immutable directory."""

    return DefaultPilotServices().init_run(config)


__all__ = ["DefaultPilotServices", "init_run"]
