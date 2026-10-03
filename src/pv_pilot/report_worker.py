"""Spawn-isolated report decode, evaluation, and serialization worker."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from .config import PilotConfig
from .evaluation import EvaluationContractError
from .reporting import ReportContractError, build_report_products

_INPUT_NAMES = {
    "exact": "exact_predictions.parquet",
    "baseline": "baseline_predictions.parquet",
    "finite": "finite_shot_predictions.parquet",
}
_OUTPUT_ATTRIBUTES = {
    "report.md": "report_markdown",
    "timestamp_scores.parquet": "timestamp_scores_parquet",
    "evaluated_predictions.parquet": "evaluated_predictions_parquet",
    "bootstrap_replicates.parquet": "bootstrap_replicates_parquet",
    "scores.json": "scores_json",
    "scores.csv": "scores_csv",
    "decision.json": "decision_json",
    "report_manifest.json": "report_manifest_json",
}


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _directory_identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    if _is_link_like(path) or not stat.S_ISDIR(value.st_mode) or value.st_nlink != 1:
        raise ReportContractError(f"report worker directory is unsafe: {path.name}")
    return value.st_dev, value.st_ino


def _ignore_resource_checkpoint(_declared_bytes: int = 0) -> None:
    return None


def _validated_directory(raw: object, *, parent: Path, name: str) -> Path:
    if not isinstance(raw, str):
        raise ReportContractError(f"report worker {name} directory is invalid")
    path = Path(raw)
    resolved = path.resolve(strict=False)
    before = path.lstat()
    if (
        _is_link_like(path)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_nlink != 1
        or resolved.parent != parent
        or resolved.name != name
    ):
        raise ReportContractError(f"report worker {name} directory is unsafe")
    return resolved


def _read_parquet_once(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    parent_identity = _directory_identity(path.parent)
    before = path.lstat()
    if _is_link_like(path) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ReportContractError(f"report worker input is unsafe: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != before.st_size
            or (before.st_ino and opened.st_ino != before.st_ino)
        ):
            raise ReportContractError(f"report worker input changed: {path.name}")
        digest = hashlib.sha256()
        captured = io.BytesIO()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            captured.write(chunk)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ReportContractError(
                f"report worker snapshot checksum is invalid: {path.name}"
            )
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or _directory_identity(path.parent) != parent_identity
        ):
            raise ReportContractError(f"report worker input changed: {path.name}")
        captured.seek(0)
        frame = pd.read_parquet(captured)
        if _directory_identity(path.parent) != parent_identity:
            raise ReportContractError(f"report worker input directory changed: {path.name}")
        return frame
    finally:
        os.close(descriptor)


def _write_output(path: Path, payload: bytes) -> dict[str, object]:
    parent_identity = _directory_identity(path.parent)
    if path.exists() or _is_link_like(path):
        raise ReportContractError(f"report worker output path is unsafe: {path.name}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    written = path.lstat()
    if (
        not stat.S_ISREG(written.st_mode)
        or written.st_nlink != 1
        or _directory_identity(path.parent) != parent_identity
    ):
        raise ReportContractError(f"report worker output is unsafe: {path.name}")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def run_report_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """Decode and evaluate only inside the resource-monitored subprocess."""

    try:
        staging_raw = request.get("staging_root")
        if not isinstance(staging_raw, str):
            raise ReportContractError("report worker staging root is invalid")
        staging_path = Path(staging_raw)
        if _is_link_like(staging_path):
            raise ReportContractError("report worker staging root is unsafe")
        staging = staging_path.resolve(strict=True)
        staging_stat = staging.lstat()
        if (
            _is_link_like(staging)
            or not stat.S_ISDIR(staging_stat.st_mode)
            or staging_stat.st_nlink != 1
        ):
            raise ReportContractError("report worker staging root is unsafe")
        staging_identity = _directory_identity(staging)
        input_root = _validated_directory(
            request.get("input_root"), parent=staging, name="inputs"
        )
        output_root = _validated_directory(
            request.get("output_root"), parent=staging, name="outputs"
        )
        input_identity = _directory_identity(input_root)
        output_identity = _directory_identity(output_root)
        snapshot_sha256 = request.get("snapshot_sha256")
        if (
            not isinstance(snapshot_sha256, dict)
            or set(snapshot_sha256) != set(_INPUT_NAMES.values())
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in snapshot_sha256.values()
            )
        ):
            raise ReportContractError("report worker snapshot checksum inventory is invalid")
        exact_name = _INPUT_NAMES["exact"]
        baseline_name = _INPUT_NAMES["baseline"]
        finite_name = _INPUT_NAMES["finite"]
        exact = _read_parquet_once(input_root / exact_name, snapshot_sha256[exact_name])
        baselines = _read_parquet_once(
            input_root / baseline_name, snapshot_sha256[baseline_name]
        )
        finite = _read_parquet_once(
            input_root / finite_name, snapshot_sha256[finite_name]
        )
        config = PilotConfig.model_validate(request.get("config"))
        products = build_report_products(
            config=config,
            exact_predictions=exact,
            baseline_predictions=baselines,
            finite_shot_predictions=finite,
            selection_manifest=request["selection_manifest"],
            circuit_manifest=request["circuit_manifest"],
            state_spaces=request["state_spaces"],
            benchmark_diagnostics=request["benchmark_diagnostics"],
            benchmark_model_manifest=request["benchmark_model_manifest"],
            resource_ledgers=request["resource_ledgers"],
            provenance=request["provenance"],
            input_hashes=request["input_hashes"],
            versions=request["versions"],
            baseline_selection=request["baseline_selection"],
            baseline_model_manifest=request["baseline_model_manifest"],
            initial_live_bytes=0,
            resource_checkpoint=_ignore_resource_checkpoint,
        )
        outputs: dict[str, dict[str, object]] = {}
        for name, attribute in _OUTPUT_ATTRIBUTES.items():
            payload = getattr(products, attribute)
            if not isinstance(payload, bytes):
                raise ReportContractError(f"report worker payload is invalid: {name}")
            outputs[name] = _write_output(output_root / name, payload)
        if (
            _directory_identity(staging) != staging_identity
            or _directory_identity(input_root) != input_identity
            or _directory_identity(output_root) != output_identity
        ):
            raise ReportContractError("report worker staging directory changed")
        bundle = products.evaluation.score_bundle
        return {
            "schema_version": "1.0",
            "status": "success",
            "stage_status": bundle.stage_status.value,
            "scientific_decision": bundle.decision.value,
            "conclusion": bundle.conclusion,
            "outputs": outputs,
        }
    except (
        EvaluationContractError,
        ReportContractError,
        ValueError,
        KeyError,
        OSError,
        pa.ArrowException,
    ) as error:
        return {
            "schema_version": "1.0",
            "status": "contract_error",
            "message": str(error)[:1000],
        }


__all__ = ["run_report_worker"]
