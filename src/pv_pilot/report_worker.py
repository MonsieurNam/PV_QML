"""Spawn-isolated report decode, evaluation, and serialization worker."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from multiprocessing import reduction
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


def duplicate_descriptor_for_worker(descriptor: int, *, writable: bool) -> object:
    """Return a spawn-picklable duplicate without exposing a data pathname."""

    if os.name == "nt":
        import _winapi
        import msvcrt

        access = _winapi.GENERIC_READ
        if writable:
            access |= _winapi.GENERIC_WRITE
        return reduction.DupHandle(msvcrt.get_osfhandle(descriptor), access)
    dup_fd = getattr(reduction, "DupFd", None)
    if dup_fd is None:  # pragma: no cover - supported CPython POSIX builds expose DupFd
        raise RuntimeError("descriptor transfer is unavailable")
    return dup_fd(descriptor)


def close_descriptor_transfer(transfer: object) -> None:
    """Close an unused parent-side duplicate after a worker run or failed spawn."""

    detach = getattr(transfer, "detach", None)
    if not callable(detach):
        return
    try:
        raw = detach()
    except (EOFError, OSError, ValueError):
        return
    if os.name == "nt":
        import _winapi

        with suppress(OSError):
            _winapi.CloseHandle(raw)
    else:
        with suppress(OSError):
            os.close(raw)


def _detach_descriptor(transfer: object, *, writable: bool) -> int:
    detach = getattr(transfer, "detach", None)
    if not callable(detach):
        raise ReportContractError("report worker descriptor transfer is invalid")
    raw = detach()
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ReportContractError("report worker descriptor transfer is invalid")
    if os.name == "nt":
        import _winapi
        import msvcrt

        flags = os.O_RDWR if writable else os.O_RDONLY
        try:
            return msvcrt.open_osfhandle(raw, flags | getattr(os, "O_BINARY", 0))
        except BaseException:
            with suppress(OSError):
                _winapi.CloseHandle(raw)
            raise
    return raw


def _ignore_resource_checkpoint(_declared_bytes: int = 0) -> None:
    return None


def _read_parquet_handle(
    transfer: object, expected_sha256: str, *, name: str
) -> pd.DataFrame:
    descriptor = _detach_descriptor(transfer, writable=False)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ReportContractError(f"report worker input is unsafe: {name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        captured = io.BytesIO()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            captured.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ReportContractError(
                f"report worker snapshot checksum is invalid: {name}"
            )
        after = os.fstat(descriptor)
        if after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns:
            raise ReportContractError(f"report worker input changed: {name}")
        captured.seek(0)
        return pd.read_parquet(captured)
    finally:
        os.close(descriptor)


def _write_output_handle(
    transfer: object, payload: bytes, *, name: str
) -> dict[str, object]:
    descriptor = _detach_descriptor(transfer, writable=True)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ReportContractError(f"report worker output is unsafe: {name}")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written : written + 1024 * 1024])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if final.st_size != len(payload):
            raise ReportContractError(f"report worker output size is invalid: {name}")
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    finally:
        os.close(descriptor)


def run_report_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """Decode and evaluate only inside the resource-monitored subprocess."""

    try:
        snapshot_sha256 = request.get("snapshot_sha256")
        input_handles = request.get("input_handles")
        output_handles = request.get("output_handles")
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
        if (
            not isinstance(input_handles, dict)
            or set(input_handles) != set(_INPUT_NAMES.values())
        ):
            raise ReportContractError("report worker input handle inventory is invalid")
        if (
            not isinstance(output_handles, dict)
            or set(output_handles) != set(_OUTPUT_ATTRIBUTES)
        ):
            raise ReportContractError("report worker output handle inventory is invalid")
        exact_name = _INPUT_NAMES["exact"]
        baseline_name = _INPUT_NAMES["baseline"]
        finite_name = _INPUT_NAMES["finite"]
        exact = _read_parquet_handle(
            input_handles[exact_name], snapshot_sha256[exact_name], name=exact_name
        )
        baselines = _read_parquet_handle(
            input_handles[baseline_name],
            snapshot_sha256[baseline_name],
            name=baseline_name,
        )
        finite = _read_parquet_handle(
            input_handles[finite_name], snapshot_sha256[finite_name], name=finite_name
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
            outputs[name] = _write_output_handle(
                output_handles[name], payload, name=name
            )
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


__all__ = [
    "close_descriptor_transfer",
    "duplicate_descriptor_for_worker",
    "run_report_worker",
]
