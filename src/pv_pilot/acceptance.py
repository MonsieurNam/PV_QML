"""Read-only verification for the checked-in synthetic acceptance run."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .artifacts import RunStore
from .config import LoadedPilotConfig, load_config
from .domain import DataProvenance

STAGE_ORDER = (
    "validate-data",
    "build-splits",
    "train-baselines",
    "build-photonic-features",
    "benchmark",
    "report",
)

_EXPECTED_STATUSES = {
    "validate-data": "SUCCESS",
    "build-splits": "SUCCESS",
    "train-baselines": "SUCCESS",
    "build-photonic-features": "SUCCESS",
    "benchmark": "SUCCESS",
    "report": "BLOCKED",
}

_RESOURCE_LEDGERS = {
    "validate-data": "validate-data/cleaning_ledger.json",
    "build-splits": "build-splits/resource_ledger.json",
    "train-baselines": "train-baselines/resource_ledger.json",
    "build-photonic-features": "build-photonic-features/resource_ledger.json",
    "benchmark": "benchmark/resource_ledger.json",
    "report": "report/resource_ledger.json",
}

_MAX_JSON_BYTES = 16 * 1024 * 1024


class AcceptanceError(ValueError):
    """The latest synthetic run does not satisfy the software acceptance contract."""


@dataclass(frozen=True, slots=True)
class AcceptanceSummary:
    """Small JSON-safe summary emitted by the CI acceptance verifier."""

    run_dir: str
    stage_order: tuple[str, ...]
    stage_statuses: dict[str, str]
    scientific_decision: str
    conclusion: str
    finite_shot: str
    resource_status: str
    execution: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _VerifiedStage:
    name: str
    status: str
    marker_sha256: str
    outputs: dict[str, str]
    allowlist: dict[str, tuple[str, ...]] | None


def _is_link_or_reparse(path: Path) -> bool:
    try:
        path_stat = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(path_stat, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    return path.is_symlink() or is_junction or bool(attributes & reparse_flag)


def _assert_safe_descendant(path: Path, *, root: Path, kind: str) -> Path:
    """Reject links/reparse points before resolving a path beneath ``root``."""

    absolute = path.absolute()
    absolute_root = root.absolute()
    try:
        relative = absolute.relative_to(absolute_root)
    except ValueError as error:
        raise AcceptanceError(f"{kind} escapes its configured artifact root") from error
    current = absolute_root
    for part in relative.parts:
        current /= part
        if _is_link_or_reparse(current):
            raise AcceptanceError(f"{kind} contains an unsafe link, junction, or reparse point")
    resolved_root = absolute_root.resolve()
    resolved = absolute.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise AcceptanceError(f"resolved {kind} escapes its configured artifact root")
    return resolved


def _safe_json_bytes(path: Path, *, root: Path) -> tuple[bytes, str]:
    resolved = _assert_safe_descendant(path, root=root, kind="artifact path")
    try:
        before = path.lstat()
    except OSError as error:
        raise AcceptanceError(f"required artifact cannot be inspected: {path}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > _MAX_JSON_BYTES
    ):
        raise AcceptanceError(f"required artifact is not a safe bounded regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_before.st_mode)
                or opened_before.st_nlink != 1
                or opened_before.st_size > _MAX_JSON_BYTES
            ):
                raise AcceptanceError(f"required artifact is unsafe: {path}")
            chunks: list[bytes] = []
            remaining = _MAX_JSON_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except OSError as error:
        raise AcceptanceError(f"cannot snapshot JSON artifact: {path}") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if len(payload) > _MAX_JSON_BYTES or any(
        getattr(opened_before, field) != getattr(opened_after, field)
        or getattr(opened_after, field) != getattr(after, field)
        or getattr(after, field) != getattr(before, field)
        for field in identity_fields
    ):
        raise AcceptanceError(f"JSON artifact changed while being snapshotted: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _decode_object(payload: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot decode JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise AcceptanceError(f"JSON artifact must contain an object: {path}")
    return value


def _snapshot_object(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    payload, digest = _safe_json_bytes(path, root=root)
    return _decode_object(payload, path=path), digest


def _raw_artifact_root(loaded: LoadedPilotConfig) -> Path:
    return (loaded.source_dir / loaded.config.paths.artifact_dir).absolute()


def _latest_run(loaded: LoadedPilotConfig) -> tuple[RunStore, Path, Path]:
    raw_artifact_root = _raw_artifact_root(loaded)
    resolved_artifact_root = _assert_safe_descendant(
        raw_artifact_root,
        root=loaded.project_root,
        kind="artifact root",
    )
    suffix = f"-{loaded.config_hash()[:8]}"
    matching = sorted(raw_artifact_root.glob(f"*{suffix}"), key=lambda path: path.name)
    if not matching:
        raise AcceptanceError(f"no run directories found under {raw_artifact_root}")
    candidate = matching[-1]
    resolved_candidate = _assert_safe_descendant(
        candidate,
        root=raw_artifact_root,
        kind="run candidate",
    )
    if resolved_candidate.parent != resolved_artifact_root or not resolved_candidate.is_dir():
        raise AcceptanceError("latest run candidate is not a direct safe directory")
    try:
        store = RunStore.open(candidate, expected_config_hash=loaded.config_hash())
    except (FileNotFoundError, ValueError) as error:
        raise AcceptanceError(f"latest run cannot be opened: {error}") from error
    if store.root != resolved_candidate:
        raise AcceptanceError("latest run identity changed while being opened")
    return store, candidate, raw_artifact_root


def _marker_allowlist(marker: dict[str, Any], stage: str) -> dict[str, tuple[str, ...]] | None:
    details = marker.get("details")
    if (
        not isinstance(details, dict)
        or details.get("prerequisite_verification") != "selected_outputs"
    ):
        return None
    raw = details.get("prerequisite_output_allowlist")
    if not isinstance(raw, dict) or not all(
        isinstance(key, str)
        and isinstance(paths, list)
        and all(isinstance(path, str) for path in paths)
        for key, paths in raw.items()
    ):
        raise AcceptanceError(f"{stage} has an invalid selected-output allowlist")
    return {key: tuple(cast(list[str], paths)) for key, paths in raw.items()}


def _verify_stage(store: RunStore, stage: str) -> _VerifiedStage:
    marker_path = store.root / stage / "stage_manifest.json"
    marker, _ = _snapshot_object(marker_path, root=store.root)
    expected_status = _EXPECTED_STATUSES[stage]
    if marker.get("stage") != stage or marker.get("status") != expected_status:
        raise AcceptanceError(
            f"{stage} expected {expected_status}, observed {marker.get('status')!r}"
        )
    allowlist = _marker_allowlist(marker, stage)
    try:
        authenticated_digest = store.verify_terminal_stage(
            stage,
            accepted_statuses=frozenset({expected_status}),
            prerequisite_output_allowlist=allowlist,
        )
    except ValueError as error:
        raise AcceptanceError(
            f"{stage} immutable inventory verification failed: {error}"
        ) from error
    authenticated_marker, marker_digest = _snapshot_object(marker_path, root=store.root)
    if marker_digest != authenticated_digest:
        raise AcceptanceError(f"{stage} marker changed after immutable verification")
    raw_outputs = authenticated_marker.get("outputs")
    if not isinstance(raw_outputs, list):
        raise AcceptanceError(f"{stage} has an invalid output inventory")
    outputs: dict[str, str] = {}
    for record in raw_outputs:
        if not isinstance(record, dict):
            raise AcceptanceError(f"{stage} has an invalid output inventory")
        relative = record.get("path")
        digest = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str) or relative in outputs:
            raise AcceptanceError(f"{stage} has an invalid output inventory")
        outputs[relative] = digest
    return _VerifiedStage(stage, expected_status, marker_digest, outputs, allowlist)


def _authenticated_object(
    store: RunStore, verified: _VerifiedStage, relative: str
) -> tuple[dict[str, Any], str]:
    expected = verified.outputs.get(relative)
    if not isinstance(expected, str):
        raise AcceptanceError(f"{relative} is absent from the authenticated output inventory")
    value, observed = _snapshot_object(store.root / relative, root=store.root)
    if observed != expected:
        raise AcceptanceError(f"authenticated artifact checksum changed: {relative}")
    return value, observed


def _after_semantic_snapshots() -> None:
    """Test seam before the final immutable-inventory re-verification."""


def _final_reverify(
    store: RunStore,
    stages: dict[str, _VerifiedStage],
    *,
    candidate: Path,
    artifact_root: Path,
) -> None:
    resolved_candidate = _assert_safe_descendant(
        candidate, root=artifact_root, kind="run candidate"
    )
    if resolved_candidate != store.root:
        raise AcceptanceError("run directory changed during acceptance verification")
    for stage in STAGE_ORDER:
        evidence = stages[stage]
        try:
            digest = store.verify_terminal_stage(
                stage,
                accepted_statuses=frozenset({evidence.status}),
                prerequisite_output_allowlist=evidence.allowlist,
            )
        except ValueError as error:
            raise AcceptanceError(
                f"{stage} final immutable inventory verification failed: {error}"
            ) from error
        if digest != evidence.marker_sha256:
            raise AcceptanceError(f"{stage} marker changed during acceptance verification")


def verify_acceptance_run(config_path: str | Path) -> AcceptanceSummary:
    """Verify the newest matching synthetic run without training or recomputation."""

    loaded = load_config(Path(config_path))
    if loaded.config.protocol.provenance is not DataProvenance.SYNTHETIC:
        raise AcceptanceError("acceptance verifier requires synthetic provenance")
    store, candidate, artifact_root = _latest_run(loaded)
    stages = {stage: _verify_stage(store, stage) for stage in STAGE_ORDER}
    statuses = {stage: evidence.status for stage, evidence in stages.items()}

    for stage, relative in _RESOURCE_LEDGERS.items():
        ledger, _ = _authenticated_object(store, stages[stage], relative)
        if ledger.get("resource_status") != "WITHIN_LIMIT":
            raise AcceptanceError(f"{stage} resource gate is not WITHIN_LIMIT")
        if ledger.get("cpu_only") is not True:
            raise AcceptanceError(f"{stage} did not declare CPU-only execution")

    decision, _ = _authenticated_object(
        store, stages["report"], "report/decision.json"
    )
    scores, _ = _authenticated_object(store, stages["report"], "report/scores.json")
    report_manifest, _ = _authenticated_object(
        store, stages["report"], "report/report_manifest.json"
    )
    data_card, data_card_sha256 = _authenticated_object(
        store, stages["validate-data"], "validate-data/data_card.json"
    )

    if (
        decision.get("stage_status") != "blocked"
        or decision.get("scientific_decision") != "not_evaluated"
        or decision.get("conclusion") != "not_evaluated"
        or decision.get("science_eligible") is not False
    ):
        raise AcceptanceError("synthetic report is not BLOCKED/not_evaluated and ineligible")
    if (
        scores.get("stage_status") != "blocked"
        or scores.get("decision") != "not_evaluated"
        or scores.get("conclusion") != "not_evaluated"
        or scores.get("science_eligible") is not False
    ):
        raise AcceptanceError("synthetic score evidence is not blocked and ineligible")
    if (
        data_card.get("provenance") != "synthetic"
        or data_card.get("synthetic") is not True
        or data_card.get("status") != "success"
    ):
        raise AcceptanceError("authenticated data card is not synthetic")
    canonical_sha256 = data_card.get("canonical_sha256")
    if not isinstance(canonical_sha256, str) or len(canonical_sha256) != 64:
        raise AcceptanceError("authenticated data card has no canonical checksum")

    report_provenance = report_manifest.get("provenance")
    artifact_inputs = report_manifest.get("artifact_input_hashes")
    if (
        report_manifest.get("data_provenance") != "synthetic"
        or report_manifest.get("scientific_decision") != "not_evaluated"
        or report_manifest.get("conclusion") != "not_evaluated"
        or report_manifest.get("protocol_approved_by") is not None
        or report_manifest.get("protocol_approved_at") is not None
        or not isinstance(report_provenance, dict)
        or report_provenance.get("run_id") != store.root.name
        or report_provenance.get("config_sha256") != store.config_hash
        or report_provenance.get("canonical_data_sha256") != canonical_sha256
        or not isinstance(artifact_inputs, dict)
        or artifact_inputs.get("validate-data/data_card.json") != data_card_sha256
    ):
        raise AcceptanceError("authenticated report provenance is not synthetic or bound")
    expected_stage_hashes = {
        stage: stages[stage].marker_sha256 for stage in STAGE_ORDER if stage != "report"
    }
    if report_provenance.get("stage_manifest_sha256") != expected_stage_hashes:
        raise AcceptanceError("report provenance is not bound to upstream stage markers")

    finite_shot = report_manifest.get("finite_shot")
    limitations = report_manifest.get("limitations")
    if not isinstance(finite_shot, dict) or finite_shot.get("ideal_shot_noise_only") is not True:
        raise AcceptanceError("finite-shot evidence is not marked ideal_shot_noise_only")
    if finite_shot.get("used_for_go_stop") is not False:
        raise AcceptanceError("finite-shot evidence was used for GO/STOP")
    if not isinstance(limitations, list) or "no_qpu_execution" not in limitations:
        raise AcceptanceError("report does not declare no QPU execution")

    _after_semantic_snapshots()
    _final_reverify(
        store,
        stages,
        candidate=candidate,
        artifact_root=artifact_root,
    )
    return AcceptanceSummary(
        run_dir=store.root.as_posix(),
        stage_order=STAGE_ORDER,
        stage_statuses=statuses,
        scientific_decision="not_evaluated",
        conclusion="not_evaluated",
        finite_shot="ideal_shot_noise_only",
        resource_status="WITHIN_LIMIT",
        execution="CPU/SLOS; no GPU/QPU",
    )


__all__ = ["STAGE_ORDER", "AcceptanceError", "AcceptanceSummary", "verify_acceptance_run"]
