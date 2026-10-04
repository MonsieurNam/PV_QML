"""Read-only verification for the checked-in synthetic acceptance run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .artifacts import RunStore
from .config import load_config

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


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AcceptanceError(f"required artifact is not a regular file: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise AcceptanceError(f"JSON artifact must contain an object: {path}")
    return value


def _latest_run(config_path: Path) -> RunStore:
    loaded = load_config(config_path)
    artifact_root = loaded.resolved_artifact_path
    suffix = f"-{loaded.config_hash()[:8]}"
    candidates = sorted(
        (
            path
            for path in artifact_root.glob(f"*{suffix}")
            if path.is_dir() and not path.is_symlink() and (path / "run_manifest.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise AcceptanceError(f"no run directories found under {artifact_root}")
    try:
        return RunStore.open(candidates[-1], expected_config_hash=loaded.config_hash())
    except (FileNotFoundError, ValueError) as error:
        raise AcceptanceError(f"latest run cannot be opened: {error}") from error


def _verify_stage(store: RunStore, stage: str) -> str:
    marker_path = store.root / stage / "stage_manifest.json"
    if not marker_path.is_file():
        raise AcceptanceError(f"{stage} has no terminal stage marker")
    marker = _read_object(marker_path)
    expected_status = _EXPECTED_STATUSES[stage]
    if marker.get("stage") != stage or marker.get("status") != expected_status:
        raise AcceptanceError(
            f"{stage} expected {expected_status}, observed {marker.get('status')!r}"
        )
    details = marker.get("details")
    allowlist: dict[str, list[str]] | None = None
    if isinstance(details, dict) and details.get("prerequisite_verification") == "selected_outputs":
        raw_allowlist = details.get("prerequisite_output_allowlist")
        if not isinstance(raw_allowlist, dict) or not all(
            isinstance(key, str)
            and isinstance(paths, list)
            and all(isinstance(path, str) for path in paths)
            for key, paths in raw_allowlist.items()
        ):
            raise AcceptanceError(f"{stage} has an invalid selected-output allowlist")
        allowlist = cast(dict[str, list[str]], raw_allowlist)
    try:
        store.verify_terminal_stage(
            stage,
            accepted_statuses=frozenset({expected_status}),
            prerequisite_output_allowlist=allowlist,
        )
    except ValueError as error:
        raise AcceptanceError(
            f"{stage} immutable inventory verification failed: {error}"
        ) from error
    return expected_status


def verify_acceptance_run(config_path: str | Path) -> AcceptanceSummary:
    """Verify the newest matching synthetic run without training or recomputation."""

    store = _latest_run(Path(config_path))
    statuses = {stage: _verify_stage(store, stage) for stage in STAGE_ORDER}
    for stage, relative in _RESOURCE_LEDGERS.items():
        ledger = _read_object(store.root / relative)
        if ledger.get("resource_status") != "WITHIN_LIMIT":
            raise AcceptanceError(f"{stage} resource gate is not WITHIN_LIMIT")
        if ledger.get("cpu_only") is not True:
            raise AcceptanceError(f"{stage} did not declare CPU-only execution")

    decision = _read_object(store.root / "report/decision.json")
    report_manifest = _read_object(store.root / "report/report_manifest.json")
    finite_shot = report_manifest.get("finite_shot")
    limitations = report_manifest.get("limitations")
    if (
        decision.get("stage_status") != "blocked"
        or decision.get("scientific_decision") != "not_evaluated"
        or decision.get("conclusion") != "not_evaluated"
    ):
        raise AcceptanceError("synthetic report is not BLOCKED/not_evaluated")
    if not isinstance(finite_shot, dict) or finite_shot.get("ideal_shot_noise_only") is not True:
        raise AcceptanceError("finite-shot evidence is not marked ideal_shot_noise_only")
    if finite_shot.get("used_for_go_stop") is not False:
        raise AcceptanceError("finite-shot evidence was used for GO/STOP")
    if not isinstance(limitations, list) or "no_qpu_execution" not in limitations:
        raise AcceptanceError("report does not declare no QPU execution")

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
