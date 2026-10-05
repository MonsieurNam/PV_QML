"""Immutable run artifacts and transactional stage contracts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Self
from uuid import uuid4

from .config import LoadedPilotConfig, PilotConfig
from .seeds import SeedManager
from .three_photon import DIAGNOSTIC_PATHS, PROFILE_FIXED_PATHS, PROFILE_PREFIX

SCHEMA_VERSION = "1.0"
Clock = Callable[[], datetime]
CommitGuard = Callable[[], None]
Now = datetime | Clock | None

STAGE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "validate-data": (),
    "build-splits": ("validate-data",),
    "train-baselines": ("build-splits",),
    "build-photonic-features": ("build-splits",),
    "benchmark": ("train-baselines", "build-photonic-features"),
    "report": ("benchmark",),
}

_BASE_ARTIFACTS = (
    "config.json",
    "seed_manifest.json",
    "environment_manifest.json",
    "source_manifest.json",
)
_PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "perceval-quandela",
    "exqalibur",
    "psutil",
    "pvlib",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "typer",
    "xgboost",
)
_PUBLICATION_JOURNAL = "ownership.json"
_PUBLICATION_JOURNAL_SCHEMA = "1"
_PUBLICATION_ABORT_PREFIX = ".publication-abort-"
_TERMINAL_STATUSES = frozenset({"SUCCESS", "BLOCKED", "INVALID"})
_TASK7_SELECTED_STAGE = "build-photonic-features"
_TASK7_SELECTED_PREREQUISITE = "build-splits"
_TASK7_SELECTED_OUTPUTS = (
    "build-splits/development_features.parquet",
    "build-splits/feature_manifest.json",
    "build-splits/minmax_scalers.json",
)
_BENCHMARK_SELECTED_STAGE = "benchmark"
_BENCHMARK_BASELINE_OUTPUTS = (
    "train-baselines/selection_manifest.json",
    "train-baselines/models/baseline_models.pkl",
    "train-baselines/models/manifest.json",
)
_BENCHMARK_PHOTONIC_REQUIRED_OUTPUTS = frozenset(
    {
        "build-photonic-features/development_photonic_features.parquet",
        "build-photonic-features/circuit_manifest.json",
        "build-photonic-features/feature_manifest.json",
        "build-photonic-features/state_spaces.json",
    }
)
_BENCHMARK_INTERNAL_INPUTS = frozenset(
    {
        "build-splits/development_features.parquet",
        "build-splits/feature_manifest.json",
        "build-splits/stage_manifest.json",
        ".benchmark-holdout-access/intent.json",
        ".benchmark-holdout-access/cache.parquet",
        ".benchmark-holdout-access/complete.json",
    }
)
_REPORT_SELECTED_STAGE = "report"
_REPORT_BENCHMARK_OUTPUTS = (
    "benchmark/selection_manifest.json",
    "benchmark/baseline_predictions.parquet",
    "benchmark/exact_predictions.parquet",
    "benchmark/finite_shot_predictions.parquet",
    "benchmark/diagnostics.json",
    "benchmark/models/manifest.json",
    "benchmark/resource_ledger.json",
    "benchmark/stage_result.json",
)
_REPORT_FIXED_INPUTS = frozenset(
    {
        "config.json",
        "environment_manifest.json",
        "source_manifest.json",
        "run_manifest.json",
        "validate-data/stage_manifest.json",
        "validate-data/data_card.json",
        "validate-data/cleaning_ledger.json",
        "validate-data/stage_result.json",
        "build-splits/stage_manifest.json",
        "build-splits/feature_manifest.json",
        "build-splits/minmax_scalers.json",
        "build-splits/clip_rates.json",
        "build-splits/resource_ledger.json",
        "build-splits/stage_result.json",
        "train-baselines/selection_manifest.json",
        "train-baselines/diagnostics.json",
        "train-baselines/models/baseline_models.pkl",
        "train-baselines/models/manifest.json",
        "train-baselines/resource_ledger.json",
        "train-baselines/stage_result.json",
        "train-baselines/stage_manifest.json",
        "build-photonic-features/stage_manifest.json",
        "build-photonic-features/development_photonic_features.parquet",
        "build-photonic-features/state_spaces.json",
        "build-photonic-features/circuit_manifest.json",
        "build-photonic-features/feature_manifest.json",
        "build-photonic-features/resource_ledger.json",
        "build-photonic-features/stage_result.json",
        "benchmark/models/photonic_models.pkl",
        "benchmark/stage_manifest.json",
        *_REPORT_BENCHMARK_OUTPUTS,
    }
)
_REPORT_TERMINAL_OUTPUTS = frozenset(
    {
        "report/report.md",
        "report/timestamp_scores.parquet",
        "report/evaluated_predictions.parquet",
        "report/bootstrap_replicates.parquet",
        "report/scores.json",
        "report/scores.csv",
        "report/decision.json",
        "report/report_manifest.json",
        "report/resource_ledger.json",
        "report/stage_result.json",
    }
)
_REPORT_INVALID_OUTPUTS = frozenset(
    {
        "report/report.md",
        "report/scores.json",
        "report/scores.csv",
        "report/decision.json",
        "report/report_manifest.json",
        "report/resource_ledger.json",
        "report/stage_result.json",
    }
)


def _validate_selected_output_shape(
    stage: str, normalized_allowlist: Mapping[str, Iterable[str]]
) -> None:
    if set(normalized_allowlist) != set(STAGE_PREREQUISITES[stage]):
        raise ValueError("prerequisite output allowlist must cover exact prerequisites")
    normalized = {
        prerequisite: tuple(sorted(paths)) for prerequisite, paths in normalized_allowlist.items()
    }
    if stage == _TASK7_SELECTED_STAGE:
        selected = normalized[_TASK7_SELECTED_PREREQUISITE]
        if selected != tuple(sorted(_TASK7_SELECTED_OUTPUTS)):
            raise ValueError("Task 7 selected_outputs allowlist must match the fixed contract")
        return
    if stage == _REPORT_SELECTED_STAGE:
        if normalized["benchmark"] != tuple(sorted(_REPORT_BENCHMARK_OUTPUTS)):
            raise ValueError("report benchmark allowlist must match the fixed contract")
        return
    if stage != _BENCHMARK_SELECTED_STAGE:
        raise ValueError("selected_outputs is not supported for this stage")
    if normalized["train-baselines"] != tuple(sorted(_BENCHMARK_BASELINE_OUTPUTS)):
        raise ValueError("benchmark baseline allowlist must match the fixed contract")
    photonic = set(normalized["build-photonic-features"])
    if not photonic >= _BENCHMARK_PHOTONIC_REQUIRED_OUTPUTS:
        raise ValueError("benchmark photonic allowlist is missing required outputs")
    optional_fixed = photonic & set(PROFILE_FIXED_PATHS)
    if optional_fixed and optional_fixed != set(PROFILE_FIXED_PATHS):
        raise ValueError("benchmark optional profile allowlist is incomplete")
    circuit_paths = photonic - _BENCHMARK_PHOTONIC_REQUIRED_OUTPUTS - optional_fixed
    if not circuit_paths or any(
        not (
            path.startswith("build-photonic-features/circuits/")
            or path.startswith(f"{PROFILE_PREFIX}/circuits/")
        )
        or not path.endswith(".npz")
        for path in circuit_paths
    ):
        raise ValueError("benchmark photonic allowlist has invalid circuit outputs")


def _normalize_terminal_status(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in _TERMINAL_STATUSES:
        raise ValueError("terminal result has an invalid status")
    return value.upper()


@dataclass(frozen=True, slots=True)
class PublicationOwnership:
    """Identity-bound permission to remove one published artifact."""

    destination: Path
    source_device: int
    source_inode: int
    quarantine_root: Path
    record: dict[str, object] | None = None

    @property
    def quarantine(self) -> Path:
        return self.quarantine_root / "entry"

    def prepare(self) -> None:
        self.quarantine_root.mkdir()
        if self.record is None:
            return
        temporary = self.quarantine_root / ".ownership.tmp"
        with temporary.open("xb") as stream:
            stream.write(_canonical_json(self.record).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.quarantine_root / _PUBLICATION_JOURNAL)

    def release(self) -> None:
        (self.quarantine_root / ".ownership.tmp").unlink(missing_ok=True)
        (self.quarantine_root / _PUBLICATION_JOURNAL).unlink(missing_ok=True)
        self.quarantine_root.rmdir()

    def _is_owned(self, path: Path) -> bool:
        path_stat = path.stat(follow_symlinks=False)
        return path_stat.st_dev == self.source_device and path_stat.st_ino == self.source_inode

    def _restore_foreign(self, quarantine: Path) -> None:
        if os.name == "nt":
            os.rename(quarantine, self.destination)
            return
        os.link(quarantine, self.destination, follow_symlinks=False)
        raise OSError(
            f"foreign publication was restored but retained in quarantine at {quarantine}"
        )

    def _settle_quarantine(self, quarantine: Path) -> bool:
        try:
            owned = self._is_owned(quarantine)
        except FileNotFoundError:
            return False
        if owned:
            quarantine.unlink()
        else:
            self._restore_foreign(quarantine)
        return True

    def rollback(self) -> None:
        """Atomically acquire the destination before deciding whether to delete it."""

        quarantine = self.quarantine
        self.quarantine_root.mkdir(exist_ok=True)
        if self._settle_quarantine(quarantine):
            self.release()
            return
        try:
            os.rename(self.destination, quarantine)
        except FileNotFoundError:
            self.release()
            return
        self._settle_quarantine(quarantine)
        self.release()


class PublicationBatch(list[PublicationOwnership]):
    """One stage publication group sharing a durable commit marker."""

    def __init__(
        self,
        *,
        transaction_id: str,
        stage: str,
        marker_relative: str,
        terminal_status: Literal["SUCCESS", "BLOCKED", "INVALID"],
        output_hashes: Mapping[str, str],
    ) -> None:
        super().__init__()
        self.transaction_id = transaction_id
        self.stage = stage
        self.marker_relative = marker_relative
        self.terminal_status = terminal_status
        self.output_hashes = dict(output_hashes)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
    checkpoint: Callable[[int], None] | None = None,
    live_bytes: int = 0,
    root: Path | None = None,
) -> bytes:
    """Capture one descriptor and authenticate exactly the bytes passed to parsers."""

    boundary = root.resolve() if root is not None else Path(path.anchor)
    if not path.resolve(strict=False).is_relative_to(boundary):
        raise ValueError(f"snapshot input is unsafe: {path}")
    for candidate in (path, *path.parents):
        info = candidate.lstat()
        if candidate.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"snapshot input is unsafe: {path}")
        if candidate == boundary:
            break
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"snapshot input is not a regular single-link file: {path}")
    enforce = checkpoint or (lambda _size: None)
    enforce(live_bytes + 2 * before.st_size)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError(f"snapshot input changed while opening: {path}")
        captured = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            captured += len(chunk)
            enforce(live_bytes + 2 * captured)
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise ValueError(f"snapshot input changed while captured: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"snapshot captured checksum mismatch: {path}")
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _clock(now: Now) -> Clock:
    if now is None:
        return lambda: datetime.now(UTC)
    if isinstance(now, datetime):
        return lambda: now
    return now


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("artifact timestamps must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_stamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("run timestamps must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_fingerprint(project_root: Path) -> tuple[str, int, str]:
    """Hash executable source, including new modules, without research/documentation noise."""

    project_root = project_root.resolve()
    paths = [
        path
        for directory in ("src", "scripts")
        for path in (project_root / directory).rglob("*")
        if path.is_file()
    ]
    paths.extend(
        path for name in ("pyproject.toml", "uv.lock") if (path := project_root / name).is_file()
    )
    records: list[dict[str, str]] = []
    for raw_path in sorted(paths):
        relative = raw_path.relative_to(project_root).as_posix()
        if "__pycache__" in raw_path.parts or raw_path.suffix in {".pyc", ".pyo"}:
            continue
        resolved = raw_path.resolve(strict=False)
        if not resolved.is_relative_to(project_root) or raw_path.is_symlink():
            raise ValueError(f"execution provenance source path is unsafe: {relative}")
        digest = sha256_file(raw_path)
        records.append({"path": relative.replace("\\", "/"), "sha256": digest})
    return _sha256_canonical(records), len(records), "source_inventory"


def _runtime_environment_payload() -> dict[str, object]:
    return {
        "packages": _package_versions(),
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _execution_provenance(
    project_root: Path, environment: Mapping[str, object] | None = None
) -> dict[str, object]:
    runtime_path = Path(__file__).resolve()
    if not runtime_path.is_file():
        raise ValueError("execution provenance executing source is unavailable")
    source_sha256, source_file_count, source_method = _source_fingerprint(project_root)
    lock_path = project_root / "uv.lock"
    lock_sha256 = sha256_file(lock_path) if lock_path.is_file() else None
    environment_sha256 = _sha256_canonical(
        dict(environment) if environment is not None else _runtime_environment_payload()
    )
    identity = {
        "source_tree_sha256": source_sha256,
        "source_method": source_method,
        "source_file_count": source_file_count,
        "uv_lock_sha256": lock_sha256,
        "environment_sha256": environment_sha256,
        "runtime_source_sha256": _source_fingerprint(runtime_path.parents[2])[0],
    }
    return {**identity, "aggregate_sha256": _sha256_canonical(identity)}


def _git_manifest(project_root: Path) -> dict[str, str | bool | None]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    try:
        head_result = git("rev-parse", "HEAD")
        status_result = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "tracked_dirty": None}
    if head_result.returncode != 0 or status_result.returncode != 0:
        return {"head": None, "tracked_dirty": None}
    return {
        "head": head_result.stdout.strip(),
        "tracked_dirty": bool(status_result.stdout.strip()),
    }


class RunStore:
    """A verified immutable run directory with no-overwrite artifact writes."""

    def __init__(self, root: Path, config_hash: str) -> None:
        self.root = root.resolve()
        self.config_hash = config_hash

    @classmethod
    def create(cls, loaded: LoadedPilotConfig, *, now: Now = None) -> Self:
        """Create a uniquely named run and publish the init marker last."""

        created_at = _clock(now)()
        config_hash = loaded.config_hash()
        artifact_root = loaded.resolved_artifact_path
        artifact_root.mkdir(parents=True, exist_ok=True)
        run_root = artifact_root / f"{_run_stamp(created_at)}-{config_hash[:8]}"
        run_root.mkdir(exist_ok=False)
        store = cls(run_root, config_hash)
        try:
            store.write_text("config.json", loaded.canonical_json())
            seed_manifest = {
                **SeedManager(loaded.config.seeds.master_seed).manifest,
                "config_hash": config_hash,
            }
            store.write_json("seed_manifest.json", seed_manifest)
            runtime_environment = _runtime_environment_payload()
            environment_manifest = {
                "config_hash": config_hash,
                "config_source": {
                    "path": loaded.config_file.as_posix(),
                    "sha256": sha256_file(loaded.config_file),
                },
                "execution_provenance": _execution_provenance(
                    loaded.project_root, runtime_environment
                ),
                "git": _git_manifest(loaded.project_root),
                "packages": runtime_environment["packages"],
                "platform": runtime_environment["platform"],
                "project_root": loaded.project_root.as_posix(),
                "python": runtime_environment["python"],
                "schema_version": SCHEMA_VERSION,
            }
            store.write_json("environment_manifest.json", environment_manifest)
            input_path = loaded.resolved_input_path
            is_file = input_path.is_file()
            input_record: dict[str, object] = {
                "path": input_path.as_posix(),
                "read_only": True,
                "sha256": sha256_file(input_path) if is_file else None,
                "status": "AVAILABLE" if is_file else "MISSING",
            }
            source_manifest = {
                "config_hash": config_hash,
                "input": input_record,
                "readiness": "READY" if is_file else "BLOCKED",
                "schema_version": SCHEMA_VERSION,
                "scientific_decision": "NOT_EVALUATED",
            }
            store.write_json("source_manifest.json", source_manifest)
            outputs = [
                {"path": path, "sha256": sha256_file(store.root / path)} for path in _BASE_ARTIFACTS
            ]
            init_manifest = {
                "config_hash": config_hash,
                "details": {"source_readiness": source_manifest["readiness"]},
                "ended_at": _timestamp(created_at),
                "input_hashes": {},
                "outputs": outputs,
                "prerequisite_manifest_hashes": {},
                "schema_version": SCHEMA_VERSION,
                "stage": "init",
                "started_at": _timestamp(created_at),
                "status": "SUCCESS",
            }
            store.write_json("run_manifest.json", init_manifest)
        except BaseException:
            shutil.rmtree(run_root, ignore_errors=True)
            raise
        return store

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        *,
        expected_config_hash: str | None = None,
        verify_execution: bool = True,
    ) -> Self:
        """Open a run only after validating its init contract and base artifacts."""

        root = Path(run_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {root}")
        manifest = cls._read_json_file(root / "run_manifest.json")
        config_hash = manifest.get("config_hash")
        if not isinstance(config_hash, str):
            raise ValueError("run manifest has no valid config hash")
        store = cls(root, config_hash)
        with store._publication_lock():
            store._recover_publication_journals()
            store._cleanup_stale_stage_directories()
        store._verify_base(expected_config_hash=expected_config_hash)
        if verify_execution:
            store.verify_execution_provenance()
        return store

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON artifact: {path.name}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact must be an object: {path.name}")
        return payload

    def _validated_target(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or candidate.drive:
            raise ValueError("artifact paths must be relative and inside the run directory")
        if not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("artifact path cannot escape the run directory")
        resolved = (self.root / candidate).resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact path would escape the run directory")
        return resolved

    def _validate_selected_output_policy(
        self, stage: str, normalized_allowlist: Mapping[str, Iterable[str]]
    ) -> None:
        _validate_selected_output_shape(stage, normalized_allowlist)
        if stage != _BENCHMARK_SELECTED_STAGE:
            return
        marker = self._read_json_file(self.root / "build-photonic-features/stage_manifest.json")
        outputs = marker.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("benchmark photonic prerequisite marker has invalid outputs")
        declared_circuits = {
            str(record.get("path"))
            for record in outputs
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and str(record.get("path")).startswith("build-photonic-features/circuits/")
            and str(record.get("path")).endswith(".npz")
        }
        frozen_config = self._read_json_file(self.root / "config.json")
        photonics = frozen_config.get("photonics", {})
        optional_paths = {
            str(record.get("path"))
            for record in outputs
            if isinstance(record, dict)
            and (
                str(record.get("path")) in PROFILE_FIXED_PATHS
                or (
                    str(record.get("path")).startswith(f"{PROFILE_PREFIX}/circuits/")
                    and str(record.get("path")).endswith(".npz")
                )
            )
        }
        if photonics.get("enable_three_photon_profile"):
            if not set(PROFILE_FIXED_PATHS) <= optional_paths:
                raise ValueError("enabled optional profile inventory is incomplete")
        elif optional_paths:
            raise ValueError("disabled optional profile has unexpected outputs")
        expected_photonic = (
            _BENCHMARK_PHOTONIC_REQUIRED_OUTPUTS | declared_circuits | optional_paths
        )
        actual_photonic = set(normalized_allowlist["build-photonic-features"])
        if not declared_circuits or actual_photonic != expected_photonic:
            raise ValueError(
                "benchmark photonic allowlist must equal the immutable declared circuit set"
            )

    @staticmethod
    def _validate_lock_identity(path: Path, descriptor: int) -> None:
        path_stat = path.lstat()
        descriptor_stat = os.fstat(descriptor)
        unsafe = (
            path.is_symlink()
            or not stat.S_ISREG(path_stat.st_mode)
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or path_stat.st_nlink != 1
            or descriptor_stat.st_nlink != 1
            or (path_stat.st_dev, path_stat.st_ino)
            != (descriptor_stat.st_dev, descriptor_stat.st_ino)
        )
        getuid = getattr(os, "getuid", None)
        if getuid is not None:
            unsafe = unsafe or path_stat.st_uid != getuid()
        if unsafe:
            raise OSError(f"unsafe lock file: {path}")

    @classmethod
    def _open_safe_lock(cls, path: Path) -> BinaryIO:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        created = False
        try:
            descriptor = os.open(path, flags, 0o600)
            created = True
        except FileExistsError:
            existing_flags = os.O_RDWR
            existing_flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, existing_flags)
        try:
            cls._validate_lock_identity(path, descriptor)
            stream = os.fdopen(descriptor, "r+b")
            descriptor = -1
            if created:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
            return stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _lock_stream(stream: BinaryIO, *, blocking: bool) -> bool:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(stream.fileno(), mode, 1)
            except OSError:
                if not blocking:
                    return False
                raise
        else:
            fcntl: Any = __import__("fcntl")
            mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(stream.fileno(), mode)
            except BlockingIOError:
                return False
        return True

    @staticmethod
    def _unlock_stream(stream: BinaryIO) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl: Any = __import__("fcntl")
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _publication_lock(self) -> Iterator[None]:
        lock_path = self.root / ".publication.lock"
        with self._open_safe_lock(lock_path) as stream:
            self._lock_stream(stream, blocking=True)
            try:
                yield
            finally:
                self._unlock_stream(stream)

    def _publication_record(self, ownership: PublicationOwnership) -> dict[str, object]:
        record = ownership.record
        if record is None:
            raise ValueError("publication journal has no ownership record")
        return record

    def _abort_path(self, transaction_id: str) -> Path:
        if len(transaction_id) != 32 or any(
            character not in "0123456789abcdef" for character in transaction_id
        ):
            raise ValueError("publication transaction id is invalid")
        return self.root / f"{_PUBLICATION_ABORT_PREFIX}{transaction_id}"

    def _mark_batch_aborted(self, publications: PublicationBatch) -> Path:
        path = self._abort_path(publications.transaction_id)
        try:
            with path.open("xb") as stream:
                stream.write(publications.transaction_id.encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
        return path

    def _clear_batch_abort(self, transaction_id: str) -> None:
        self._abort_path(transaction_id).unlink(missing_ok=True)

    def _journal_group_committed(self, publications: list[PublicationOwnership]) -> bool:
        first = self._publication_record(publications[0])
        marker_relative = first["marker"]
        if not isinstance(marker_relative, str):
            raise ValueError("publication journal marker is invalid")
        marker = self._validated_target(marker_relative)
        if not marker.exists() and not marker.is_symlink():
            return False
        marker_publications = [
            publication
            for publication in publications
            if self._publication_record(publication).get("kind") == "marker"
        ]
        if len(marker_publications) != 1 or marker.is_symlink() or not marker.is_file():
            raise ValueError("ambiguous publication success marker")
        marker_publication = marker_publications[0]
        if not marker_publication._is_owned(marker):
            raise ValueError("publication success marker has foreign identity")
        manifest = self._read_json_file(marker)
        transaction_id = first["transaction_id"]
        stage = first["stage"]
        terminal_status = first["terminal_status"]
        if terminal_status not in {"SUCCESS", "BLOCKED", "INVALID"}:
            raise ValueError("publication journal terminal status is invalid")
        common_fields = (
            "config_hash",
            "marker",
            "run_id",
            "schema",
            "stage",
            "terminal_status",
            "transaction_id",
        )
        if any(
            any(
                self._publication_record(publication).get(field) != first.get(field)
                for field in common_fields
            )
            for publication in publications
        ):
            raise ValueError("publication journal group metadata is inconsistent")
        if (
            manifest.get("transaction_id") != transaction_id
            or manifest.get("stage") != stage
            or manifest.get("config_hash") != self.config_hash
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != terminal_status
        ):
            raise ValueError("publication terminal marker does not match journal")
        expected_outputs = {
            self._publication_record(publication)["destination"]: self._publication_record(
                publication
            )["sha256"]
            for publication in publications
            if self._publication_record(publication).get("kind") == "output"
        }
        manifest_outputs = manifest.get("outputs")
        if not isinstance(manifest_outputs, list):
            raise ValueError("publication success marker outputs are invalid")
        actual_outputs = {
            item.get("path"): item.get("sha256")
            for item in manifest_outputs
            if isinstance(item, dict)
        }
        if any(actual_outputs.get(path) != digest for path, digest in expected_outputs.items()):
            raise ValueError("publication success marker outputs do not match journal")
        for relative_path, expected_digest in expected_outputs.items():
            if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
                raise ValueError("publication output checksum record is invalid")
            output = self._validated_target(relative_path)
            if not output.is_file() or sha256_file(output) != expected_digest:
                raise ValueError(f"publication output checksum mismatch: {relative_path}")
        self._verify_terminal_result_contract(manifest)
        return True

    def _recover_publication_journals(self) -> None:
        abort_paths = sorted(self.root.glob(f"{_PUBLICATION_ABORT_PREFIX}*"))
        aborted_transactions: set[str] = set()
        for abort_path in abort_paths:
            transaction_id = abort_path.name.removeprefix(_PUBLICATION_ABORT_PREFIX)
            if abort_path.is_symlink() or not abort_path.is_file():
                raise ValueError(f"unsafe publication abort record: {abort_path.name}")
            self._abort_path(transaction_id)
            aborted_transactions.add(transaction_id)
        journal_roots = sorted(self.root.rglob(".rollback-*"), key=lambda path: path.as_posix())
        if len(journal_roots) > 1024:
            raise ValueError("too many publication journals")
        publications: list[PublicationOwnership] = []
        incomplete: list[Path] = []
        for journal_root in journal_roots:
            is_junction = getattr(journal_root, "is_junction", lambda: False)()
            if (
                journal_root.is_symlink()
                or is_junction
                or not journal_root.is_dir()
                or not journal_root.resolve().is_relative_to(self.root)
            ):
                raise ValueError(f"unsafe publication journal: {journal_root.name}")
            record_path = journal_root / _PUBLICATION_JOURNAL
            if not record_path.is_file() or record_path.is_symlink():
                entries = {path.name for path in journal_root.iterdir()}
                if entries <= {".ownership.tmp"}:
                    incomplete.append(journal_root)
                    continue
                raise ValueError(f"ambiguous publication journal: {journal_root.name}")
            record = self._read_json_file(record_path)
            required_strings = (
                "config_hash",
                "destination",
                "kind",
                "marker",
                "run_id",
                "schema",
                "sha256",
                "stage",
                "terminal_status",
                "transaction_id",
            )
            if any(not isinstance(record.get(key), str) for key in required_strings):
                raise ValueError(f"invalid publication journal: {journal_root.name}")
            if (
                record["schema"] != _PUBLICATION_JOURNAL_SCHEMA
                or record["config_hash"] != self.config_hash
                or record["run_id"] != self.root.name
                or record["kind"] not in {"output", "marker"}
                or record["terminal_status"] not in {"SUCCESS", "BLOCKED", "INVALID"}
                or not isinstance(record.get("source_device"), int)
                or not isinstance(record.get("source_inode"), int)
            ):
                raise ValueError(f"invalid publication journal: {journal_root.name}")
            self._abort_path(str(record["transaction_id"]))
            destination = self._validated_target(str(record["destination"]))
            self._validated_target(str(record["marker"]))
            publications.append(
                PublicationOwnership(
                    destination=destination,
                    source_device=int(record["source_device"]),
                    source_inode=int(record["source_inode"]),
                    quarantine_root=journal_root,
                    record=record,
                )
            )
        destinations = [publication.destination for publication in publications]
        if len(destinations) != len(set(destinations)):
            raise ValueError("conflicting publication journals")
        groups: dict[str, list[PublicationOwnership]] = {}
        for publication in publications:
            record_transaction_id = self._publication_record(publication)["transaction_id"]
            assert isinstance(record_transaction_id, str)
            groups.setdefault(record_transaction_id, []).append(publication)
        decisions = {
            transaction_id: transaction_id not in aborted_transactions
            and self._journal_group_committed(group)
            for transaction_id, group in groups.items()
        }
        for journal_root in incomplete:
            (journal_root / ".ownership.tmp").unlink(missing_ok=True)
            journal_root.rmdir()
        recovery_error = RuntimeError("recovering interrupted artifact publication")
        recovery_failures: list[BaseException] = []
        for transaction_id, group in groups.items():
            if decisions[transaction_id]:
                self._release_publications(group)
            else:
                recovery_failures.extend(self._rollback_publications(group, recovery_error))
            if not any(publication.quarantine_root.exists() for publication in group):
                self._clear_batch_abort(transaction_id)
        for transaction_id in aborted_transactions - groups.keys():
            self._clear_batch_abort(transaction_id)
        if recovery_failures:
            for failure in recovery_failures:
                recovery_error.add_note(f"publication recovery failed: {failure}")
            raise recovery_error

    def _cleanup_stale_stage_directories(self) -> None:
        for stage_root in sorted(self.root.glob(".stage-*")):
            is_junction = getattr(stage_root, "is_junction", lambda: False)()
            if stage_root.is_symlink() or is_junction or not stage_root.is_dir():
                raise ValueError(f"unsafe stage lease directory: {stage_root.name}")
            lease_path = stage_root / ".lease"
            if not lease_path.is_file() or lease_path.is_symlink():
                raise ValueError(f"invalid stage lease: {stage_root.name}")
            stale = False
            with self._open_safe_lock(lease_path) as lease:
                if not self._lock_stream(lease, blocking=False):
                    continue
                self._unlock_stream(lease)
                stale = True
            if stale:
                shutil.rmtree(stage_root)

    @staticmethod
    def _rollback_publications(
        publications: Iterable[PublicationOwnership], error: BaseException
    ) -> tuple[BaseException, ...]:
        cleanup_errors: list[BaseException] = []
        for publication in reversed(tuple(publications)):
            try:
                publication.rollback()
            except BaseException as cleanup_error:
                try:
                    publication.rollback()
                except BaseException as retry_error:
                    cleanup_error.add_note(f"publication cleanup retry failed: {retry_error}")
                    cleanup_errors.append(cleanup_error)
        for recorded_cleanup_error in cleanup_errors:
            error.add_note(f"publication cleanup failed: {recorded_cleanup_error}")
        return tuple(cleanup_errors)

    @staticmethod
    def _release_publications(
        publications: Iterable[PublicationOwnership],
    ) -> None:
        ordered = sorted(
            publications,
            key=lambda publication: (
                publication.record is not None and publication.record.get("kind") == "marker"
            ),
        )
        for publication in ordered:
            publication.release()

    def _publish_temp(
        self,
        source: Path,
        destination: Path,
        publications: list[PublicationOwnership] | None = None,
    ) -> None:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"artifact already exists: {destination}")
        source_stat = source.stat(follow_symlinks=False)
        record: dict[str, object] | None = None
        if isinstance(publications, PublicationBatch):
            relative = destination.relative_to(self.root).as_posix()
            kind = "marker" if relative == publications.marker_relative else "output"
            record = {
                "config_hash": self.config_hash,
                "destination": relative,
                "kind": kind,
                "marker": publications.marker_relative,
                "run_id": self.root.name,
                "schema": _PUBLICATION_JOURNAL_SCHEMA,
                "sha256": sha256_file(source),
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "stage": publications.stage,
                "terminal_status": publications.terminal_status,
                "transaction_id": publications.transaction_id,
            }
        ownership = PublicationOwnership(
            destination=destination,
            source_device=source_stat.st_dev,
            source_inode=source_stat.st_ino,
            quarantine_root=destination.parent / f".rollback-{destination.name}-{uuid4().hex}",
            record=record,
        )
        local_publication = publications is None
        if publications is None:
            publications = []
        publications.append(ownership)
        try:
            ownership.prepare()
            os.link(source, destination)
            source.unlink()
            if local_publication:
                ownership.release()
        except BaseException as error:
            self._rollback_publications((ownership,), error)
            try:
                source.unlink(missing_ok=True)
            except OSError as source_cleanup_error:
                error.add_note(f"publication cleanup failed: {source_cleanup_error}")
            raise

    def _write_bytes(
        self,
        relative_path: str | Path,
        data: bytes,
        publications: list[PublicationOwnership] | None = None,
    ) -> Path:
        destination = self._validated_target(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"artifact already exists: {relative_path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".write-", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if publications is None:
                self._publish_temp(temporary, destination)
            else:
                self._publish_temp(temporary, destination, publications)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def write_bytes(self, relative_path: str | Path, data: bytes) -> Path:
        """Atomically publish bytes inside the run without overwriting a file."""

        return self._write_bytes(relative_path, data)

    def write_text(
        self,
        relative_path: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        return self.write_bytes(relative_path, text.encode(encoding))

    def write_json(self, relative_path: str | Path, value: object) -> Path:
        return self.write_text(relative_path, _canonical_json(value))

    def checksum(self, relative_path: str | Path) -> str:
        target = self._validated_target(relative_path)
        if not target.is_file():
            raise FileNotFoundError(f"artifact is not a file: {relative_path}")
        return sha256_file(target)

    def _verify_base(self, *, expected_config_hash: str | None = None) -> None:
        manifest = self._read_json_file(self.root / "run_manifest.json")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != "init"
            or manifest.get("status") != "SUCCESS"
        ):
            raise ValueError("run init manifest is not a schema 1.0 SUCCESS marker")
        if manifest.get("config_hash") != self.config_hash:
            raise ValueError("run config hash mismatch")
        if expected_config_hash is not None and expected_config_hash != self.config_hash:
            raise ValueError("expected config hash does not match run config hash")
        if sha256_file(self.root / "config.json") != self.config_hash:
            raise ValueError("config snapshot checksum does not match run config hash")
        raw_outputs = manifest.get("outputs")
        if not isinstance(raw_outputs, list):
            raise ValueError("run init manifest outputs are invalid")
        outputs = {
            record.get("path"): record.get("sha256")
            for record in raw_outputs
            if isinstance(record, dict)
        }
        if set(outputs) != set(_BASE_ARTIFACTS):
            raise ValueError("run init manifest does not list the exact base artifacts")
        for relative_path, expected in outputs.items():
            if not isinstance(relative_path, str) or not isinstance(expected, str):
                raise ValueError("run init manifest contains an invalid checksum record")
            target = self._validated_target(relative_path)
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f"base artifact checksum mismatch: {relative_path}")
        for name in ("seed_manifest.json", "environment_manifest.json", "source_manifest.json"):
            base_manifest = self._read_json_file(self.root / name)
            if (
                base_manifest.get("schema_version") != SCHEMA_VERSION
                or base_manifest.get("config_hash") != self.config_hash
            ):
                raise ValueError(f"base manifest config hash mismatch: {name}")

    def verify_execution_provenance(self) -> None:
        """Require the same executing source, project inputs, and installed environment."""

        environment_manifest = json.loads(self.read_authenticated("environment_manifest.json"))
        project_root_raw = environment_manifest.get("project_root")
        expected_provenance = environment_manifest.get("execution_provenance")
        if not isinstance(project_root_raw, str) or not isinstance(expected_provenance, dict):
            raise ValueError("execution provenance manifest is missing")
        project_root = Path(project_root_raw).resolve()
        if not project_root.is_dir():
            raise ValueError("execution provenance project root is unavailable")
        if _execution_provenance(project_root) != expected_provenance:
            raise ValueError("execution provenance drift detected")
        config_source = environment_manifest.get("config_source")
        if not isinstance(config_source, dict) or not isinstance(config_source.get("path"), str):
            raise ValueError("execution provenance config source is missing")
        config_path = Path(config_source["path"])
        if not config_path.is_file() or sha256_file(config_path) != config_source.get("sha256"):
            raise ValueError("execution provenance configuration drift detected")

    def read_authenticated(self, relative_path: str) -> bytes:
        """Return base artifact bytes authenticated against the immutable init marker."""

        manifest = self._read_json_file(self.root / "run_manifest.json")
        outputs = {record["path"]: record["sha256"] for record in manifest["outputs"]}
        if relative_path not in _BASE_ARTIFACTS or relative_path not in outputs:
            raise ValueError("authenticated base artifact is not declared")
        return authenticated_snapshot(
            self.root / relative_path, expected_sha256=outputs[relative_path], root=self.root
        )

    def _verify_stage(
        self,
        stage: str,
        *,
        verified: set[str] | None = None,
        accepted_statuses: frozenset[str] = frozenset({"SUCCESS"}),
    ) -> str:
        if verified is None:
            verified = set()
        if stage in verified:
            return sha256_file(self.root / stage / "stage_manifest.json")
        marker_path = self.root / stage / "stage_manifest.json"
        if (
            (
                accepted_statuses == frozenset({"SUCCESS"})
                or (stage == _REPORT_SELECTED_STAGE and accepted_statuses == frozenset({"BLOCKED"}))
            )
            and stage
            in {
                _TASK7_SELECTED_STAGE,
                _BENCHMARK_SELECTED_STAGE,
                _REPORT_SELECTED_STAGE,
            }
            and marker_path.is_file()
            and not marker_path.is_symlink()
        ):
            marker = self._read_json_file(marker_path)
            details = marker.get("details")
            if (
                isinstance(details, dict)
                and details.get("prerequisite_verification") == "selected_outputs"
            ):
                raw_allowlist = details.get("prerequisite_output_allowlist")
                if not isinstance(raw_allowlist, dict) or any(
                    not isinstance(prerequisite, str)
                    or not isinstance(paths, list)
                    or any(not isinstance(path, str) for path in paths)
                    for prerequisite, paths in raw_allowlist.items()
                ):
                    raise ValueError(
                        f"stage {stage!r} has an invalid selective verification policy"
                    )
                return self.verify_terminal_stage(
                    stage,
                    prerequisite_output_allowlist={
                        prerequisite: tuple(paths) for prerequisite, paths in raw_allowlist.items()
                    },
                )
        prerequisite_hashes: dict[str, str] = {}
        for prerequisite in STAGE_PREREQUISITES[stage]:
            prerequisite_hashes[prerequisite] = self._verify_stage(prerequisite, verified=verified)
        path = self.root / stage / "stage_manifest.json"
        if not path.is_file():
            raise ValueError(f"prerequisite stage {stage!r} has no SUCCESS marker")
        manifest = self._read_json_file(path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("status") not in accepted_statuses
            or manifest.get("config_hash") != self.config_hash
        ):
            expected_statuses = ", ".join(sorted(accepted_statuses))
            raise ValueError(
                f"stage {stage!r} has an invalid marker; expected status in {{{expected_statuses}}}"
            )
        if manifest.get("prerequisite_manifest_hashes") != prerequisite_hashes:
            raise ValueError(f"prerequisite manifest hash mismatch for stage {stage!r}")
        input_hashes = manifest.get("input_hashes")
        if not isinstance(input_hashes, dict) or any(
            not isinstance(relative_path, str) or not isinstance(expected, str)
            for relative_path, expected in input_hashes.items()
        ):
            raise ValueError(f"prerequisite stage {stage!r} has invalid input hashes")
        for relative_path, expected in input_hashes.items():
            target = self._validated_target(relative_path)
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f"prerequisite input checksum mismatch: {relative_path}")
        external_inputs = manifest.get("external_input_hashes")
        if not isinstance(external_inputs, list):
            raise ValueError(f"stage {stage!r} has invalid external input hashes")
        for record in external_inputs:
            if not isinstance(record, dict):
                raise ValueError(f"stage {stage!r} has invalid external input hashes")
            raw_path = record.get("path")
            expected_external = record.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(expected_external, str):
                raise ValueError(f"stage {stage!r} has invalid external input hashes")
            external_path = Path(raw_path)
            if (
                not external_path.is_absolute()
                or not external_path.is_file()
                or sha256_file(external_path) != expected_external
            ):
                raise ValueError(f"external input checksum mismatch: {raw_path}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"prerequisite stage {stage!r} has invalid outputs")
        for record in outputs:
            if not isinstance(record, dict):
                raise ValueError(f"prerequisite stage {stage!r} has invalid outputs")
            relative_path = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(expected, str):
                raise ValueError(f"prerequisite stage {stage!r} has invalid outputs")
            target = self._validated_target(relative_path)
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f"prerequisite output checksum mismatch: {relative_path}")
        self._verify_terminal_result_contract(manifest)
        verified.add(stage)
        return sha256_file(path)

    def _verify_stage_marker_only(
        self,
        stage: str,
        *,
        verified: set[str] | None = None,
    ) -> str:
        """Verify immutable prerequisite markers without opening their data outputs.

        This deliberately narrow mode exists only so a downstream stage can publish a
        bounded INVALID diagnostic after its resource budget expires.  A SUCCESS stage
        must always use the full prerequisite verification above.
        """

        if verified is None:
            verified = set()
        path = self.root / stage / "stage_manifest.json"
        if stage in verified:
            return sha256_file(path)
        prerequisite_hashes = {
            prerequisite: self._verify_stage_marker_only(prerequisite, verified=verified)
            for prerequisite in STAGE_PREREQUISITES[stage]
        }
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"prerequisite stage {stage!r} has no SUCCESS marker")
        manifest = self._read_json_file(path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("status") != "SUCCESS"
            or manifest.get("config_hash") != self.config_hash
            or manifest.get("prerequisite_manifest_hashes") != prerequisite_hashes
        ):
            raise ValueError(f"stage {stage!r} has an invalid SUCCESS marker")
        for field in ("input_hashes", "outputs"):
            records = manifest.get(field)
            if field == "input_hashes":
                valid = isinstance(records, dict) and all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in records.items()
                )
            else:
                valid = isinstance(records, list) and all(
                    isinstance(record, dict)
                    and isinstance(record.get("path"), str)
                    and isinstance(record.get("sha256"), str)
                    for record in records
                )
            if not valid:
                raise ValueError(f"stage {stage!r} marker has invalid {field}")
        external_inputs = manifest.get("external_input_hashes")
        if not isinstance(external_inputs, list) or not all(
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str)
            for record in external_inputs
        ):
            raise ValueError(f"stage {stage!r} marker has invalid external input hashes")
        verified.add(stage)
        return sha256_file(path)

    def _verify_stage_selected_outputs(
        self,
        stage: str,
        allowlist: tuple[str, ...],
    ) -> str:
        """Verify a SUCCESS marker and only explicitly selected committed outputs.

        This mode protects unopened partitions (notably scientific holdout data) while
        still binding a downstream SUCCESS stage to immutable, declared inputs.  The
        complete prerequisite marker remains structurally checked, but output bytes
        outside ``allowlist`` are never opened or hashed.
        """

        marker_hash = self._verify_stage_marker_only(stage)
        marker_path = self.root / stage / "stage_manifest.json"
        manifest = self._read_json_file(marker_path)
        raw_outputs = manifest.get("outputs")
        if not isinstance(raw_outputs, list):
            raise ValueError(f"stage {stage!r} marker has invalid outputs")
        output_hashes = {
            record.get("path"): record.get("sha256")
            for record in raw_outputs
            if isinstance(record, dict)
        }
        normalized: list[str] = []
        for raw_path in allowlist:
            target = self._validated_target(raw_path)
            relative = target.relative_to(self.root).as_posix()
            if not relative.startswith(f"{stage}/"):
                raise ValueError("prerequisite output allowlist crosses a stage boundary")
            if relative in normalized:
                raise ValueError("prerequisite output allowlist contains duplicates")
            expected = output_hashes.get(relative)
            if not isinstance(expected, str):
                raise ValueError(f"prerequisite output allowlist path is not declared: {relative}")
            if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                raise ValueError(f"prerequisite output checksum mismatch: {relative}")
            normalized.append(relative)
        if not normalized:
            raise ValueError("prerequisite output allowlist cannot be empty")
        return marker_hash

    def _declared_report_input_sha256(self, relative: str) -> str | None:
        """Return a five-stage output's authoritative checksum, or reject it."""

        source_stage = relative.split("/", 1)[0]
        if source_stage not in {
            "validate-data",
            "build-splits",
            "train-baselines",
            "build-photonic-features",
            "benchmark",
        }:
            return None
        marker_path = self.root / source_stage / "stage_manifest.json"
        if not marker_path.is_file() or marker_path.is_symlink():
            return None
        manifest = self._read_json_file(marker_path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != source_stage
            or manifest.get("status") != "SUCCESS"
            or manifest.get("config_hash") != self.config_hash
        ):
            return None
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            return None
        matches = [
            record.get("sha256")
            for record in outputs
            if isinstance(record, dict) and record.get("path") == relative
        ]
        if len(matches) != 1 or not isinstance(matches[0], str):
            return None
        return matches[0]

    @staticmethod
    def _exact_output_map(manifest: Mapping[str, object], stage: str) -> dict[str, str]:
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"stage {stage!r} has an invalid output inventory")
        result: dict[str, str] = {}
        for record in outputs:
            if not isinstance(record, dict):
                raise ValueError(f"stage {stage!r} has an invalid output inventory")
            path = record.get("path")
            digest = record.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str) or path in result:
                raise ValueError(f"stage {stage!r} has duplicate or invalid outputs")
            result[path] = digest
        return result

    def _expected_report_inputs_and_verify_lineage(self) -> frozenset[str]:
        """Derive the exact report input inventory and bind Task 8 to upstream outputs."""

        config_payload = json.loads(self.read_authenticated("config.json"))
        if not isinstance(config_payload, dict) or not isinstance(
            config_payload.get("paths"), dict
        ):
            raise ValueError("authenticated config snapshot is invalid")
        config_paths = config_payload["paths"]
        if any(
            not isinstance(config_paths.get(key), dict)
            or not isinstance(config_paths[key].get("value"), str)
            for key in ("input_data", "artifact_dir")
        ):
            raise ValueError("authenticated config paths are invalid")
        config_payload["paths"] = {
            key: config_paths[key]["value"] for key in ("input_data", "artifact_dir")
        }
        captured_config = PilotConfig.model_validate(config_payload)
        if captured_config.config_hash() != self.config_hash:
            raise ValueError("authenticated config does not reproduce its hash")
        self._verify_stage_marker_only("benchmark")
        marker_by_stage: dict[str, Mapping[str, object]] = {}
        marker_hashes: dict[str, str] = {}
        for stage in (
            "validate-data",
            "build-splits",
            "train-baselines",
            "build-photonic-features",
            "benchmark",
        ):
            marker_path = self.root / stage / "stage_manifest.json"
            if not marker_path.is_file() or marker_path.is_symlink():
                raise ValueError(f"stage {stage!r} has no regular SUCCESS marker")
            marker = self._read_json_file(marker_path)
            if (
                marker.get("schema_version") != SCHEMA_VERSION
                or marker.get("stage") != stage
                or marker.get("status") != "SUCCESS"
                or marker.get("config_hash") != self.config_hash
            ):
                raise ValueError(f"stage {stage!r} has an invalid SUCCESS marker")
            marker_by_stage[stage] = marker
            marker_hashes[stage] = sha256_file(marker_path)

        return self.report_input_inventory_from_captured_manifests(
            marker_by_stage, marker_hashes, captured_config=captured_config
        )

    def report_input_inventory_from_captured_manifests(
        self,
        marker_by_stage: Mapping[str, Mapping[str, object]],
        marker_hashes: Mapping[str, str],
        *,
        captured_config: PilotConfig,
    ) -> frozenset[str]:
        """Validate captured marker snapshots without reopening their paths."""

        expected_stages = (
            "validate-data",
            "build-splits",
            "train-baselines",
            "build-photonic-features",
            "benchmark",
        )
        if set(marker_by_stage) != set(expected_stages) or set(marker_hashes) != set(
            expected_stages
        ):
            raise ValueError("report stage-marker snapshots must cover the exact lineage")
        output_by_stage: dict[str, dict[str, str]] = {}
        for stage in expected_stages:
            marker = marker_by_stage[stage]
            digest = marker_hashes[stage]
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or marker.get("schema_version") != SCHEMA_VERSION
                or marker.get("stage") != stage
                or marker.get("status") != "SUCCESS"
                or marker.get("config_hash") != self.config_hash
            ):
                raise ValueError(f"stage {stage!r} has an invalid captured SUCCESS marker")
            expected_prerequisites = {
                prerequisite: marker_hashes[prerequisite]
                for prerequisite in STAGE_PREREQUISITES[stage]
            }
            if marker.get("prerequisite_manifest_hashes") != expected_prerequisites:
                raise ValueError(f"stage {stage!r} captured prerequisite lineage is invalid")
            output_by_stage[stage] = self._exact_output_map(marker, stage)

        circuit_paths = {
            path
            for path in output_by_stage["build-photonic-features"]
            if path.startswith("build-photonic-features/circuits/") and path.endswith(".npz")
        }
        if not circuit_paths:
            raise ValueError("photonic output marker has no circuit inventory")
        optional_profile_paths = {
            path
            for path in output_by_stage["build-photonic-features"]
            if path in PROFILE_FIXED_PATHS
            or (path.startswith(f"{PROFILE_PREFIX}/circuits/") and path.endswith(".npz"))
        }
        diagnostic_paths = set(output_by_stage["benchmark"]) & set(DIAGNOSTIC_PATHS)
        if captured_config.config_hash() != self.config_hash:
            raise ValueError("captured report config hash does not match the run")
        photonics = captured_config.photonics
        if (
            bool(optional_profile_paths) != photonics.enable_three_photon_profile
            or (optional_profile_paths and not set(PROFILE_FIXED_PATHS) <= optional_profile_paths)
            or (
                diagnostic_paths
                != (set(DIAGNOSTIC_PATHS) if photonics.enable_three_photon_benchmark else set())
            )
        ):
            raise ValueError("optional report inventory does not match frozen opt-ins")

        benchmark_inputs = marker_by_stage["benchmark"].get("input_hashes")
        if not isinstance(benchmark_inputs, dict) or any(
            not isinstance(path, str) or not isinstance(digest, str)
            for path, digest in benchmark_inputs.items()
        ):
            raise ValueError("benchmark input lineage is invalid")
        upstream_paths = {
            *_BENCHMARK_BASELINE_OUTPUTS,
            *_BENCHMARK_PHOTONIC_REQUIRED_OUTPUTS,
            *circuit_paths,
            *optional_profile_paths,
        }
        exact_benchmark_inputs = upstream_paths | set(_BENCHMARK_INTERNAL_INPUTS)
        if set(benchmark_inputs) != exact_benchmark_inputs:
            raise ValueError("benchmark input lineage has missing or extra entries")
        for path in upstream_paths:
            source_stage = path.split("/", 1)[0]
            authoritative = output_by_stage[source_stage].get(path)
            if authoritative is None or benchmark_inputs.get(path) != authoritative:
                raise ValueError("benchmark input checksum is not anchored upstream")
        for path in _BENCHMARK_INTERNAL_INPUTS:
            digest = benchmark_inputs[path]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("benchmark internal input checksum is invalid")
        return frozenset(
            {*_REPORT_FIXED_INPUTS, *circuit_paths, *optional_profile_paths, *diagnostic_paths}
        )

    @staticmethod
    def _verify_report_output_inventory(manifest: Mapping[str, object]) -> None:
        status = manifest.get("status")
        expected = _REPORT_INVALID_OUTPUTS if status == "INVALID" else _REPORT_TERMINAL_OUTPUTS
        actual = set(RunStore._exact_output_map(manifest, "report"))
        if actual != set(expected):
            raise ValueError("report output inventory does not match exact outputs")

    def _verify_terminal_result_contract(self, manifest: Mapping[str, object]) -> None:
        """Bind the conventional JSON result to its marker even if declaration is removed."""

        stage = manifest.get("stage")
        outputs = manifest.get("outputs")
        if not isinstance(stage, str) or not isinstance(outputs, list):
            raise ValueError("stage terminal result declaration is invalid")
        conventional_path = f"{stage}/stage_result.json"
        matching_outputs = [
            record
            for record in outputs
            if isinstance(record, dict) and record.get("path") == conventional_path
        ]
        result_path = self._validated_target(conventional_path)
        result_exists = result_path.exists() or result_path.is_symlink()
        declaration = manifest.get("terminal_result")
        if not matching_outputs:
            if result_exists:
                raise ValueError("terminal result exists but is not a committed stage output")
            if declaration is not None:
                raise ValueError("terminal result declaration has no committed stage output")
            return
        if len(matching_outputs) != 1:
            raise ValueError("terminal result must appear exactly once in stage outputs")
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("committed terminal result is not a regular file")
        marker_status = _normalize_terminal_status(manifest.get("status"))
        payload = self._read_json_file(result_path)
        if payload.get("stage") != stage:
            raise ValueError("terminal result stage does not match stage marker")
        if _normalize_terminal_status(payload.get("status")) != marker_status:
            raise ValueError("terminal result status does not match stage marker")
        if not isinstance(declaration, dict):
            raise ValueError("terminal result declaration is required")
        if declaration.get("path") != conventional_path:
            raise ValueError("terminal result declaration path is inconsistent")
        if _normalize_terminal_status(declaration.get("status")) != marker_status:
            raise ValueError("terminal result status does not match stage marker")

    def verify_terminal_stage(
        self,
        stage: str,
        *,
        accepted_statuses: frozenset[str] = frozenset({"SUCCESS"}),
        prerequisite_output_allowlist: Mapping[str, Iterable[str | Path]] | None = None,
    ) -> str:
        """Verify a terminal stage and all outputs without relaxing prerequisites."""

        if stage not in STAGE_PREREQUISITES:
            raise ValueError(f"unknown pipeline stage: {stage!r}")
        if not accepted_statuses or not accepted_statuses <= {"SUCCESS", "BLOCKED", "INVALID"}:
            raise ValueError("accepted terminal statuses are invalid")
        if prerequisite_output_allowlist is not None:
            if stage not in {
                _TASK7_SELECTED_STAGE,
                _BENCHMARK_SELECTED_STAGE,
                _REPORT_SELECTED_STAGE,
            } or (
                accepted_statuses != frozenset({"SUCCESS"})
                and not (
                    stage == _REPORT_SELECTED_STAGE and accepted_statuses == frozenset({"BLOCKED"})
                )
            ):
                raise ValueError(
                    "selected_outputs verification is limited to audited terminal stages"
                )
            policy_paths = {
                prerequisite: tuple(
                    self._validated_target(path).relative_to(self.root).as_posix() for path in paths
                )
                for prerequisite, paths in prerequisite_output_allowlist.items()
            }
            self._validate_selected_output_policy(stage, policy_paths)
            prerequisite_hashes = {
                prerequisite: (
                    self._verify_stage_marker_only(prerequisite)
                    if stage == _REPORT_SELECTED_STAGE
                    else self._verify_stage_selected_outputs(
                        prerequisite,
                        tuple(str(path) for path in prerequisite_output_allowlist[prerequisite]),
                    )
                )
                for prerequisite in STAGE_PREREQUISITES[stage]
            }
            marker_path = self.root / stage / "stage_manifest.json"
            if not marker_path.is_file() or marker_path.is_symlink():
                raise ValueError(f"stage {stage!r} has no terminal marker")
            manifest = self._read_json_file(marker_path)
            normalized_allowlist = {
                prerequisite: sorted(
                    self._validated_target(path).relative_to(self.root).as_posix()
                    for path in prerequisite_output_allowlist[prerequisite]
                )
                for prerequisite in STAGE_PREREQUISITES[stage]
            }
            details = manifest.get("details")
            if (
                manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("stage") != stage
                or manifest.get("status") not in accepted_statuses
                or manifest.get("config_hash") != self.config_hash
                or manifest.get("prerequisite_manifest_hashes") != prerequisite_hashes
                or not isinstance(details, dict)
                or details.get("prerequisite_verification") != "selected_outputs"
                or details.get("prerequisite_output_allowlist") != normalized_allowlist
            ):
                raise ValueError(f"stage {stage!r} has an invalid selective verification policy")
            inputs = manifest.get("input_hashes")
            if not isinstance(inputs, dict):
                raise ValueError(f"stage {stage!r} has invalid input hashes")
            report_expected_inputs = (
                self._expected_report_inputs_and_verify_lineage()
                if stage == _REPORT_SELECTED_STAGE
                else None
            )
            selected_paths = {path for paths in normalized_allowlist.values() for path in paths}
            for relative, expected in inputs.items():
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise ValueError(f"stage {stage!r} has invalid input hashes")
                target = self._validated_target(relative)
                if (
                    relative not in selected_paths
                    and not (
                        stage == _BENCHMARK_SELECTED_STAGE
                        and relative in _BENCHMARK_INTERNAL_INPUTS
                    )
                    and not (
                        stage == _REPORT_SELECTED_STAGE
                        and report_expected_inputs is not None
                        and relative in report_expected_inputs
                    )
                ):
                    raise ValueError("selected-output stage input is not audited")
                if stage == _REPORT_SELECTED_STAGE:
                    declared_checksum = self._declared_report_input_sha256(relative)
                    if declared_checksum is not None and declared_checksum != expected:
                        raise ValueError(
                            "report input checksum is not anchored to its source marker"
                        )
                if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                    raise ValueError(f"stage input checksum mismatch: {relative}")
            expected_inputs = selected_paths | (
                _BENCHMARK_INTERNAL_INPUTS if stage == _BENCHMARK_SELECTED_STAGE else frozenset()
            )
            report_inputs_invalid = False
            if stage == _REPORT_SELECTED_STAGE:
                assert report_expected_inputs is not None
                report_inputs_invalid = set(inputs) != set(report_expected_inputs)
            if report_inputs_invalid or (
                stage != _REPORT_SELECTED_STAGE and set(inputs) != expected_inputs
            ):
                raise ValueError("selected-output stage inputs do not match the fixed contract")
            outputs = manifest.get("outputs")
            if not isinstance(outputs, list):
                raise ValueError(f"stage {stage!r} has invalid outputs")
            for record in outputs:
                if not isinstance(record, dict):
                    raise ValueError(f"stage {stage!r} has invalid outputs")
                relative = record.get("path")
                expected = record.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise ValueError(f"stage {stage!r} has invalid outputs")
                target = self._validated_target(relative)
                if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                    raise ValueError(f"stage output checksum mismatch: {relative}")
            if stage == _REPORT_SELECTED_STAGE:
                self._verify_report_output_inventory(manifest)
            self._verify_terminal_result_contract(manifest)
            return sha256_file(marker_path)
        return self._verify_stage(stage, accepted_statuses=accepted_statuses)

    def verify_selected_stage_outputs(self, stage: str, output_paths: Iterable[str | Path]) -> str:
        """Public audited boundary for holdout-safe prerequisite verification."""

        paths = tuple(
            self._validated_target(path).relative_to(self.root).as_posix() for path in output_paths
        )
        valid_task7 = stage == _TASK7_SELECTED_PREREQUISITE and tuple(sorted(paths)) == tuple(
            sorted(_TASK7_SELECTED_OUTPUTS)
        )
        valid_report = stage == _BENCHMARK_SELECTED_STAGE and tuple(sorted(paths)) == tuple(
            sorted(_REPORT_BENCHMARK_OUTPUTS)
        )
        if not (valid_task7 or valid_report):
            raise ValueError("selected stage output verification violates a fixed contract")
        return self._verify_stage_selected_outputs(stage, paths)

    def report_input_inventory(self) -> tuple[str, ...]:
        """Return the fixed, lineage-verified report input inventory."""

        return tuple(sorted(self._expected_report_inputs_and_verify_lineage()))

    def verify_terminal_diagnostic_stage(self, stage: str) -> str:
        """Verify an INVALID marker-only diagnostic without reading unrelated data."""

        if stage not in STAGE_PREREQUISITES:
            raise ValueError(f"unknown pipeline stage: {stage!r}")
        prerequisite_hashes = {
            prerequisite: self._verify_stage_marker_only(prerequisite)
            for prerequisite in STAGE_PREREQUISITES[stage]
        }
        marker_path = self.root / stage / "stage_manifest.json"
        if not marker_path.is_file() or marker_path.is_symlink():
            raise ValueError(f"stage {stage!r} has no INVALID marker")
        manifest = self._read_json_file(marker_path)
        details = manifest.get("details")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("status") != "INVALID"
            or manifest.get("config_hash") != self.config_hash
            or manifest.get("prerequisite_manifest_hashes") != prerequisite_hashes
            or not isinstance(details, dict)
            or details.get("prerequisite_verification") != "marker_only"
        ):
            raise ValueError(f"stage {stage!r} has an invalid diagnostic policy")
        inputs = manifest.get("input_hashes")
        if not isinstance(inputs, dict):
            raise ValueError(f"stage {stage!r} has invalid input hashes")
        for relative, expected in inputs.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError(f"stage {stage!r} has invalid input hashes")
            target = self._validated_target(relative)
            if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                raise ValueError(f"stage input checksum mismatch: {relative}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"stage {stage!r} has invalid outputs")
        for record in outputs:
            if not isinstance(record, dict):
                raise ValueError(f"stage {stage!r} has invalid outputs")
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError(f"stage {stage!r} has invalid outputs")
            target = self._validated_target(relative)
            if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                raise ValueError(f"stage output checksum mismatch: {relative}")
        if stage == _REPORT_SELECTED_STAGE:
            self._verify_report_output_inventory(manifest)
        self._verify_terminal_result_contract(manifest)
        return sha256_file(marker_path)

    def stage(
        self,
        stage: str,
        *,
        input_paths: Iterable[str | Path] = (),
        external_input_hashes: Mapping[str | Path, str] | None = None,
        details: Mapping[str, object] | None = None,
        terminal_status: Literal["SUCCESS", "BLOCKED", "INVALID"] = "SUCCESS",
        prerequisite_verification: Literal["full", "marker_only", "selected_outputs"] = "full",
        prerequisite_output_allowlist: Mapping[str, Iterable[str | Path]] | None = None,
        commit_guard: CommitGuard | None = None,
        private_write_guard: CommitGuard | None = None,
        preverified_input_hashes: Mapping[str, str] | None = None,
        preverified_prerequisite_manifest_hashes: Mapping[str, str] | None = None,
        config_hash: str | None = None,
        now: Now = None,
    ) -> StageTransaction:
        """Validate prerequisites and create a transactional stage context."""

        if stage not in STAGE_PREREQUISITES:
            raise ValueError(f"unknown pipeline stage: {stage!r}")
        if terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("terminal status must be SUCCESS, BLOCKED, or INVALID")
        if prerequisite_verification not in {"full", "marker_only", "selected_outputs"}:
            raise ValueError("prerequisite verification mode is invalid")
        if prerequisite_verification == "marker_only" and terminal_status != "INVALID":
            raise ValueError("marker_only prerequisite verification is limited to INVALID stages")
        if prerequisite_verification == "selected_outputs":
            if stage not in {
                _TASK7_SELECTED_STAGE,
                _BENCHMARK_SELECTED_STAGE,
                _REPORT_SELECTED_STAGE,
            } or (
                terminal_status != "SUCCESS"
                and not (stage == _REPORT_SELECTED_STAGE and terminal_status == "BLOCKED")
            ):
                raise ValueError("selected_outputs is limited to audited terminal stages")
            if prerequisite_output_allowlist is None:
                raise ValueError("selected_outputs requires a prerequisite output allowlist")
            policy_paths = {
                prerequisite: tuple(
                    self._validated_target(path).relative_to(self.root).as_posix() for path in paths
                )
                for prerequisite, paths in prerequisite_output_allowlist.items()
            }
            self._validate_selected_output_policy(stage, policy_paths)
        elif prerequisite_output_allowlist is not None:
            raise ValueError("prerequisite output allowlist requires selected_outputs verification")
        report_preverified = (
            stage == _REPORT_SELECTED_STAGE and preverified_input_hashes is not None
        )
        if preverified_input_hashes is not None and not report_preverified:
            expected_prerequisites = set(STAGE_PREREQUISITES[stage])
            if (
                preverified_prerequisite_manifest_hashes is None
                or set(preverified_prerequisite_manifest_hashes) != expected_prerequisites
            ):
                raise ValueError(
                    "preverified stage inputs require the exact prerequisite marker hashes"
                )
        elif preverified_prerequisite_manifest_hashes is not None and not report_preverified:
            raise ValueError("preverified prerequisite hashes require preverified inputs")
        for digest in {
            **dict(preverified_input_hashes or {}),
            **dict(preverified_prerequisite_manifest_hashes or {}),
        }.values():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("preverified hashes must be lowercase SHA-256 digests")
        if commit_guard is not None:
            commit_guard()
        with self._publication_lock():
            self._recover_publication_journals()
            self._cleanup_stale_stage_directories()
        if commit_guard is not None:
            commit_guard()
        if not report_preverified:
            self._verify_base(expected_config_hash=config_hash)
        elif config_hash is not None and config_hash != self.config_hash:
            raise ValueError("expected config hash does not match run config hash")
        self.verify_execution_provenance()
        if commit_guard is not None:
            commit_guard()
        marker = self.root / stage / "stage_manifest.json"
        if marker.exists() or marker.is_symlink():
            raise FileExistsError(f"stage {stage!r} is already completed")
        prerequisite_hashes: dict[str, str] = {}
        normalized_allowlist: dict[str, list[str]] = {}
        for prerequisite in STAGE_PREREQUISITES[stage]:
            if commit_guard is not None:
                commit_guard()
            if preverified_prerequisite_manifest_hashes is not None:
                prerequisite_hashes[prerequisite] = preverified_prerequisite_manifest_hashes[
                    prerequisite
                ]
                if prerequisite_verification == "selected_outputs":
                    assert prerequisite_output_allowlist is not None
                    normalized_allowlist[prerequisite] = sorted(
                        self._validated_target(path).relative_to(self.root).as_posix()
                        for path in prerequisite_output_allowlist[prerequisite]
                    )
            elif prerequisite_verification == "marker_only":
                prerequisite_hashes[prerequisite] = self._verify_stage_marker_only(prerequisite)
            elif prerequisite_verification == "selected_outputs":
                assert prerequisite_output_allowlist is not None
                paths = tuple(
                    self._validated_target(path).relative_to(self.root).as_posix()
                    for path in prerequisite_output_allowlist[prerequisite]
                )
                if report_preverified:
                    assert preverified_input_hashes is not None
                    marker_digest = preverified_input_hashes.get(
                        f"{prerequisite}/stage_manifest.json"
                    )
                    if not isinstance(marker_digest, str):
                        raise ValueError("captured prerequisite marker checksum is invalid")
                    prerequisite_hashes[prerequisite] = marker_digest
                else:
                    prerequisite_hashes[prerequisite] = self._verify_stage_selected_outputs(
                        prerequisite, paths
                    )
                normalized_allowlist[prerequisite] = sorted(paths)
            else:
                prerequisite_hashes[prerequisite] = self._verify_stage(prerequisite)
            if commit_guard is not None:
                commit_guard()
        inputs: dict[str, str] = {}
        selected_paths = {path for paths in normalized_allowlist.values() for path in paths}
        normalized_requested_inputs = tuple(
            self._validated_target(path).relative_to(self.root).as_posix() for path in input_paths
        )
        if preverified_input_hashes is not None:
            if len(normalized_requested_inputs) != len(set(normalized_requested_inputs)):
                raise ValueError("report input paths contain duplicates")
            if set(normalized_requested_inputs) != set(preverified_input_hashes):
                raise ValueError("captured report inputs do not match requested inputs")
        for relative in normalized_requested_inputs:
            if commit_guard is not None:
                commit_guard()
            target = self._validated_target(relative)
            if (
                prerequisite_verification == "selected_outputs"
                and relative not in selected_paths
                and not (
                    stage == _BENCHMARK_SELECTED_STAGE and relative in _BENCHMARK_INTERNAL_INPUTS
                )
                and not (
                    stage == _REPORT_SELECTED_STAGE
                    and (
                        (
                            report_preverified
                            and (
                                relative in _REPORT_FIXED_INPUTS
                                or relative in PROFILE_FIXED_PATHS
                                or relative in DIAGNOSTIC_PATHS
                                or (
                                    relative.startswith(f"{PROFILE_PREFIX}/circuits/")
                                    and relative.endswith(".npz")
                                )
                                or (
                                    relative.startswith("build-photonic-features/circuits/")
                                    and relative.endswith(".npz")
                                )
                            )
                        )
                        or self._declared_report_input_sha256(relative) is not None
                    )
                )
            ):
                raise ValueError("selected-output stage inputs must be explicitly allowlisted")
            if preverified_input_hashes is None:
                if not target.is_file():
                    raise FileNotFoundError(f"stage input is not a file: {relative}")
                inputs[relative] = sha256_file(target)
            else:
                captured = preverified_input_hashes.get(relative)
                if not isinstance(captured, str):
                    raise ValueError("captured report input checksum is invalid")
                inputs[relative] = captured
            if (
                prerequisite_verification == "selected_outputs"
                and stage == _REPORT_SELECTED_STAGE
                and not report_preverified
                and self._declared_report_input_sha256(relative) != inputs[relative]
            ):
                raise ValueError("report input checksum is not anchored to its source marker")
            if commit_guard is not None:
                commit_guard()
        if prerequisite_verification == "selected_outputs":
            expected_inputs = selected_paths | (
                _BENCHMARK_INTERNAL_INPUTS if stage == _BENCHMARK_SELECTED_STAGE else frozenset()
            )
            report_circuits = {
                path
                for path in inputs
                if path.startswith("build-photonic-features/circuits/") and path.endswith(".npz")
            }
            optional_report_inputs = {
                path
                for path in inputs
                if path in PROFILE_FIXED_PATHS
                or path in DIAGNOSTIC_PATHS
                or (path.startswith(f"{PROFILE_PREFIX}/circuits/") and path.endswith(".npz"))
            }
            captured_report_inputs = frozenset(
                {*_REPORT_FIXED_INPUTS, *report_circuits, *optional_report_inputs}
            )
            if (
                stage == _REPORT_SELECTED_STAGE
                and (
                    not report_circuits
                    or set(inputs)
                    != (
                        captured_report_inputs
                        if report_preverified
                        else self._expected_report_inputs_and_verify_lineage()
                    )
                )
            ) or (stage != _REPORT_SELECTED_STAGE and set(inputs) != expected_inputs):
                raise ValueError("selected-output stage inputs do not match the fixed contract")
        external_inputs: dict[str, str] = {}
        for input_path, expected in (external_input_hashes or {}).items():
            if commit_guard is not None:
                commit_guard()
            target = Path(input_path).resolve()
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ValueError("external input checksum must be a lowercase SHA-256 digest")
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f"external stage input changed: {target}")
            external_inputs[target.as_posix()] = expected
            if commit_guard is not None:
                commit_guard()
        if commit_guard is not None:
            commit_guard()
        return StageTransaction(
            store=self,
            stage=stage,
            input_hashes=inputs,
            external_input_hashes=external_inputs,
            prerequisite_manifest_hashes=prerequisite_hashes,
            details={
                **dict(details or {}),
                **(
                    {"prerequisite_verification": "marker_only"}
                    if prerequisite_verification == "marker_only"
                    else {}
                ),
                **(
                    {
                        "prerequisite_verification": "selected_outputs",
                        "prerequisite_output_allowlist": normalized_allowlist,
                    }
                    if prerequisite_verification == "selected_outputs"
                    else {}
                ),
            },
            terminal_status=terminal_status,
            commit_guard=commit_guard,
            private_write_guard=private_write_guard,
            rehash_inputs_on_commit=preverified_input_hashes is None,
            clock=_clock(now),
        )


class StageTransaction:
    """Collect outputs privately and commit them before the SUCCESS marker."""

    def __init__(
        self,
        *,
        store: RunStore,
        stage: str,
        input_hashes: dict[str, str],
        external_input_hashes: dict[str, str],
        prerequisite_manifest_hashes: dict[str, str],
        details: dict[str, object],
        terminal_status: Literal["SUCCESS", "BLOCKED", "INVALID"],
        commit_guard: CommitGuard | None,
        private_write_guard: CommitGuard | None,
        rehash_inputs_on_commit: bool,
        clock: Clock,
    ) -> None:
        if terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("terminal status must be SUCCESS, BLOCKED, or INVALID")
        self._store = store
        self.stage = stage
        self._input_hashes = input_hashes
        self._external_input_hashes = external_input_hashes
        self._prerequisite_manifest_hashes = prerequisite_manifest_hashes
        self._details = details
        self._terminal_status = terminal_status
        self._commit_guard = commit_guard
        self._private_write_guard = private_write_guard
        self._rehash_inputs_on_commit = rehash_inputs_on_commit
        self._clock = clock
        self._started_at: datetime | None = None
        self._temporary_root: Path | None = None
        self._lease_stream: BinaryIO | None = None
        self._outputs: dict[str, Path] = {}

    def __enter__(self) -> Self:
        if self._private_write_guard is not None:
            self._private_write_guard()
        with self._store._publication_lock():
            self._started_at = self._clock()
            temporary_root = Path(
                tempfile.mkdtemp(prefix=f".stage-{self.stage}-", dir=self._store.root)
            )
            try:
                lease = self._store._open_safe_lock(temporary_root / ".lease")
                if not self._store._lock_stream(lease, blocking=False):
                    lease.close()
                    raise RuntimeError("could not acquire private stage lease")
            except BaseException:
                shutil.rmtree(temporary_root, ignore_errors=True)
                raise
            self._temporary_root = temporary_root
            self._lease_stream = lease
        return self

    def _release_lease(self) -> None:
        lease = self._lease_stream
        if lease is None:
            return
        self._store._unlock_stream(lease)
        lease.close()
        self._lease_stream = None

    def _temporary_target(self, relative_path: str | Path) -> tuple[str, Path]:
        if self._temporary_root is None:
            raise RuntimeError("stage transaction is not active")
        candidate = Path(relative_path)
        if candidate.is_absolute() or candidate.drive:
            raise ValueError("stage output paths must be relative")
        if not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("stage output path cannot escape its transaction")
        logical = (Path(self.stage) / candidate).as_posix()
        if logical.endswith("/stage_manifest.json"):
            raise ValueError("stage_manifest.json is reserved for the success marker")
        final = self._store._validated_target(logical)
        temporary = (self._temporary_root / candidate).resolve(strict=False)
        if not temporary.is_relative_to(self._temporary_root):
            raise ValueError("stage output path cannot escape its transaction")
        if logical in self._outputs or final.exists() or final.is_symlink():
            raise FileExistsError(f"stage output already exists: {logical}")
        return logical, temporary

    def write_bytes(self, relative_path: str | Path, data: bytes) -> Path:
        if self._private_write_guard is not None:
            self._private_write_guard()
        logical, temporary = self._temporary_target(relative_path)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if self._private_write_guard is not None:
            self._private_write_guard()
        self._outputs[logical] = temporary
        return self._store.root / logical

    def write_file(
        self,
        relative_path: str | Path,
        source_path: str | Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> Path:
        """Stream a verified private worker output into this transaction."""

        if self._private_write_guard is not None:
            self._private_write_guard()
        source = Path(source_path)
        resolved = source.resolve(strict=False)
        source_parent_before = source.parent.lstat()
        if (
            not resolved.is_relative_to(self._store.root)
            or source.is_symlink()
            or source.is_junction()
            or source.parent.is_symlink()
            or source.parent.is_junction()
            or not stat.S_ISDIR(source_parent_before.st_mode)
            or source_parent_before.st_nlink != 1
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
            or expected_size < 0
        ):
            raise ValueError("worker output source contract is invalid")
        before = source.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ValueError("worker output is not a regular single-link file")
        logical, temporary = self._temporary_target(relative_path)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        digest = hashlib.sha256()
        copied = 0
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != expected_size
                or (before.st_ino and opened.st_ino != before.st_ino)
            ):
                raise ValueError("worker output changed while opening")
            with temporary.open("xb") as destination:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > expected_size:
                        raise ValueError("worker output exceeded its declared size")
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            after = os.fstat(descriptor)
            source_parent_after = source.parent.lstat()
            if (
                copied != expected_size
                or after.st_size != opened.st_size
                or after.st_mtime_ns != opened.st_mtime_ns
                or digest.hexdigest() != expected_sha256
                or source.parent.is_symlink()
                or source.parent.is_junction()
                or source_parent_after.st_dev != source_parent_before.st_dev
                or (
                    source_parent_before.st_ino
                    and source_parent_after.st_ino != source_parent_before.st_ino
                )
            ):
                raise ValueError("worker output checksum or size mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        if self._private_write_guard is not None:
            self._private_write_guard()
        self._outputs[logical] = temporary
        return self._store.root / logical

    def write_text(
        self,
        relative_path: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        if self._private_write_guard is not None:
            self._private_write_guard()
        data = text.encode(encoding)
        if self._private_write_guard is not None:
            self._private_write_guard()
        return self.write_bytes(relative_path, data)

    def write_json(self, relative_path: str | Path, value: object) -> Path:
        if self._private_write_guard is not None:
            self._private_write_guard()
        payload = _canonical_json(value)
        if self._private_write_guard is not None:
            self._private_write_guard()
        return self.write_text(relative_path, payload)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception, traceback
        temporary_root = self._temporary_root
        if temporary_root is None or self._started_at is None:
            raise RuntimeError("stage transaction was not entered")
        if exception_type is not None:
            self._release_lease()
            shutil.rmtree(temporary_root, ignore_errors=True)
            return False
        result_logical = f"{self.stage}/stage_result.json"
        terminal_result: dict[str, str] | None = None
        result_temporary = self._outputs.get(result_logical)
        if result_temporary is not None:
            try:
                result_payload = self._store._read_json_file(result_temporary)
                if result_payload.get("stage") != self.stage:
                    raise ValueError("terminal result stage does not match its transaction")
                result_status = _normalize_terminal_status(result_payload.get("status"))
                if result_status != self._terminal_status:
                    raise ValueError("terminal result status does not match stage marker")
                terminal_result = {"path": result_logical, "status": result_status}
            except BaseException:
                self._release_lease()
                shutil.rmtree(temporary_root, ignore_errors=True)
                raise
        transaction_id = uuid4().hex
        marker_relative = f"{self.stage}/stage_manifest.json"
        published = PublicationBatch(
            transaction_id=transaction_id,
            stage=self.stage,
            marker_relative=marker_relative,
            terminal_status=self._terminal_status,
            output_hashes={},
        )
        try:
            with self._store._publication_lock():
                self._store._recover_publication_journals()
                if self._commit_guard is not None:
                    self._commit_guard()
                output_records = [
                    {"path": logical, "sha256": sha256_file(temporary)}
                    for logical, temporary in sorted(self._outputs.items())
                ]
                published.output_hashes.update(
                    {str(record["path"]): str(record["sha256"]) for record in output_records}
                )
                if self._commit_guard is not None:
                    self._commit_guard()
                if self._rehash_inputs_on_commit:
                    for relative_path, expected in self._input_hashes.items():
                        if self._commit_guard is not None:
                            self._commit_guard()
                        target = self._store._validated_target(relative_path)
                        if not target.is_file() or sha256_file(target) != expected:
                            raise ValueError(f"stage input changed: {relative_path}")
                        if self._commit_guard is not None:
                            self._commit_guard()
                for raw_path, expected in self._external_input_hashes.items():
                    if self._commit_guard is not None:
                        self._commit_guard()
                    external_path = Path(raw_path)
                    if not external_path.is_file() or sha256_file(external_path) != expected:
                        raise ValueError(f"external stage input changed: {raw_path}")
                    if self._commit_guard is not None:
                        self._commit_guard()
                marker = self._store._validated_target(marker_relative)
                if marker.exists() or marker.is_symlink():
                    raise FileExistsError(f"stage {self.stage!r} is already completed")
                for logical in self._outputs:
                    destination = self._store._validated_target(logical)
                    if destination.exists() or destination.is_symlink():
                        raise FileExistsError(f"stage output already exists: {logical}")
                for logical, temporary in sorted(self._outputs.items()):
                    if self._commit_guard is not None:
                        self._commit_guard()
                    destination = self._store._validated_target(logical)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self._store._publish_temp(temporary, destination, published)
                    if self._commit_guard is not None:
                        self._commit_guard()
                ended_at = self._clock()
                manifest = {
                    "config_hash": self._store.config_hash,
                    "details": self._details,
                    "ended_at": _timestamp(ended_at),
                    "input_hashes": self._input_hashes,
                    "external_input_hashes": [
                        {"path": path, "sha256": digest}
                        for path, digest in sorted(self._external_input_hashes.items())
                    ],
                    "outputs": output_records,
                    "prerequisite_manifest_hashes": self._prerequisite_manifest_hashes,
                    "schema_version": SCHEMA_VERSION,
                    "stage": self.stage,
                    "started_at": _timestamp(self._started_at),
                    "status": self._terminal_status,
                    "terminal_result": terminal_result,
                    "transaction_id": transaction_id,
                }
                if self._commit_guard is not None:
                    self._commit_guard()
                self._store._write_bytes(
                    marker_relative,
                    _canonical_json(manifest).encode("utf-8"),
                    published,
                )
                if self._commit_guard is not None:
                    self._commit_guard()
                # The last guard is intentionally before journal release. A guard
                # failure can still be durably rolled back while ownership records
                # exist; checking after release would weaken crash recovery.
                self._store._release_publications(published)
        except BaseException as error:
            with self._store._publication_lock():
                if published:
                    self._store._mark_batch_aborted(published)
                try:
                    self._store._rollback_publications(published, error)
                finally:
                    if published and not any(
                        publication.quarantine_root.exists() for publication in published
                    ):
                        self._store._clear_batch_abort(published.transaction_id)
            raise
        finally:
            self._release_lease()
            shutil.rmtree(temporary_root, ignore_errors=True)
        return False


__all__ = [
    "SCHEMA_VERSION",
    "STAGE_PREREQUISITES",
    "RunStore",
    "StageTransaction",
    "sha256_file",
]
