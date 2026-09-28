"""Immutable run artifacts and transactional stage contracts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from .config import LoadedPilotConfig
from .seeds import SeedManager

SCHEMA_VERSION = "1.0"
Clock = Callable[[], datetime]
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


@dataclass(frozen=True, slots=True)
class PublicationOwnership:
    """Identity-bound permission to remove one published artifact."""

    destination: Path
    source_device: int
    source_inode: int

    def _is_owned(self, path: Path) -> bool:
        path_stat = path.stat(follow_symlinks=False)
        return (
            path_stat.st_dev == self.source_device
            and path_stat.st_ino == self.source_inode
        )

    def _restore_foreign(self, quarantine: Path) -> None:
        if os.name == "nt":
            os.rename(quarantine, self.destination)
            return
        os.link(quarantine, self.destination, follow_symlinks=False)
        raise OSError(
            "foreign publication was restored but retained in quarantine at "
            f"{quarantine}"
        )

    def _settle_quarantine(self, quarantine: Path) -> None:
        try:
            owned = self._is_owned(quarantine)
        except FileNotFoundError:
            return
        if owned:
            quarantine.unlink()
            return
        self._restore_foreign(quarantine)

    def _recover_quarantine(
        self, quarantine: Path, interrupted_operation: BaseException
    ) -> None:
        try:
            self._settle_quarantine(quarantine)
        except BaseException as recovery_error:
            interrupted_operation.add_note(
                f"publication quarantine recovery failed: {recovery_error}"
            )

    def rollback(self) -> None:
        """Atomically acquire the destination before deciding whether to delete it."""

        quarantine_root = Path(
            tempfile.mkdtemp(
                prefix=f".rollback-{self.destination.name}-",
                dir=self.destination.parent,
            )
        )
        quarantine = quarantine_root / "entry"
        try:
            try:
                os.rename(self.destination, quarantine)
            except FileNotFoundError:
                return
            except BaseException as move_error:
                self._recover_quarantine(quarantine, move_error)
                raise
            try:
                self._settle_quarantine(quarantine)
            except BaseException as settle_error:
                self._recover_quarantine(quarantine, settle_error)
                raise
        finally:
            with suppress(OSError):
                quarantine_root.rmdir()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            environment_manifest = {
                "config_hash": config_hash,
                "git": _git_manifest(loaded.project_root),
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
                {"path": path, "sha256": sha256_file(store.root / path)}
                for path in _BASE_ARTIFACTS
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
        store._verify_base(expected_config_hash=expected_config_hash)
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

    @staticmethod
    def _rollback_publications(
        publications: Iterable[PublicationOwnership], error: BaseException
    ) -> None:
        cleanup_errors: list[BaseException] = []
        for publication in reversed(tuple(publications)):
            try:
                publication.rollback()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for recorded_cleanup_error in cleanup_errors:
            error.add_note(f"publication cleanup failed: {recorded_cleanup_error}")

    def _publish_temp(
        self,
        source: Path,
        destination: Path,
        publications: list[PublicationOwnership] | None = None,
    ) -> None:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"artifact already exists: {destination}")
        source_stat = source.stat(follow_symlinks=False)
        ownership = PublicationOwnership(
            destination=destination,
            source_device=source_stat.st_dev,
            source_inode=source_stat.st_ino,
        )
        if publications is None:
            publications = []
        publications.append(ownership)
        try:
            os.link(source, destination)
            source.unlink()
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

    def _verify_stage(self, stage: str, *, verified: set[str] | None = None) -> str:
        if verified is None:
            verified = set()
        if stage in verified:
            return sha256_file(self.root / stage / "stage_manifest.json")
        prerequisite_hashes: dict[str, str] = {}
        for prerequisite in STAGE_PREREQUISITES[stage]:
            prerequisite_hashes[prerequisite] = self._verify_stage(
                prerequisite, verified=verified
            )
        path = self.root / stage / "stage_manifest.json"
        if not path.is_file():
            raise ValueError(f"prerequisite stage {stage!r} has no SUCCESS marker")
        manifest = self._read_json_file(path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("status") != "SUCCESS"
            or manifest.get("config_hash") != self.config_hash
        ):
            raise ValueError(f"prerequisite stage {stage!r} has an invalid SUCCESS marker")
        if manifest.get("prerequisite_manifest_hashes") != prerequisite_hashes:
            raise ValueError(f"prerequisite manifest hash mismatch for stage {stage!r}")
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
                raise ValueError(
                    f"prerequisite output checksum mismatch: {relative_path}"
                )
        verified.add(stage)
        return sha256_file(path)

    def stage(
        self,
        stage: str,
        *,
        input_paths: Iterable[str | Path] = (),
        details: Mapping[str, object] | None = None,
        config_hash: str | None = None,
        now: Now = None,
    ) -> StageTransaction:
        """Validate prerequisites and create a transactional stage context."""

        if stage not in STAGE_PREREQUISITES:
            raise ValueError(f"unknown pipeline stage: {stage!r}")
        self._verify_base(expected_config_hash=config_hash)
        marker = self.root / stage / "stage_manifest.json"
        if marker.exists() or marker.is_symlink():
            raise FileExistsError(f"stage {stage!r} is already completed")
        prerequisite_hashes: dict[str, str] = {}
        for prerequisite in STAGE_PREREQUISITES[stage]:
            prerequisite_hashes[prerequisite] = self._verify_stage(prerequisite)
        inputs: dict[str, str] = {}
        for input_path in input_paths:
            target = self._validated_target(input_path)
            relative = target.relative_to(self.root).as_posix()
            if not target.is_file():
                raise FileNotFoundError(f"stage input is not a file: {relative}")
            inputs[relative] = sha256_file(target)
        return StageTransaction(
            store=self,
            stage=stage,
            input_hashes=inputs,
            prerequisite_manifest_hashes=prerequisite_hashes,
            details=dict(details or {}),
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
        prerequisite_manifest_hashes: dict[str, str],
        details: dict[str, object],
        clock: Clock,
    ) -> None:
        self._store = store
        self.stage = stage
        self._input_hashes = input_hashes
        self._prerequisite_manifest_hashes = prerequisite_manifest_hashes
        self._details = details
        self._clock = clock
        self._started_at: datetime | None = None
        self._temporary_root: Path | None = None
        self._outputs: dict[str, Path] = {}

    def __enter__(self) -> Self:
        self._started_at = self._clock()
        self._temporary_root = Path(
            tempfile.mkdtemp(prefix=f".stage-{self.stage}-", dir=self._store.root)
        )
        return self

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
        logical, temporary = self._temporary_target(relative_path)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self._outputs[logical] = temporary
        return self._store.root / logical

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
            shutil.rmtree(temporary_root, ignore_errors=True)
            return False
        published: list[PublicationOwnership] = []
        try:
            output_records = [
                {"path": logical, "sha256": sha256_file(temporary)}
                for logical, temporary in sorted(self._outputs.items())
            ]
            for logical, temporary in sorted(self._outputs.items()):
                destination = self._store._validated_target(logical)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._store._publish_temp(temporary, destination, published)
            ended_at = self._clock()
            manifest = {
                "config_hash": self._store.config_hash,
                "details": self._details,
                "ended_at": _timestamp(ended_at),
                "input_hashes": self._input_hashes,
                "outputs": output_records,
                "prerequisite_manifest_hashes": self._prerequisite_manifest_hashes,
                "schema_version": SCHEMA_VERSION,
                "stage": self.stage,
                "started_at": _timestamp(self._started_at),
                "status": "SUCCESS",
            }
            self._store._write_bytes(
                f"{self.stage}/stage_manifest.json",
                _canonical_json(manifest).encode("utf-8"),
                published,
            )
        except BaseException as error:
            self._store._rollback_publications(published, error)
            raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return False


__all__ = [
    "SCHEMA_VERSION",
    "STAGE_PREREQUISITES",
    "RunStore",
    "StageTransaction",
    "sha256_file",
]
