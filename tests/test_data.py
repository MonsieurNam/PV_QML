from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from test_config import valid_config
from typer.testing import CliRunner

import pv_pilot.services as services_module
from pv_pilot.artifacts import RunStore, sha256_file
from pv_pilot.cli import app
from pv_pilot.config import DataConfig, load_config
from pv_pilot.data import (
    CANONICAL_COLUMNS,
    EXCLUSION_COLUMNS,
    MAX_SITE_SPAN_DAYS,
    canonical_arrow_schema,
    canonicalize_data,
    dataframe_parquet_bytes,
    exclusion_arrow_schema,
)
from pv_pilot.domain import StageStatus
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import build_feature_splits


def data_config(
    *, cadence: int = 5, timezone: str = "UTC", availability_columns: bool = False
) -> DataConfig:
    payload = valid_config()["data"]
    payload["raw_cadence_minutes"] = cadence
    payload["timezone"] = timezone
    if availability_columns:
        payload["availability"] = {
            "power": {"column": "power_available"},
            "weather": {"column": "weather_available"},
        }
    else:
        payload["availability"] = {
            "power": {"fixed_latency_minutes": 1},
            "weather": {"fixed_latency_minutes": 2},
        }
    return DataConfig.model_validate(payload)


def frame_5m() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["A", "A", "A"],
            "timestamp": ["2025-01-01 00:00", "2025-01-01 00:05", "2025-01-01 00:10"],
            "power_kw": [1.0, 2.0, 3.0],
            "capacity_kw": [10.0, 10.0, 10.0],
            "ghi_w_m2": [100.0, 200.0, 300.0],
            "temperature_c": [20.0, 21.0, 22.0],
        }
    )


def write_input(
    path: Path, frame: pd.DataFrame, format_: Literal["csv", "parquet"]
) -> None:
    if format_ == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def test_five_minute_normalization_is_utc_stable_and_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    raw = frame_5m()
    before = raw.copy(deep=True)
    path = tmp_path / "data.csv"
    write_input(path, raw, "csv")

    result = canonicalize_data(path, data_config(timezone="Asia/Ho_Chi_Minh"), "csv")

    pd.testing.assert_frame_equal(raw, before)
    assert result.status is StageStatus.SUCCESS
    assert tuple(result.canonical.columns) == CANONICAL_COLUMNS
    assert result.canonical["interval_end_utc"].tolist() == [
        pd.Timestamp("2024-12-31T17:00:00Z"),
        pd.Timestamp("2024-12-31T17:05:00Z"),
        pd.Timestamp("2024-12-31T17:10:00Z"),
    ]
    assert result.canonical["power_available_at_utc"].iloc[0] == pd.Timestamp(
        "2024-12-31T17:01:00Z"
    )
    assert result.canonical["weather_available_at_utc"].iloc[0] == pd.Timestamp(
        "2024-12-31T17:02:00Z"
    )
    assert result.canonical["sample_id"].is_unique


@pytest.mark.parametrize("format_", ["csv", "parquet"])
def test_csv_and_parquet_adapters_have_equivalent_canonical_values(
    tmp_path: Path, format_: Literal["csv", "parquet"]
) -> None:
    reference_path = tmp_path / "reference.csv"
    frame_5m().to_csv(reference_path, index=False)
    candidate_path = tmp_path / f"candidate.{format_}"
    write_input(candidate_path, frame_5m(), format_)

    reference = canonicalize_data(reference_path, data_config(), "csv").canonical
    candidate = canonicalize_data(candidate_path, data_config(), format_).canonical

    pd.testing.assert_frame_equal(candidate, reference)


def test_one_minute_bins_are_right_closed_exact_five_point_means(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01 00:01", periods=6, freq="1min")
    frame = pd.DataFrame(
        {
            "site_id": "A",
            "timestamp": timestamps.astype(str),
            "power_kw": [1, 2, 3, 4, 5, 100],
            "capacity_kw": 10,
            "ghi_w_m2": [10, 20, 30, 40, 50, 600],
            "temperature_c": [20, 21, 22, 23, 24, 30],
        }
    )
    path = tmp_path / "minute.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(cadence=1), "csv")

    assert len(result.canonical) == 1
    row = result.canonical.iloc[0]
    assert row["interval_end_utc"] == pd.Timestamp("2025-01-01T00:05:00Z")
    assert row["ac_power_kw"] == 3.0
    assert row["ghi_w_m2"] == 30.0
    assert row["air_temp_c"] == 22.0
    assert "incomplete_bin" in set(result.exclusions["reason"])


def test_incomplete_one_minute_bin_excludes_every_constituent(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "site_id": "A",
            "timestamp": pd.date_range("2025-01-01 00:01", periods=4, freq="1min").astype(str),
            "power_kw": [1, 2, 3, 4],
            "capacity_kw": 10,
            "ghi_w_m2": [10, 20, 30, 40],
            "temperature_c": [20, 21, 22, 23],
        }
    )
    path = tmp_path / "incomplete.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(cadence=1), "csv")

    incomplete = result.exclusions[result.exclusions["reason"] == "incomplete_bin"]
    assert len(incomplete) == 4
    assert incomplete["source_row_id"].nunique() == 4
    assert result.accepted_input_row_count == 0
    assert result.excluded_input_row_count == 4
    assert result.input_row_count == (
        result.accepted_input_row_count + result.excluded_input_row_count
    )


def test_explicit_availability_uses_latest_constituent_and_preserves_delay(
    tmp_path: Path,
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["power_available"] = ["2025-01-01T00:08:00Z"]
    frame["weather_available"] = ["2025-01-01T00:30:00Z"]
    path = tmp_path / "delayed.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(
        path, data_config(availability_columns=True), "csv"
    )

    assert result.status is StageStatus.SUCCESS
    assert result.canonical["weather_available_at_utc"].iloc[0] == pd.Timestamp(
        "2025-01-01T00:30:00Z"
    )
    assert result.exclusions.empty


def test_nullable_daylight_is_excluded_without_ambiguous_boolean_error(
    tmp_path: Path,
) -> None:
    frame = frame_5m().iloc[:2].copy()
    frame["is_daylight"] = pd.Series([True, pd.NA], dtype="boolean")
    path = tmp_path / "nullable-daylight.parquet"
    frame.to_parquet(path, index=False)
    config = data_config()
    payload = config.model_dump()
    payload["columns"]["daylight"] = "is_daylight"

    result = canonicalize_data(path, DataConfig.model_validate(payload), "parquet")

    assert result.status is StageStatus.SUCCESS
    assert result.canonical["is_daylight"].tolist() == [True]
    assert result.exclusions["reason"].tolist() == ["invalid_daylight"]


@pytest.mark.parametrize(
    ("timestamp", "expected_reason"),
    [
        ("2025-11-02 01:30", "ambiguous_local_time"),
        ("2025-03-09 02:30", "nonexistent_local_time"),
    ],
)
def test_dst_ambiguous_and_nonexistent_local_times_are_excluded(
    tmp_path: Path, timestamp: str, expected_reason: str
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["timestamp"] = timestamp
    path = tmp_path / "dst.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(timezone="America/New_York"), "csv")

    assert result.status is StageStatus.INVALID
    assert result.canonical.empty
    assert result.exclusions["reason"].tolist() == [expected_reason]


def test_qc_excludes_duplicate_bad_weather_capacity_and_reports_gaps(
    tmp_path: Path,
) -> None:
    frame = frame_5m()
    duplicate = frame.iloc[[0]].copy()
    duplicate["power_kw"] = 9.0
    bad_weather = frame.iloc[[1]].copy()
    bad_weather["timestamp"] = "2025-01-01 00:15"
    bad_weather["ghi_w_m2"] = float("nan")
    bad_capacity = frame.iloc[[2]].copy()
    bad_capacity["timestamp"] = "2025-01-01 00:25"
    bad_capacity["capacity_kw"] = 11.0
    combined = pd.concat([frame, duplicate, bad_weather, bad_capacity], ignore_index=True)
    path = tmp_path / "bad.csv"
    combined.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(), "csv")

    reasons = set(result.exclusions["reason"])
    assert {"duplicate_interval", "missing_weather", "inconsistent_capacity"} <= reasons
    assert "missing_interval" in reasons
    assert not result.canonical[["ghi_w_m2", "air_temp_c"]].isna().any().any()


def test_off_grid_and_nonfinite_rows_are_excluded_without_interpolation(
    tmp_path: Path,
) -> None:
    frame = frame_5m()
    frame.loc[0, "timestamp"] = "2025-01-01 00:03"
    frame.loc[1, "power_kw"] = float("inf")
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(), "csv")

    assert set(result.exclusions["reason"]) >= {"off_grid", "nonfinite_numeric"}
    assert result.canonical["interval_end_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:10:00Z")
    ]


def test_five_minute_alignment_rejects_nanosecond_remainder(tmp_path: Path) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["timestamp"] = ["2025-01-01T00:05:00.000000001Z"]
    path = tmp_path / "nanosecond.parquet"
    frame.to_parquet(path, index=False)

    result = canonicalize_data(path, data_config(), "parquet")

    assert result.status is StageStatus.INVALID
    assert result.exclusions["reason"].tolist() == ["off_grid"]


def test_reserved_working_prefix_is_rejected_in_configured_mapping() -> None:
    payload = valid_config()["data"]
    payload["columns"]["site"] = "_pv_pilot_source_row_id"

    with pytest.raises(ValueError, match="reserved"):
        DataConfig.model_validate(payload)


def test_missing_site_and_timestamp_are_safely_excluded(tmp_path: Path) -> None:
    frame = frame_5m().iloc[:2].copy()
    frame.loc[0, "site_id"] = None
    frame.loc[0, "timestamp"] = None
    path = tmp_path / "missing-identifiers.csv"
    frame.to_csv(path, index=False)

    result = canonicalize_data(path, data_config(), "csv")

    assert result.status is StageStatus.SUCCESS
    assert "invalid_timestamp" in set(result.exclusions["reason"])
    assert result.canonical["site_id"].tolist() == ["A"]


def make_run(
    tmp_path: Path,
    frame: pd.DataFrame | None,
    *,
    format_: Literal["csv", "parquet"] = "csv",
    availability_columns: bool = False,
    max_wall_seconds: int = 3600,
    max_memory_gb: float = 8.0,
) -> tuple[RunStore, Path]:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    config_dir = project / "configs"
    data_dir = project / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    payload = valid_config()
    payload["protocol"] = {
        "version": "1.0",
        "provenance": "synthetic",
        "approved_by": None,
        "approved_at": None,
    }
    payload["paths"] = {
        "input_data": f"../data/input.{format_}",
        "artifact_dir": "../artifacts",
    }
    payload["data"]["format"] = format_
    payload["data"]["timezone"] = "UTC"
    if availability_columns:
        payload["data"]["availability"] = {
            "power": {"column": "power_available"},
            "weather": {"column": "weather_available"},
        }
    else:
        payload["data"]["availability"] = {
            "power": {"fixed_latency_minutes": 0},
            "weather": {"fixed_latency_minutes": 0},
        }
    payload["resources"]["max_wall_clock_seconds"] = max_wall_seconds
    payload["resources"]["max_memory_gb"] = max_memory_gb
    config_path = config_dir / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    input_path = data_dir / f"input.{format_}"
    if frame is not None:
        write_input(input_path, frame, format_)
    loaded = load_config(config_path, project_root=project)
    return RunStore.create(
        loaded, now=datetime(2026, 9, 29, 1, 2, 3, tzinfo=UTC)
    ), input_path


def read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_validate_data_service_writes_atomic_success_artifacts_and_resumes(
    tmp_path: Path,
) -> None:
    store, _ = make_run(tmp_path, frame_5m())

    first = runner.invoke(app, ["validate-data", "--run-dir", str(store.root)])
    second = runner.invoke(app, ["validate-data", "--run-dir", str(store.root)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already successful" in second.output.lower()
    for name in (
        "canonical.parquet",
        "qc_exclusions.parquet",
        "data_card.json",
        "cleaning_ledger.json",
        "stage_result.json",
        "stage_manifest.json",
    ):
        assert (store.root / "validate-data" / name).is_file()
    manifest = read_json(store.root / "validate-data/stage_manifest.json")
    result = read_json(store.root / "validate-data/stage_result.json")
    card = read_json(store.root / "validate-data/data_card.json")
    assert manifest["status"] == "SUCCESS"
    assert result["status"] == "success"
    assert card["provenance"] == "synthetic"
    assert card["scientific_decision"] == "NOT_EVALUATED"
    assert card["input_sha256"] == read_json(store.root / "source_manifest.json")["input"]["sha256"]


def test_missing_source_is_durable_blocked_not_evaluated(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, None)

    result = runner.invoke(app, ["validate-data", "--run-dir", str(store.root)])

    assert result.exit_code == 2, result.output
    stage = read_json(store.root / "validate-data/stage_result.json")
    manifest = read_json(store.root / "validate-data/stage_manifest.json")
    assert stage["status"] == "blocked"
    assert stage["details"]["scientific_decision"] == "NOT_EVALUATED"
    assert manifest["status"] == "BLOCKED"


def test_resume_rejects_tampered_blocked_stage_outputs(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, None)
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.BLOCKED
    result_path = store.root / "validate-data/stage_result.json"
    payload = read_json(result_path)
    payload["message"] = "tampered"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        service.validate_data(store.root)


def test_result_status_tamper_fails_checksum_before_semantic_check(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    result_path = store.root / "validate-data/stage_result.json"
    payload = read_json(result_path)
    payload["status"] = "blocked"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="output checksum"):
        service.validate_data(store.root)


def test_valid_checksums_cannot_hide_result_marker_status_mismatch(
    tmp_path: Path,
) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    result_path = store.root / "validate-data/stage_result.json"
    payload = read_json(result_path)
    payload["status"] = "blocked"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    marker_path = store.root / "validate-data/stage_manifest.json"
    marker = read_json(marker_path)
    for output in marker["outputs"]:
        if output["path"] == "validate-data/stage_result.json":
            output["sha256"] = sha256_file(result_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal result status"):
        service.validate_data(store.root)
    with pytest.raises(ValueError, match="terminal result status"):
        RunStore.open(store.root).stage("build-splits")


def test_removing_terminal_declaration_cannot_hide_rechecksummed_status_mismatch(
    tmp_path: Path,
) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    result_path = store.root / "validate-data/stage_result.json"
    result = read_json(result_path)
    result["status"] = "blocked"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    marker_path = store.root / "validate-data/stage_manifest.json"
    marker = read_json(marker_path)
    marker.pop("terminal_result")
    for output in marker["outputs"]:
        if output["path"] == "validate-data/stage_result.json":
            output["sha256"] = sha256_file(result_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal result status"):
        service.validate_data(store.root)
    with pytest.raises(ValueError, match="terminal result status"):
        RunStore.open(store.root).stage("build-splits")


def test_matching_conventional_result_still_requires_declaration(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    marker_path = store.root / "validate-data/stage_manifest.json"
    marker = read_json(marker_path)
    marker.pop("terminal_result")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="declaration is required"):
        service.validate_data(store.root)


def test_existing_result_removed_from_marker_outputs_is_rejected(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    marker_path = store.root / "validate-data/stage_manifest.json"
    marker = read_json(marker_path)
    marker.pop("terminal_result")
    marker["outputs"] = [
        output
        for output in marker["outputs"]
        if output["path"] != "validate-data/stage_result.json"
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="committed stage output"):
        service.validate_data(store.root)
    with pytest.raises(ValueError, match="committed stage output"):
        RunStore.open(store.root).stage("build-splits")


def test_duplicate_or_inconsistent_terminal_result_declaration_is_rejected(
    tmp_path: Path,
) -> None:
    store, _ = make_run(tmp_path, frame_5m())
    assert DefaultPilotServices().validate_data(store.root).status is StageStatus.SUCCESS
    marker_path = store.root / "validate-data/stage_manifest.json"
    marker = read_json(marker_path)
    result_record = next(
        output
        for output in marker["outputs"]
        if output["path"] == "validate-data/stage_result.json"
    )
    marker["outputs"].append(dict(result_record))
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        RunStore.open(store.root).verify_terminal_stage("validate-data")


def test_source_tamper_after_init_is_durable_invalid(tmp_path: Path) -> None:
    store, input_path = make_run(tmp_path, frame_5m())
    original = sha256_file(input_path)
    input_path.write_text("changed\n", encoding="utf-8")

    result = runner.invoke(app, ["validate-data", "--run-dir", str(store.root)])

    assert result.exit_code == 2, result.output
    stage = read_json(store.root / "validate-data/stage_result.json")
    assert stage["status"] == "invalid"
    assert stage["details"]["expected_sha256"] == original
    assert stage["details"]["scientific_decision"] == "NOT_EVALUATED"


def test_source_deleted_after_available_init_is_invalid_not_blocked(tmp_path: Path) -> None:
    store, input_path = make_run(tmp_path, frame_5m())
    input_path.unlink()

    result = DefaultPilotServices().validate_data(store.root)

    assert result.status is StageStatus.INVALID
    assert "checksum" in (result.message or "")
    assert read_json(store.root / "validate-data/stage_manifest.json")["status"] == "INVALID"


@pytest.mark.parametrize("tamper", ["mutate", "delete"])
def test_source_change_during_serialization_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    store, input_path = make_run(tmp_path, frame_5m())
    original_serializer = services_module.dataframe_parquet_bytes
    mutated = False

    def mutate_then_serialize(
        frame: pd.DataFrame, *, kind: Literal["canonical", "exclusions"]
    ) -> bytes:
        nonlocal mutated
        if not mutated:
            mutated = True
            if tamper == "mutate":
                input_path.write_text("mutated during publication\n", encoding="utf-8")
            else:
                input_path.unlink()
        return original_serializer(frame, kind=kind)

    monkeypatch.setattr(services_module, "dataframe_parquet_bytes", mutate_then_serialize)

    with pytest.raises(ValueError, match="external stage input changed"):
        DefaultPilotServices().validate_data(store.root)

    assert mutated
    assert not (store.root / "validate-data/stage_manifest.json").exists()
    assert not (store.root / "validate-data/canonical.parquet").exists()


def test_success_manifest_tracks_and_revalidates_external_source(tmp_path: Path) -> None:
    store, input_path = make_run(tmp_path, frame_5m())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    manifest = read_json(store.root / "validate-data/stage_manifest.json")

    assert manifest["external_input_hashes"] == [
        {"path": input_path.resolve().as_posix(), "sha256": sha256_file(input_path)}
    ]
    input_path.write_text("changed after success\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external input checksum"):
        service.validate_data(store.root)


def test_empty_exclusion_parquet_has_stable_schema(tmp_path: Path) -> None:
    path = tmp_path / "good.csv"
    frame_5m().to_csv(path, index=False)
    result = canonicalize_data(path, data_config(), "csv")

    assert tuple(result.exclusions.columns) == EXCLUSION_COLUMNS
    assert str(result.exclusions["source_row_id"].dtype) == "string"
    assert str(result.exclusions["source_timestamp_utc"].dtype) == "datetime64[ns, UTC]"


def test_parquet_physical_schemas_are_stable_for_empty_and_nonempty_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "good.csv"
    frame_5m().to_csv(path, index=False)
    nonempty = canonicalize_data(path, data_config(), "csv")
    empty_path = tmp_path / "empty-invalid.csv"
    invalid = frame_5m().iloc[:1].copy()
    invalid["ghi_w_m2"] = ""
    invalid.to_csv(empty_path, index=False)
    empty = canonicalize_data(empty_path, data_config(), "csv")

    canonical_schemas = {
        pq.read_schema(pa.BufferReader(dataframe_parquet_bytes(frame, kind="canonical")))
        for frame in (nonempty.canonical, empty.canonical)
    }
    exclusion_schemas = {
        pq.read_schema(pa.BufferReader(dataframe_parquet_bytes(frame, kind="exclusions")))
        for frame in (nonempty.exclusions, empty.exclusions)
    }

    assert canonical_schemas == {canonical_arrow_schema(include_daylight=False)}
    assert exclusion_schemas == {exclusion_arrow_schema()}
    assert canonical_arrow_schema(False).field("quality_flags").type.value_type == pa.string()
    assert canonical_arrow_schema(False).field("interval_end_utc").type == pa.timestamp(
        "ns", tz="UTC"
    )


def test_nanosecond_availability_is_preserved_in_canonical_parquet(tmp_path: Path) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["power_available"] = ["2025-01-01T00:01:00.000000001Z"]
    frame["weather_available"] = ["2025-01-01T00:02:00.000000002Z"]
    path = tmp_path / "availability.parquet"
    frame.to_parquet(path, index=False)

    result = canonicalize_data(
        path, data_config(availability_columns=True), "parquet"
    )
    payload = dataframe_parquet_bytes(result.canonical, kind="canonical")
    roundtrip = pq.read_table(pa.BufferReader(payload)).to_pandas()

    assert roundtrip["power_available_at_utc"].iloc[0].nanosecond == 1
    assert roundtrip["weather_available_at_utc"].iloc[0].nanosecond == 2


def test_off_grid_nanosecond_exclusion_serializes_without_precision_loss(
    tmp_path: Path,
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["timestamp"] = ["2025-01-01T00:05:00.000000001Z"]
    path = tmp_path / "off-grid.parquet"
    frame.to_parquet(path, index=False)

    result = canonicalize_data(path, data_config(), "parquet")
    payload = dataframe_parquet_bytes(result.exclusions, kind="exclusions")
    roundtrip = pq.read_table(pa.BufferReader(payload)).to_pandas()

    assert roundtrip["source_timestamp_utc"].iloc[0].nanosecond == 1


@pytest.mark.parametrize("format_", ["csv", "parquet"])
@pytest.mark.parametrize("daylight", [0, 1, 0.0, 1.0])
def test_integral_numeric_daylight_has_csv_parquet_parity(
    tmp_path: Path,
    format_: Literal["csv", "parquet"],
    daylight: int | float,
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["is_daylight"] = daylight
    path = tmp_path / f"daylight.{format_}"
    write_input(path, frame, format_)
    payload = data_config().model_dump()
    payload["columns"]["daylight"] = "is_daylight"

    result = canonicalize_data(path, DataConfig.model_validate(payload), format_)

    assert result.status is StageStatus.SUCCESS
    assert result.canonical["is_daylight"].tolist() == [bool(daylight)]


def test_csv_identifier_na_is_preserved_and_matches_parquet(tmp_path: Path) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["site_id"] = "NA"
    csv_path = tmp_path / "identifier.csv"
    parquet_path = tmp_path / "identifier.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    csv_result = canonicalize_data(csv_path, data_config(), "csv")
    parquet_result = canonicalize_data(parquet_path, data_config(), "parquet")

    assert csv_result.canonical["site_id"].tolist() == ["NA"]
    pd.testing.assert_frame_equal(csv_result.canonical, parquet_result.canonical)


def test_csv_duplicate_quoted_or_bom_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text(
        '\ufeff"site_id","timestamp","power_kw","capacity_kw","ghi_w_m2",'
        '"temperature_c","site_id"\nA,2025-01-01T00:00:00Z,1,10,100,20,A\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate CSV header"):
        canonicalize_data(path, data_config(), "csv")


def test_csv_leading_blank_records_do_not_bypass_duplicate_header_check(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leading-blank-duplicate.csv"
    path.write_text(
        '\ufeff\n\n"site_id","timestamp","power_kw","capacity_kw","ghi_w_m2",'
        '"temperature_c","site_id"\nA,2025-01-01T00:00:00Z,1,10,100,20,A\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate CSV header"):
        canonicalize_data(path, data_config(), "csv")


def test_resource_deadline_across_serializers_publishes_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = make_run(tmp_path, frame_5m(), max_wall_seconds=1)
    original = services_module.dataframe_parquet_bytes

    def slow_serializer(
        frame: pd.DataFrame, *, kind: Literal["canonical", "exclusions"]
    ) -> bytes:
        time.sleep(0.7)
        return original(frame, kind=kind)

    monkeypatch.setattr(services_module, "dataframe_parquet_bytes", slow_serializer)

    result = DefaultPilotServices().validate_data(store.root)
    ledger = read_json(store.root / "validate-data/cleaning_ledger.json")

    assert result.status is StageStatus.INVALID
    assert ledger["resource_status"] == "EXCEEDED"
    assert ledger["elapsed_seconds"] >= 1
    assert read_json(store.root / "validate-data/stage_manifest.json")["status"] == "INVALID"


def test_resource_memory_guard_publishes_durable_invalid(tmp_path: Path) -> None:
    store, _ = make_run(tmp_path, frame_5m(), max_memory_gb=0.000001)

    result = DefaultPilotServices().validate_data(store.root)
    ledger = read_json(store.root / "validate-data/cleaning_ledger.json")

    assert result.status is StageStatus.INVALID
    assert ledger["resource_status"] == "EXCEEDED"
    assert ledger["peak_rss_bytes_observed"] > 0


@pytest.mark.parametrize("format_", ["csv", "parquet"])
@pytest.mark.parametrize("year", [1500, 2500])
def test_out_of_nanosecond_interval_is_durable_invalid(
    tmp_path: Path, format_: Literal["csv", "parquet"], year: int
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["timestamp"] = [f"{year}-01-01T00:00:00Z"]
    store, _ = make_run(tmp_path, frame, format_=format_)

    result = DefaultPilotServices().validate_data(store.root)
    exclusions = pd.read_parquet(store.root / "validate-data/qc_exclusions.parquet")

    assert result.status is StageStatus.INVALID
    assert exclusions["reason"].tolist() == ["invalid_timestamp"]
    assert exclusions["source_timestamp_utc"].isna().all()
    assert read_json(store.root / "validate-data/stage_manifest.json")["status"] == "INVALID"


@pytest.mark.parametrize("format_", ["csv", "parquet"])
@pytest.mark.parametrize("year", [1500, 2500])
def test_out_of_nanosecond_availability_is_durable_invalid(
    tmp_path: Path, format_: Literal["csv", "parquet"], year: int
) -> None:
    frame = frame_5m().iloc[:1].copy()
    frame["power_available"] = [f"{year}-01-01T00:00:00Z"]
    frame["weather_available"] = ["2025-01-01T00:00:00Z"]
    store, _ = make_run(
        tmp_path, frame, format_=format_, availability_columns=True
    )

    result = DefaultPilotServices().validate_data(store.root)
    exclusions = pd.read_parquet(store.root / "validate-data/qc_exclusions.parquet")

    assert result.status is StageStatus.INVALID
    assert exclusions["reason"].tolist() == ["invalid_availability"]
    assert read_json(store.root / "validate-data/stage_manifest.json")["status"] == "INVALID"


def test_resource_guard_covers_atomic_stage_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = make_run(tmp_path, frame_5m(), max_wall_seconds=1)
    original_publish = RunStore._publish_temp

    def slow_publish(
        self: RunStore,
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        time.sleep(0.22)
        original_publish(self, source, destination, publications)

    monkeypatch.setattr(RunStore, "_publish_temp", slow_publish)

    first = DefaultPilotServices().validate_data(store.root)
    resumed = DefaultPilotServices().validate_data(store.root)
    ledger = read_json(store.root / "validate-data/cleaning_ledger.json")

    assert first.status is StageStatus.INVALID
    assert resumed.status is StageStatus.INVALID
    assert ledger["resource_status"] == "EXCEEDED"
    assert ledger["elapsed_seconds"] >= 1
    assert read_json(store.root / "validate-data/stage_manifest.json")["status"] == "INVALID"
    assert not list(store.root.rglob(".rollback-*"))
    assert not list(store.root.glob(".publication-abort-*"))
    assert not list(store.root.glob(".stage-*"))


def test_hundred_year_timestamp_typo_is_bounded_and_invalid(tmp_path: Path) -> None:
    frame = frame_5m().iloc[:2].copy()
    frame["timestamp"] = ["1925-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    path = tmp_path / "century.csv"
    frame.to_csv(path, index=False)
    started = time.perf_counter()

    with pytest.raises(ValueError, match="maximum site span"):
        canonicalize_data(path, data_config(), "csv")

    assert time.perf_counter() - started < 2
    assert MAX_SITE_SPAN_DAYS < 36500


def test_hundred_year_typo_service_is_durable_invalid_with_bounded_ledger(
    tmp_path: Path,
) -> None:
    frame = frame_5m().iloc[:2].copy()
    frame["timestamp"] = ["1925-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    store, _ = make_run(tmp_path, frame)

    result = DefaultPilotServices().validate_data(store.root)
    ledger = read_json(store.root / "validate-data/cleaning_ledger.json")
    card = read_json(store.root / "validate-data/data_card.json")

    assert result.status is StageStatus.INVALID
    assert ledger["resource_status"] == "WITHIN_LIMIT"
    assert card["excluded_row_count"] == 0
    assert card["missing_expected_interval_count"] == 0


runner = CliRunner()


def test_repository_synthetic_fixture_exists_and_spans_frozen_splits() -> None:
    root = Path(__file__).parents[1]
    loaded = load_config(root / "configs/pilot.synthetic.yaml", project_root=root)

    assert loaded.resolved_input_path.is_file()
    result = canonicalize_data(
        loaded.resolved_input_path, loaded.config.data, loaded.config.data.format
    )

    assert result.status is StageStatus.SUCCESS
    timestamps = result.canonical["interval_end_utc"]
    assert loaded.resolved_input_path.name == "synthetic_pv_compact.csv"
    assert len(result.canonical) == 45 * 16
    assert timestamps.min() == pd.Timestamp("2025-01-01T08:00:00Z")
    assert timestamps.max() == pd.Timestamp("2025-02-14T09:15:00Z")
    assert timestamps.dt.floor("D").nunique() == 45

    split_result = build_feature_splits(result.canonical, loaded.config)

    assert split_result.status is StageStatus.SUCCESS
    counts = split_result.features.groupby("split").size().to_dict()
    assert counts == {"train": 15, "validation": 10, "test": 20}
    assert split_result.features["issued_at_utc"].dt.floor("D").nunique() == 45
