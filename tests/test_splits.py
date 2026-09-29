from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from test_config import valid_config

from pv_pilot.artifacts import RunStore
from pv_pilot.config import PilotConfig, load_config
from pv_pilot.domain import StageStatus
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import (
    MATCHED4_COLUMNS,
    MATCHED6_COLUMNS,
    XGB_BASE_COLUMNS,
    build_feature_splits,
    feature_dataframe_parquet_bytes,
    split_exclusion_dataframe_parquet_bytes,
)


def pilot_config(*, daylight: str = "column") -> PilotConfig:
    payload = valid_config()
    payload["split"] = {
        "train_end_utc": "2025-01-01T02:00:00Z",
        "validation_end_utc": "2025-01-01T03:00:00Z",
        "test_end_utc": "2025-01-01T04:00:00Z",
    }
    payload["data"]["timezone"] = "UTC"
    if daylight == "column":
        payload["data"]["columns"]["daylight"] = "is_daylight"
        payload["daylight"] = {"source": "column"}
    else:
        payload["data"]["columns"]["daylight"] = None
        payload["daylight"] = {
            "source": "solar_elevation",
            "latitude": 10.8231,
            "longitude": 106.6297,
        }
    return PilotConfig.model_validate(payload)


def canonical_frame(
    *,
    start: str = "2025-01-01T00:00:00Z",
    periods: int = 55,
    sites: tuple[str, ...] = ("A",),
    daylight: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    times = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    for site_index, site in enumerate(sites):
        for index, timestamp in enumerate(times):
            rows.append(
                {
                    "sample_id": f"{site}-{index:04d}",
                    "site_id": site,
                    "interval_end_utc": timestamp,
                    "power_available_at_utc": timestamp,
                    "weather_available_at_utc": timestamp,
                    "ac_power_kw": float(site_index * 1000 + index),
                    "capacity_kw": 2000.0,
                    "ghi_w_m2": float(100 + index),
                    "air_temp_c": float(20 + index / 10),
                    "quality_flags": (),
                    "is_daylight": daylight,
                }
            )
    return pd.DataFrame(rows)


def test_exact_target_split_purge_and_feature_orders() -> None:
    result = build_feature_splits(canonical_frame(), pilot_config())

    assert result.status is StageStatus.SUCCESS
    assert result.feature_manifest["matched4_columns"] == list(MATCHED4_COLUMNS)
    assert result.feature_manifest["matched6_columns"] == list(MATCHED6_COLUMNS)
    assert result.feature_manifest["xgb_columns"] == list(XGB_BASE_COLUMNS)
    row = result.features.set_index("issued_at_utc").loc[
        pd.Timestamp("2025-01-01T01:30:00Z")
    ]
    assert row["target_at_utc"] == pd.Timestamp("2025-01-01T01:45:00Z")
    assert row["target_ac_power_kw"] == 21.0
    assert row["power_lag_15m_kw"] == 15.0
    assert row["power_lag_60m_kw"] == 6.0
    assert set(result.features["split"]) == {"train", "validation", "test"}
    assert not (
        (result.features["split"] == "train")
        & (result.features["target_at_utc"] > pd.Timestamp("2025-01-01T02:00:00Z"))
    ).any()
    purged = result.exclusions[result.exclusions["reason"] == "split_boundary_purge"]
    assert pd.Timestamp("2025-01-01T01:50:00Z") in set(purged["issued_at_utc"])
    assert pd.Timestamp("2025-01-01T01:45:00Z") in set(result.features["issued_at_utc"])
    assert pd.Timestamp("2025-01-01T02:00:00Z") in set(purged["issued_at_utc"])


def test_absolute_lags_never_bridge_gaps_or_sites() -> None:
    frame = canonical_frame(sites=("A", "B"))
    frame = frame[
        ~(
            (frame["site_id"] == "A")
            & (frame["interval_end_utc"] == pd.Timestamp("2025-01-01T00:25:00Z"))
        )
    ].copy()

    result = build_feature_splits(frame, pilot_config())

    excluded = result.exclusions.set_index(["site_id", "issued_at_utc"])
    assert excluded.loc[("A", pd.Timestamp("2025-01-01T00:30:00Z")), "reason"] == (
        "missing_power_lag"
    )
    b_row = result.features.set_index(["site_id", "issued_at_utc"]).loc[
        ("B", pd.Timestamp("2025-01-01T01:30:00Z"))
    ]
    assert b_row["power_lag_60m_kw"] == 1006.0


def test_availability_equality_is_accepted_and_delays_are_excluded() -> None:
    frame = canonical_frame()
    delayed_current = pd.Timestamp("2025-01-01T01:25:00Z")
    frame.loc[
        frame["interval_end_utc"] == pd.Timestamp("2025-01-01T01:20:00Z"),
        "weather_available_at_utc",
    ] = delayed_current
    frame.loc[
        frame["interval_end_utc"] == pd.Timestamp("2025-01-01T01:25:00Z"),
        "power_available_at_utc",
    ] = pd.Timestamp("2025-01-01T01:41:00Z")
    # Equality at issue time must remain usable.
    frame.loc[
        frame["interval_end_utc"] == pd.Timestamp("2025-01-01T01:15:00Z"),
        "weather_available_at_utc",
    ] = pd.Timestamp("2025-01-01T01:15:00Z")
    # Target availability is deliberately late and must not affect the sample.
    frame.loc[
        frame["interval_end_utc"] == pd.Timestamp("2025-01-01T01:30:00Z"),
        "power_available_at_utc",
    ] = pd.Timestamp("2025-01-02T00:00:00Z")

    result = build_feature_splits(frame, pilot_config())
    reasons = result.exclusions.set_index("issued_at_utc")["reason"]

    assert reasons[pd.Timestamp("2025-01-01T01:20:00Z")] == "delayed_weather"
    assert reasons[pd.Timestamp("2025-01-01T01:40:00Z")] == "delayed_power_lag"
    sample = result.features.set_index("issued_at_utc").loc[
        pd.Timestamp("2025-01-01T01:15:00Z")
    ]
    assert sample["target_at_utc"] == pd.Timestamp("2025-01-01T01:30:00Z")


def test_scalers_fit_train_only_clip_extremes_and_record_rates() -> None:
    frame = canonical_frame()
    frame.loc[
        frame["interval_end_utc"] >= pd.Timestamp("2025-01-01T02:05:00Z"),
        "ghi_w_m2",
    ] = 100_000.0

    result = build_feature_splits(frame, pilot_config())
    scaler = result.scalers["matched4"]

    assert scaler["fitted_split"] == "train"
    assert scaler["max"]["ghi_w_m2"] < 100_000.0
    assert result.features.loc[
        result.features["split"] == "validation", "matched4_scaled_2"
    ].eq(1.0).all()
    matched4_rates = result.clip_rates["profiles"]["matched4"]["rates"]
    assert matched4_rates["validation"]["ghi_w_m2"] == 1.0
    assert scaler["train_sample_count"] == result.feature_manifest["split_counts"]["train"]
    assert scaler["train_sample_sha256"]


def test_constant_train_feature_scales_to_zero_without_refitting() -> None:
    frame = canonical_frame()
    frame.loc[
        frame["interval_end_utc"] <= pd.Timestamp("2025-01-01T02:00:00Z"),
        "air_temp_c",
    ] = 25.0
    frame.loc[
        frame["interval_end_utc"] > pd.Timestamp("2025-01-01T02:00:00Z"),
        "air_temp_c",
    ] = 99.0

    result = build_feature_splits(frame, pilot_config())

    assert "air_temp_c" in result.scalers["matched4"]["constant_features"]
    assert result.features["matched4_scaled_3"].eq(0.0).all()
    profile = result.clip_rates["profiles"]["matched4"]
    assert profile["rates"]["train"]["air_temp_c"] == 0.0
    assert profile["rates"]["validation"]["air_temp_c"] == 1.0
    assert profile["rates"]["test"]["air_temp_c"] == 1.0
    assert profile["counts"]["validation"]["air_temp_c"] == (
        result.feature_manifest["split_counts"]["validation"]
    )
    assert result.clip_rates["out_of_domain_policy"] == {
        "comparison": "raw_value < train_min or raw_value > train_max",
        "constant_feature": "raw_value != train_constant",
        "equality_is_out_of_domain": False,
    }
    assert result.scalers["matched4"]["constant_feature_policy"] == {
        "out_of_domain_rule": "raw_value != train_constant",
        "scaled_value": 0.0,
    }


def test_constant_feature_ood_counts_exact_and_deviating_values() -> None:
    frame = canonical_frame()
    train_mask = frame["interval_end_utc"] <= pd.Timestamp("2025-01-01T02:00:00Z")
    validation_mask = (
        (frame["interval_end_utc"] > pd.Timestamp("2025-01-01T02:00:00Z"))
        & (frame["interval_end_utc"] <= pd.Timestamp("2025-01-01T03:00:00Z"))
    )
    frame.loc[train_mask, "air_temp_c"] = 25.0
    validation_indices = frame.index[validation_mask]
    frame.loc[validation_indices, "air_temp_c"] = [
        25.0 if position % 2 == 0 else 99.0
        for position in range(len(validation_indices))
    ]

    result = build_feature_splits(frame, pilot_config())
    validation = result.features[result.features["split"] == "validation"]
    expected_count = int(validation["air_temp_c"].ne(25.0).sum())
    profile = result.clip_rates["profiles"]["matched4"]

    assert validation["matched4_scaled_3"].eq(0.0).all()
    assert profile["counts"]["validation"]["air_temp_c"] == expected_count
    assert profile["rates"]["validation"]["air_temp_c"] == pytest.approx(
        expected_count / len(validation)
    )


def test_resource_memory_scans_are_constant_count_for_large_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pd.DataFrame.memory_usage
    calls = 0

    def counted_memory_usage(self: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.Series:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "memory_usage", counted_memory_usage)
    build_feature_splits(
        canonical_frame(periods=4096),
        pilot_config(),
        resource_checkpoint=lambda declared: None,
    )
    calls_for_4096 = calls
    calls = 0
    build_feature_splits(
        canonical_frame(periods=8192),
        pilot_config(),
        resource_checkpoint=lambda declared: None,
    )

    assert calls_for_4096 <= 4
    assert calls == calls_for_4096


def test_column_daylight_excludes_false_and_missing_fail_closed() -> None:
    frame = canonical_frame()
    frame["is_daylight"] = frame["is_daylight"].astype("boolean")
    frame.loc[frame.index[15], "is_daylight"] = False
    frame.loc[frame.index[16], "is_daylight"] = pd.NA

    result = build_feature_splits(frame, pilot_config())
    reasons = result.exclusions.set_index("issued_at_utc")["reason"]

    assert reasons[frame.loc[15, "interval_end_utc"]] == "not_daylight"
    assert reasons[frame.loc[16, "interval_end_utc"]] == "missing_daylight"


def test_solar_mode_adds_geometry_only_to_xgb() -> None:
    frame = canonical_frame(start="2025-01-01T00:00:00Z", periods=50).drop(
        columns="is_daylight"
    )
    result = build_feature_splits(frame, pilot_config(daylight="solar"))

    xgb_columns = result.feature_manifest["xgb_columns"]
    assert xgb_columns == [*XGB_BASE_COLUMNS, "solar_elevation_deg", "solar_azimuth_deg"]
    assert "solar_elevation_deg" in result.features
    assert "solar_azimuth_deg" in result.features
    assert result.feature_manifest["solar_geometry"]["library"] == "pvlib"
    assert result.features["solar_elevation_deg"].gt(0.0).all()
    assert not set(("solar_elevation_deg", "solar_azimuth_deg")) & set(MATCHED6_COLUMNS)


def test_two_sites_have_independent_targets_lags_and_scaler_order() -> None:
    result = build_feature_splits(canonical_frame(sites=("B", "A")), pilot_config())

    rows = result.features.set_index(["site_id", "issued_at_utc"])
    issued_at = pd.Timestamp("2025-01-01T01:30:00Z")
    assert rows.loc[("A", issued_at), "target_ac_power_kw"] == 1021.0
    assert rows.loc[("B", issued_at), "target_ac_power_kw"] == 21.0
    assert rows.loc[("A", issued_at), "power_lag_60m_kw"] == 1006.0
    assert rows.loc[("B", issued_at), "power_lag_60m_kw"] == 6.0
    assert result.features[["site_id", "issued_at_utc"]].values.tolist() == sorted(
        result.features[["site_id", "issued_at_utc"]].values.tolist(),
        key=lambda value: (value[0], value[1]),
    )


def test_feature_and_exclusion_parquet_schemas_are_stable_when_empty() -> None:
    successful = build_feature_splits(canonical_frame(), pilot_config())
    invalid = build_feature_splits(canonical_frame(periods=14), pilot_config())

    successful_feature_schema = pq.read_schema(
        pa.BufferReader(feature_dataframe_parquet_bytes(successful.features, pilot_config()))
    )
    invalid_feature_schema = pq.read_schema(
        pa.BufferReader(feature_dataframe_parquet_bytes(invalid.features, pilot_config()))
    )
    successful_exclusion_schema = pq.read_schema(
        pa.BufferReader(split_exclusion_dataframe_parquet_bytes(successful.exclusions))
    )
    empty_exclusion_schema = pq.read_schema(
        pa.BufferReader(
            split_exclusion_dataframe_parquet_bytes(successful.exclusions.iloc[0:0])
        )
    )

    assert successful_feature_schema == invalid_feature_schema
    assert successful_exclusion_schema == empty_exclusion_schema
    assert successful_feature_schema.field("issued_at_utc").type == pa.timestamp(
        "ns", tz="UTC"
    )
    assert successful_exclusion_schema.field("reason").type == pa.string()


def make_run(tmp_path: Path, frame: pd.DataFrame) -> RunStore:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    payload = valid_config()
    payload["protocol"] = {
        "version": "1.0",
        "provenance": "synthetic",
        "approved_by": None,
        "approved_at": None,
    }
    payload["paths"] = {
        "input_data": "../data/input.csv",
        "artifact_dir": "../artifacts",
    }
    payload["data"]["timezone"] = "UTC"
    payload["data"]["availability"] = {
        "power": {"fixed_latency_minutes": 0},
        "weather": {"fixed_latency_minutes": 0},
    }
    payload["split"] = pilot_config().split.model_dump(mode="json")
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    raw = frame.rename(
        columns={
            "interval_end_utc": "timestamp",
            "ac_power_kw": "power_kw",
            "air_temp_c": "temperature_c",
        }
    )[
        [
            "site_id",
            "timestamp",
            "power_kw",
            "capacity_kw",
            "ghi_w_m2",
            "temperature_c",
            "is_daylight",
        ]
    ]
    raw.to_csv(project / "data/input.csv", index=False)
    loaded = load_config(config_path, project_root=project)
    return RunStore.create(loaded, now=datetime(2026, 9, 29, tzinfo=UTC))


def read_json(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_build_splits_service_publishes_and_resumes(tmp_path: Path) -> None:
    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS

    first = service.build_splits(store.root)
    second = service.build_splits(store.root)

    assert first.status is StageStatus.SUCCESS
    assert second.status is StageStatus.SUCCESS
    assert "already successful" in (second.message or "")
    for name in (
        "feature_matrix.parquet",
        "development_features.parquet",
        "holdout_features.parquet",
        "exclusions.parquet",
        "feature_manifest.json",
        "minmax_scalers.json",
        "clip_rates.json",
        "stage_result.json",
        "resource_ledger.json",
        "stage_manifest.json",
    ):
        assert (store.root / "build-splits" / name).is_file()
    development = pd.read_parquet(
        store.root / "build-splits/development_features.parquet"
    )
    holdout = pd.read_parquet(store.root / "build-splits/holdout_features.parquet")
    assert set(development["split"]) == {"train", "validation"}
    assert set(holdout["split"]) == {"test"}
    manifest = read_json(store.root / "build-splits/feature_manifest.json")
    assert manifest["partitions"]["development"]["row_count"] == len(development)
    assert manifest["partitions"]["holdout"]["row_count"] == len(holdout)
    assert len(manifest["partitions"]["development"]["schema_sha256"]) == 64
    assert len(manifest["partitions"]["development"]["sample_id_order_sha256"]) == 64
    assert len(manifest["partitions"]["holdout"]["schema_sha256"]) == 64
    assert len(manifest["partitions"]["holdout"]["sample_id_order_sha256"]) == 64
    store.verify_terminal_stage("build-splits")
    assert read_json(store.root / "build-splits/stage_result.json")["status"] == "success"


def test_empty_validation_split_is_durable_invalid(tmp_path: Path) -> None:
    store = make_run(tmp_path, canonical_frame(periods=14))
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS

    result = service.build_splits(store.root)
    resumed = service.build_splits(store.root)

    assert result.status is StageStatus.INVALID
    assert resumed.status is StageStatus.INVALID
    assert "already invalid" in (resumed.message or "")
    store.verify_terminal_stage("build-splits", accepted_statuses=frozenset({"INVALID"}))
    assert read_json(store.root / "build-splits/stage_manifest.json")["status"] == "INVALID"


def test_canonical_tamper_before_build_is_rejected(tmp_path: Path) -> None:
    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    canonical = store.root / "validate-data/canonical.parquet"
    canonical.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        service.build_splits(store.root)

    assert not (store.root / "build-splits/stage_manifest.json").exists()


def test_canonical_tamper_during_commit_rolls_back_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    canonical = store.root / "validate-data/canonical.parquet"
    original = RunStore._validated_target
    tampered = False

    def is_build_commit(frame: inspect.FrameInfo) -> bool:
        candidate = frame.frame.f_locals.get("self")
        return frame.function == "__exit__" and getattr(candidate, "stage", None) == "build-splits"

    def tamper_on_commit(self: RunStore, relative_path: str | Path) -> Path:
        nonlocal tampered
        resolved = original(self, relative_path)
        in_transaction_commit = any(is_build_commit(frame) for frame in inspect.stack())
        if (
            not tampered
            and in_transaction_commit
            and str(relative_path).replace("\\", "/") == "validate-data/canonical.parquet"
        ):
            canonical.write_bytes(b"changed-during-commit")
            tampered = True
        return resolved

    monkeypatch.setattr(RunStore, "_validated_target", tamper_on_commit)

    with pytest.raises(ValueError, match="stage input changed"):
        service.build_splits(store.root)

    assert not (store.root / "build-splits/stage_manifest.json").exists()
    assert not (store.root / "build-splits/feature_matrix.parquet").exists()


def test_resource_limit_during_build_publishes_bounded_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.services as services_module

    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS

    def exceed(self: object, declared_bytes: int = 0) -> None:
        del self, declared_bytes
        raise services_module._ResourceLimitExceeded("wall_clock")

    monkeypatch.setattr(services_module._ResourceGuard, "checkpoint", exceed)
    result = service.build_splits(store.root)

    assert result.status is StageStatus.INVALID
    store.verify_terminal_stage("build-splits", accepted_statuses=frozenset({"INVALID"}))
    ledger = read_json(store.root / "build-splits/resource_ledger.json")
    assert ledger["resource_status"] == "EXCEEDED"
    assert (store.root / "build-splits/feature_matrix.parquet").stat().st_size != 0


def test_serialization_contract_error_uses_independent_invalid_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.services as services_module

    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS

    def fail_candidate(frame: pd.DataFrame, config: PilotConfig) -> bytes:
        del frame, config
        raise ValueError("candidate serializer failed")

    monkeypatch.setattr(services_module, "feature_dataframe_parquet_bytes", fail_candidate)
    result = service.build_splits(store.root)

    assert result.status is StageStatus.INVALID
    store.verify_terminal_stage("build-splits", accepted_statuses=frozenset({"INVALID"}))
    assert pq.read_table(store.root / "build-splits/feature_matrix.parquet").num_rows == 0


def test_resource_limit_during_commit_rolls_back_success_then_publishes_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    import pv_pilot.services as services_module

    store = make_run(tmp_path, canonical_frame())
    service = DefaultPilotServices()
    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    original_checkpoint = services_module._ResourceGuard.checkpoint
    raised = False

    def exceed_in_commit(self: Any, declared_bytes: int = 0) -> None:
        nonlocal raised
        in_commit = any(
            frame.function == "__exit__"
            and getattr(frame.frame.f_locals.get("self"), "stage", None) == "build-splits"
            for frame in inspect.stack()
        )
        if in_commit and not raised:
            raised = True
            raise services_module._ResourceLimitExceeded("wall_clock")
        original_checkpoint(self, declared_bytes)

    monkeypatch.setattr(services_module._ResourceGuard, "checkpoint", exceed_in_commit)
    result = service.build_splits(store.root)

    assert raised
    assert result.status is StageStatus.INVALID
    store.verify_terminal_stage("build-splits", accepted_statuses=frozenset({"INVALID"}))
    assert read_json(store.root / "build-splits/resource_ledger.json")["resource_status"] == (
        "EXCEEDED"
    )
