from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from pv_pilot.config import LoadedPilotConfig, PilotConfig, load_config


def valid_config() -> dict[str, Any]:
    return {
        "protocol": {
            "version": "1.0",
            "provenance": "real",
            "approved_by": "Quang",
            "approved_at": "2026-09-28T00:00:00Z",
        },
        "paths": {"input_data": "data/pv.csv", "artifact_dir": "artifacts/run"},
        "data": {
            "format": "csv",
            "raw_cadence_minutes": 5,
            "timezone": "Asia/Ho_Chi_Minh",
            "columns": {
                "site": "site_id",
                "timestamp": "timestamp",
                "power": "power_kw",
                "capacity": "capacity_kw",
                "ghi": "ghi_w_m2",
                "temperature": "temperature_c",
            },
            "availability": {
                "power": {"fixed_latency_minutes": 0},
                "weather": {"column": "weather_available_at"},
            },
        },
        "target": {"interval_minutes": 5, "horizon_minutes": 15},
        "split": {
            "train_end_utc": "2025-01-01T00:00:00Z",
            "validation_end_utc": "2025-06-01T00:00:00Z",
            "test_end_utc": "2026-01-01T00:00:00Z",
        },
        "daylight": {"source": "solar_elevation", "latitude": 10.82, "longitude": 106.63},
        "features": {
            "matched_4_mode_power_lags_minutes": [0, 15],
            "matched_6_mode_power_lags_minutes": [0, 5, 15, 60],
            "include_current_ghi": True,
            "include_current_temperature": True,
            "strong_xgb_power_lags_minutes": [0, 5, 10, 15, 30, 60],
        },
        "photonics": {
            "modes": [4, 6],
            "primary_photons": 2,
            "max_photons": 2,
            "backend": "SLOS",
            "phase_scales": [math.pi / 2, math.pi],
            "raw_attempt_budgets": [1000, 10000],
        },
        "models": {
            "quantile_alpha_grid": [0.0001, 0.001, 0.01, 0.1],
            "rff": {"gamma_grid": [0.1, 1.0, 10.0], "n_components": "fock_dimension"},
            "xgboost": {
                "max_depth_grid": [3, 6],
                "learning_rate_grid": [0.03, 0.1],
                "n_estimators": 300,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "n_jobs": 1,
                "objective": "reg:quantileerror",
            },
        },
        "evaluation": {
            "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            "bootstrap": {
                "block_days": 1,
                "sensitivity_block_days": [2, 7],
                "replicates": 2000,
                "min_valid_blocks": 20,
            },
            "minimum_relative_wis_skill": 0.02,
            "coverage_acceptance": [0.85, 0.95],
        },
        "resources": {
            "max_wall_clock_seconds": 3600,
            "max_memory_gb": 8.0,
            "max_workers": 1,
        },
        "seeds": {"master_seed": 20260928},
    }


def test_valid_config_exposes_approval_and_derived_circuit_seeds() -> None:
    config = PilotConfig.model_validate(valid_config())

    assert config.science_decision_allowed is True
    assert config.seeds.master_seed == 20260928
    assert len(config.seeds.circuit_seeds) == 5
    assert len(set(config.seeds.circuit_seeds)) == 5


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_numbers(invalid: float) -> None:
    payload = valid_config()
    payload["resources"]["max_memory_gb"] = invalid
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_config_tree_is_deeply_immutable() -> None:
    config = PilotConfig.model_validate(valid_config())
    modes: Any = config.photonics.modes

    with pytest.raises(ValidationError):
        config.target.horizon_minutes = 30
    with pytest.raises(AttributeError):
        modes.append(8)


def test_synthetic_or_unapproved_protocol_cannot_authorize_science_decision() -> None:
    synthetic = valid_config()
    synthetic["protocol"] = {
        "version": "1.0",
        "provenance": "synthetic",
        "approved_by": None,
        "approved_at": None,
    }
    assert PilotConfig.model_validate(synthetic).science_decision_allowed is False


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("data", "raw_cadence_minutes"), 2),
        (("data", "format"), "json"),
        (("target", "interval_minutes"), 10),
        (("target", "horizon_minutes"), 30),
        (("photonics", "modes"), [4, 8]),
        (("photonics", "primary_photons"), 3),
        (("photonics", "max_photons"), 4),
        (("photonics", "backend"), "Naive"),
        (("photonics", "phase_scales"), [math.pi]),
        (("models", "quantile_alpha_grid"), [0.001, 0.01]),
        (("evaluation", "quantiles"), [0.1, 0.5, 0.9]),
        (("evaluation", "minimum_relative_wis_skill"), 0.01),
        (("evaluation", "coverage_acceptance"), [0.8, 0.95]),
    ],
)
def test_protocol_literals_cannot_drift(path: tuple[str, str], invalid_value: object) -> None:
    payload = valid_config()
    payload[path[0]][path[1]] = invalid_value

    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_photonic_modes_may_select_a_nonempty_subset_of_four_and_six() -> None:
    payload = valid_config()
    payload["photonics"]["modes"] = [4]
    assert PilotConfig.model_validate(payload).photonics.modes == (4,)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("features", "matched_4_mode_power_lags_minutes", [0, 10]),
        ("features", "matched_6_mode_power_lags_minutes", [0, 15, 30, 60]),
        ("features", "strong_xgb_power_lags_minutes", [0, 15, 60]),
        ("models", "rff", {"gamma_grid": [1.0], "n_components": "fock_dimension"}),
        (
            "models",
            "xgboost",
            {
                **valid_config()["models"]["xgboost"],
                "max_depth_grid": [2, 4],
            },
        ),
    ],
)
def test_feature_and_model_grids_are_locked(
    section: str, field: str, invalid_value: object
) -> None:
    payload = valid_config()
    payload[section][field] = invalid_value
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_unknown_fields_and_coercion_are_rejected() -> None:
    payload = valid_config()
    payload["surprise"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["data"]["raw_cadence_minutes"] = "5"
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_protocol_version_is_exactly_one_point_zero() -> None:
    payload = valid_config()
    payload["protocol"]["version"] = "1"
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("data", "raw_cadence_minutes"),
        ("target", "interval_minutes"),
        ("target", "horizon_minutes"),
    ],
)
def test_integer_protocol_fields_reject_equal_float_values(section: str, field: str) -> None:
    payload = valid_config()
    payload[section][field] = float(payload[section][field])
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("models", "xgboost", "n_estimators"),
        ("models", "xgboost", "n_jobs"),
        ("evaluation", "bootstrap", "block_days"),
        ("evaluation", "bootstrap", "replicates"),
        ("evaluation", "bootstrap", "min_valid_blocks"),
    ],
)
def test_nested_integer_protocol_fields_reject_equal_float_values(
    path: tuple[str, str, str],
) -> None:
    payload = valid_config()
    container = payload[path[0]][path[1]]
    container[path[2]] = float(container[path[2]])
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_availability_requires_column_or_nonnegative_fixed_latency_for_each_source() -> None:
    payload = valid_config()
    payload["data"]["availability"]["power"] = {}
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["data"]["availability"]["weather"] = {"fixed_latency_minutes": -1}
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["data"]["availability"]["weather"] = {
        "column": "available_at",
        "fixed_latency_minutes": 0,
    }
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_split_cutoffs_must_be_utc_and_strictly_ordered() -> None:
    payload = valid_config()
    payload["split"]["validation_end_utc"] = payload["split"]["train_end_utc"]
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["split"]["test_end_utc"] = "2026-01-01T00:00:00+07:00"
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("protocol", "approved_at"),
        ("split", "train_end_utc"),
        ("split", "validation_end_utc"),
        ("split", "test_end_utc"),
    ],
)
def test_datetime_fields_reject_numeric_timestamps(section: str, field: str) -> None:
    payload = valid_config()
    payload[section][field] = 0
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_datetime_fields_accept_iso_strings_and_yaml_datetime_objects() -> None:
    payload = valid_config()
    parsed = yaml.safe_load(yaml.safe_dump(payload))
    parsed["protocol"]["approved_at"] = "2026-09-28T00:00:00Z"
    parsed["split"]["train_end_utc"] = yaml.safe_load("2025-01-01T00:00:00Z")
    parsed["split"]["validation_end_utc"] = yaml.safe_load("2025-06-01T00:00:00Z")
    parsed["split"]["test_end_utc"] = yaml.safe_load("2026-01-01T00:00:00Z")

    config = PilotConfig.model_validate(parsed)

    assert config.protocol.approved_at is not None
    assert config.split.train_end_utc.utcoffset() is not None


def test_daylight_source_requires_its_matching_metadata() -> None:
    payload = valid_config()
    payload["daylight"] = {"source": "solar_elevation", "latitude": None, "longitude": 1.0}
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["daylight"] = {"source": "column"}
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)

    payload["data"]["columns"]["daylight"] = "is_daylight"
    assert PilotConfig.model_validate(payload).daylight.source.value == "column"


def test_load_config_resolves_paths_and_hash_is_stable_across_yaml_key_order(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    payload = valid_config()
    first_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    second_path.write_text(yaml.safe_dump(copy.deepcopy(payload), sort_keys=True), encoding="utf-8")

    first = load_config(first_path, project_root=tmp_path)
    second = load_config(second_path, project_root=tmp_path)

    assert first.resolved_input_path == (tmp_path / "data/pv.csv").resolve()
    assert first.resolved_artifact_path == (tmp_path / "artifacts/run").resolve()
    assert first.canonical_json() == second.canonical_json()
    assert first.config_hash() == second.config_hash()


def test_config_hash_is_independent_of_checkout_location(tmp_path: Path) -> None:
    payload = valid_config()
    payload["paths"]["artifact_dir"] = "../artifacts/run"
    hashes: list[str] = []
    for checkout in (tmp_path / "one", tmp_path / "two"):
        config_dir = checkout / "configs"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "pilot.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        config = load_config(config_path, project_root=checkout)
        hashes.append(config.config_hash())
        assert config.resolved_artifact_path == checkout / "artifacts/run"
    assert hashes[0] == hashes[1]


def test_artifact_path_must_be_relative_and_stay_under_project_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "project" / "configs"
    config_dir.mkdir(parents=True)
    payload = valid_config()
    for artifact_dir in (str(tmp_path / "absolute"), "../../escape"):
        payload["paths"]["artifact_dir"] = artifact_dir
        path = config_dir / "pilot.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises((ValidationError, ValueError), match="artifact"):
            load_config(path, project_root=config_dir.parent)


@pytest.mark.parametrize("duplicate_key", ["provenance", "approved_by"])
def test_load_config_rejects_duplicate_keys_at_any_level(
    tmp_path: Path, duplicate_key: str
) -> None:
    source = Path(__file__).parents[1] / "configs/pilot.synthetic.yaml"
    text = source.read_text(encoding="utf-8")
    needle = f"  {duplicate_key}:"
    original = next(line for line in text.splitlines() if line.startswith(needle))
    duplicate = f"{original}\n{original}"
    path = tmp_path / "duplicate.yaml"
    path.write_text(text.replace(original, duplicate, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_config(path)


def test_load_config_requires_project_root_when_git_cannot_be_discovered(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(valid_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="project_root"):
        load_config(path)


def test_loaded_config_context_is_frozen_and_config_is_context_free(tmp_path: Path) -> None:
    config_dir = tmp_path / "project/configs"
    config_dir.mkdir(parents=True)
    payload = valid_config()
    payload["paths"]["artifact_dir"] = "../artifacts/run"
    path = config_dir / "pilot.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_config(path, project_root=config_dir.parent)
    mutable_view: Any = loaded

    assert isinstance(loaded, LoadedPilotConfig)
    assert not hasattr(loaded.config, "resolved_artifact_path")
    with pytest.raises(FrozenInstanceError):
        mutable_view.source_dir = tmp_path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_data", "   "),
        ("input_data", "."),
        ("artifact_dir", "   "),
        ("artifact_dir", "."),
    ],
)
def test_paths_reject_blank_and_current_directory(field: str, value: str) -> None:
    payload = valid_config()
    payload["paths"][field] = value
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("data_format", "input_path"),
    [("csv", "data/input.parquet"), ("parquet", "data/input.csv")],
)
def test_input_suffix_must_match_data_format(data_format: str, input_path: str) -> None:
    payload = valid_config()
    payload["data"]["format"] = data_format
    payload["paths"]["input_data"] = input_path
    with pytest.raises(ValidationError, match="suffix"):
        PilotConfig.model_validate(payload)


def test_column_names_are_stripped_and_all_mappings_are_unique() -> None:
    payload = valid_config()
    payload["data"]["columns"]["site"] = "  site_id  "
    assert PilotConfig.model_validate(payload).data.columns.site == "site_id"

    payload = valid_config()
    payload["data"]["columns"]["power"] = payload["data"]["columns"]["capacity"]
    with pytest.raises(ValidationError, match="duplicate"):
        PilotConfig.model_validate(payload)

    payload = valid_config()
    payload["data"]["availability"]["weather"] = {"column": " ghi_w_m2 "}
    with pytest.raises(ValidationError, match="duplicate"):
        PilotConfig.model_validate(payload)


def test_example_configs_validate() -> None:
    root = Path(__file__).parents[1]
    real = load_config(root / "configs/pilot.example.yaml")
    synthetic = load_config(root / "configs/pilot.synthetic.yaml")

    assert real.config.protocol.provenance.value == "real"
    assert real.config.resources.max_wall_clock_seconds == 12 * 60 * 60
    assert real.config.resources.max_memory_gb == 24.0
    assert real.config.photonics.max_photons == 3
    assert synthetic.config.protocol.provenance.value == "synthetic"
    assert synthetic.config.photonics.max_photons == 3
    assert not real.config.photonics.enable_three_photon_profile
    assert not synthetic.config.photonics.enable_three_photon_profile
    assert synthetic.science_decision_allowed is False


def test_optional_three_photon_profile_is_explicitly_disabled() -> None:
    payload = valid_config()
    payload["photonics"]["max_photons"] = 3
    assert not PilotConfig.model_validate(payload).photonics.enable_three_photon_profile
    payload["photonics"]["enable_three_photon_profile"] = True
    with pytest.raises(ValidationError, match="3-photon profiling is disabled"):
        PilotConfig.model_validate(payload)
