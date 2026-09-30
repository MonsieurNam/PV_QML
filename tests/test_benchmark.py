from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml

from pv_pilot.artifacts import RunStore
from pv_pilot.baselines import DirectFitExecutor, QuantileHeads
from pv_pilot.benchmark import (
    PhotonicBenchmarkContractError,
    build_holdout_photonic_features,
    candidate_sort_key,
    finite_shot_screen,
    model_pickle_bytes,
    predict_exact,
    predict_finite_shot,
    select_photonic_models,
)
from pv_pilot.config import PilotConfig, load_config
from pv_pilot.domain import StageResult, StageStatus, deterministic_data
from pv_pilot.photonics import circuit_checksum, generate_circuit
from pv_pilot.seeds import SeedManager
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import feature_dataframe_parquet_bytes, feature_schema_sha256


def _config() -> PilotConfig:
    return load_config(Path("configs/pilot.synthetic.yaml")).config


def _development_frame() -> pd.DataFrame:
    config = _config()
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2025-01-01", periods=16, freq="12h", tz="UTC")
    for index, timestamp in enumerate(timestamps):
        split = "train" if index < 8 else "validation"
        target = float((index % 5) + 1)
        for modes in (4, 6):
            dimension = math.comb(modes + 1, 2)
            for scale in (math.pi / 2, math.pi):
                for seed_index, seed in enumerate(config.seeds.circuit_seeds):
                    base = np.arange(1, dimension + 1, dtype=np.float64)
                    base[(index + seed_index) % dimension] += target * (1 + modes / 10)
                    probabilities = base / base.sum()
                    circuit = generate_circuit(circuit_seed=seed, modes=modes, photons=2)
                    for control in ("indistinguishable", "distinguishable"):
                        values = probabilities.copy()
                        if control == "distinguishable":
                            values = np.roll(values, 1)
                        rows.append(
                            {
                                "sample_id": f"sample-{index}",
                                "site_id": "site-a",
                                "issued_at_utc": timestamp,
                                "target_at_utc": timestamp + pd.Timedelta(minutes=15),
                                "split": split,
                                "target_ac_power_kw": target,
                                "capacity_kw": 10.0,
                                "modes": modes,
                                "photons": 2,
                                "phase_scale": scale,
                                "circuit_seed": seed,
                                "control": control,
                                "probabilities": values.tolist(),
                                "dimension": dimension,
                                "state_order_sha256": circuit.state_order_sha256,
                                "circuit_sha256": circuit_checksum(circuit),
                            }
                        )
    return pd.DataFrame(rows)


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.value)


def _fast_heads(
    features: np.ndarray,
    target: np.ndarray,
    quantiles: tuple[float, ...],
    alpha: float,
    executor: object,
) -> QuantileHeads:
    del features, executor
    center = float(np.mean(target)) + alpha
    return QuantileHeads(
        quantiles=quantiles,
        estimators=tuple(
            _ConstantEstimator(center + index / 10) for index in range(len(quantiles))
        ),
    )


def _feature_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_times = (
        ("train", "2025-01-01T00:00:00Z"),
        ("train", "2025-01-01T00:05:00Z"),
        ("validation", "2025-01-16T00:00:00Z"),
        ("validation", "2025-01-17T00:00:00Z"),
        ("test", "2025-01-26T00:00:00Z"),
    )
    for index, (split, raw_time) in enumerate(split_times):
        issued = pd.Timestamp(raw_time)
        value = float(index + 1)
        record: dict[str, object] = {
            "sample_id": f"{split}-{index}",
            "site_id": "site-a",
            "issued_at_utc": issued,
            "target_at_utc": issued + pd.Timedelta(minutes=15),
            "split": split,
            "target_ac_power_kw": value,
            "capacity_kw": 10.0,
            "is_daylight": True,
            "power_lag_0m_kw": value,
            "power_lag_5m_kw": value - 0.1,
            "power_lag_10m_kw": value - 0.2,
            "power_lag_15m_kw": value - 0.3,
            "power_lag_30m_kw": value - 0.4,
            "power_lag_60m_kw": value - 0.5,
            "ghi_w_m2": 100.0 * value,
            "air_temp_c": 20.0 + value,
        }
        record.update({f"matched4_scaled_{offset}": 0.1 * (offset + index) for offset in range(4)})
        record.update({f"matched6_scaled_{offset}": 0.05 * (offset + index) for offset in range(6)})
        rows.append(record)
    return (
        pd.DataFrame(rows)
        .sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def _make_service_run(tmp_path: Path) -> RunStore:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    source = Path(__file__).parents[1] / "configs/pilot.synthetic.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    payload["photonics"]["modes"] = [4]
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (project / "data/input.csv").write_text("placeholder\n", encoding="utf-8")
    loaded = load_config(config_path, project_root=project)
    store = RunStore.create(loaded)
    with store.stage("validate-data") as transaction:
        transaction.write_bytes("canonical.parquet", b"canonical")
        transaction.write_json(
            "stage_result.json",
            deterministic_data(StageResult(stage="validate-data", status=StageStatus.SUCCESS)),
        )
    frame = _feature_frame()
    development = frame.loc[frame["split"].isin(("train", "validation"))].copy()
    holdout = frame.loc[frame["split"] == "test"].copy()

    def sample_order_hash(partition: pd.DataFrame) -> str:
        value = json.dumps(partition["sample_id"].astype(str).tolist(), separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()

    combined_members = json.dumps(
        sorted(frame["sample_id"].astype(str).tolist()), separators=(",", ":")
    )
    with store.stage("build-splits") as transaction:
        transaction.write_bytes(
            "development_features.parquet",
            feature_dataframe_parquet_bytes(development, loaded.config),
        )
        transaction.write_bytes(
            "holdout_features.parquet",
            feature_dataframe_parquet_bytes(holdout, loaded.config),
        )
        transaction.write_json(
            "feature_manifest.json",
            {
                "schema_version": "1.0",
                "status": "success",
                "matched4_columns": [
                    "power_lag_0m_kw",
                    "power_lag_15m_kw",
                    "ghi_w_m2",
                    "air_temp_c",
                ],
                "matched6_columns": [
                    "power_lag_0m_kw",
                    "power_lag_5m_kw",
                    "power_lag_15m_kw",
                    "power_lag_60m_kw",
                    "ghi_w_m2",
                    "air_temp_c",
                ],
                "partitions": {
                    "development": {
                        "path": "build-splits/development_features.parquet",
                        "splits": ["train", "validation"],
                        "row_count": len(development),
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "order_policy": ["site_id", "issued_at_utc", "sample_id"],
                        "sample_id_order_sha256": sample_order_hash(development),
                    },
                    "holdout": {
                        "path": "build-splits/holdout_features.parquet",
                        "splits": ["test"],
                        "row_count": len(holdout),
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "order_policy": ["site_id", "issued_at_utc", "sample_id"],
                        "sample_id_order_sha256": sample_order_hash(holdout),
                    },
                    "combined": {
                        "row_count": len(frame),
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "membership_policy": "sorted_unique_sample_id",
                        "sample_id_membership_sha256": hashlib.sha256(
                            combined_members.encode()
                        ).hexdigest(),
                    },
                },
            },
        )
        transaction.write_json(
            "minmax_scalers.json",
            {
                "matched4": {
                    "schema_version": "1.0",
                    "feature_order": [
                        "power_lag_0m_kw",
                        "power_lag_15m_kw",
                        "ghi_w_m2",
                        "air_temp_c",
                    ],
                    "fitted_split": "train",
                },
                "matched6": {
                    "schema_version": "1.0",
                    "feature_order": [
                        "power_lag_0m_kw",
                        "power_lag_5m_kw",
                        "power_lag_15m_kw",
                        "power_lag_60m_kw",
                        "ghi_w_m2",
                        "air_temp_c",
                    ],
                    "fitted_split": "train",
                },
            },
        )
        transaction.write_json(
            "stage_result.json",
            deterministic_data(StageResult(stage="build-splits", status=StageStatus.SUCCESS)),
        )
    with store.stage("train-baselines") as transaction:
        transaction.write_bytes("predictions.parquet", b"immutable-baseline-predictions")
        transaction.write_json(
            "stage_result.json",
            deterministic_data(StageResult(stage="train-baselines", status=StageStatus.SUCCESS)),
        )
    result = DefaultPilotServices().build_photonic_features(store.root)
    assert result.status is StageStatus.SUCCESS
    return store


def test_candidate_tie_break_order_is_locked() -> None:
    candidates = [
        {"modes": 6, "phase_scale": math.pi / 2, "alpha": 0.1, "config_key": "a"},
        {"modes": 4, "phase_scale": math.pi, "alpha": 0.1, "config_key": "b"},
        {"modes": 4, "phase_scale": math.pi / 2, "alpha": 0.001, "config_key": "c"},
        {"modes": 4, "phase_scale": math.pi / 2, "alpha": 0.1, "config_key": "d"},
    ]
    assert min(candidates, key=candidate_sort_key)["config_key"] == "d"


def test_selection_scores_all_five_seeds_and_blocked_day_origins() -> None:
    config = _config()
    development = _development_frame()
    result = select_photonic_models(development, config)
    assert result.selected["control"] == "indistinguishable"
    assert result.selected["photons"] == 2
    assert len(result.selected_models) == 10
    primary_rows = result.validation_scores.loc[
        result.validation_scores["selection_role"] == "primary_grid"
    ]
    expected_candidates = 2 * 2 * 4
    origins = primary_rows["origin_id"].nunique()
    assert primary_rows["config_key"].nunique() == expected_candidates
    assert len(primary_rows) == expected_candidates * 5 * origins
    assert primary_rows.groupby(["config_key", "origin_id"])["circuit_seed"].nunique().eq(5).all()
    assert result.selection_manifest["seed_policy"] == "equal_weight_mean_no_best_seed"
    assert result.selection_manifest["origin_policy"] == "utc_calendar_day"
    assert len(result.selection_manifest["selection_freeze_sha256"]) == 64
    first_seed = config.seeds.circuit_seeds[0]
    model = result.selected_models[("indistinguishable", first_seed)]
    selected_train = development.loc[
        (development["split"] == "train")
        & (development["control"] == "indistinguishable")
        & (development["circuit_seed"] == first_seed)
        & (development["modes"] == result.selected["modes"])
        & np.isclose(development["phase_scale"], result.selected["phase_scale"], atol=0, rtol=0)
    ]
    expected_mean = np.vstack(selected_train["probabilities"].map(np.asarray)).mean(axis=0)
    assert model.scaler.mean_ == pytest.approx(expected_mean)


def test_selection_rejects_missing_seed_origin() -> None:
    frame = _development_frame()
    config = _config()
    missing_seed = config.seeds.circuit_seeds[-1]
    mask = ~(
        (frame["split"] == "validation")
        & (frame["circuit_seed"] == missing_seed)
        & (frame["control"] == "indistinguishable")
    )
    with pytest.raises(PhotonicBenchmarkContractError, match="complete"):
        select_photonic_models(frame.loc[mask].copy(), config)


def test_finite_shot_is_deterministic_and_isolated_by_sampling_seed() -> None:
    manager = SeedManager(20260928)
    exact = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "site_id": "a",
                "issued_at_utc": pd.Timestamp("2025-02-01T00:00:00Z"),
                "target_at_utc": pd.Timestamp("2025-02-01T00:15:00Z"),
                "target_ac_power_kw": 2.0,
                "capacity_kw": 10.0,
                "modes": 4,
                "photons": 2,
                "phase_scale": math.pi / 2,
                "circuit_seed": 42,
                "control": "indistinguishable",
                "probabilities": [0.2, 0.3, 0.5],
                "dimension": 3,
                "state_order_sha256": "a" * 64,
                "circuit_sha256": "b" * 64,
            }
        ]
    )
    first = finite_shot_screen(exact, (1000, 10000), manager)
    second = finite_shot_screen(exact, (1000, 10000), manager)
    assert first.equals(second)
    assert len(first) == 2
    assert first["counts"].map(sum).tolist() == [1000, 10000]
    assert first["accepted_samples"].tolist() == [1000, 10000]
    assert first["raw_attempts"].tolist() == [1000, 10000]
    assert first["ideal_shot_noise_only"].all()
    assert (~first["loss_modeled"]).all()
    changed = finite_shot_screen(exact, (1000, 10000), SeedManager(99))
    assert not first["counts"].equals(changed["counts"])
    for column in (
        "sample_id",
        "circuit_seed",
        "budget",
        "circuit_sha256",
        "probabilities_exact",
    ):
        assert first[column].equals(changed[column])


def test_real_slos_holdout_both_controls_predictions_and_serialization() -> None:
    import pickle

    config = _config()
    selection = select_photonic_models(_development_frame(), config)
    modes = int(selection.selected["modes"])
    holdout = pd.DataFrame(
        [
            {
                "sample_id": "holdout-1",
                "site_id": "site-a",
                "issued_at_utc": pd.Timestamp("2025-02-01T00:00:00Z"),
                "target_at_utc": pd.Timestamp("2025-02-01T00:15:00Z"),
                "split": "test",
                "target_ac_power_kw": 3.0,
                "capacity_kw": 10.0,
                **{f"matched{modes}_scaled_{index}": 0.1 * index for index in range(modes)},
            }
        ]
    )
    circuits = {
        seed: generate_circuit(circuit_seed=seed, modes=modes, photons=2)
        for seed in config.seeds.circuit_seeds
    }
    exact = build_holdout_photonic_features(holdout, selected=selection.selected, circuits=circuits)
    assert len(exact) == 10
    assert set(exact["control"]) == {"indistinguishable", "distinguishable"}
    assert exact.groupby("control")["circuit_seed"].nunique().eq(5).all()
    assert exact["probabilities"].map(sum).to_numpy() == pytest.approx(np.ones(10))
    predictions, diagnostics = predict_exact(
        exact, selection.selected_models, tuple(config.evaluation.quantiles)
    )
    assert len(predictions) == 10
    assert predictions.groupby("control")["circuit_seed"].nunique().eq(5).all()
    quantile_columns = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    values = predictions[quantile_columns].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (values >= 0).all() and (values <= 10).all()
    assert len(diagnostics) == 10

    restored = pickle.loads(model_pickle_bytes(selection.selected_models))
    assert set(restored) == set(selection.selected_models)
    repeated, _ = predict_exact(exact, restored, tuple(config.evaluation.quantiles))
    assert repeated[quantile_columns].to_numpy() == pytest.approx(values)

    counts = finite_shot_screen(
        exact, tuple(config.photonics.raw_attempt_budgets), SeedManager(config.seeds.master_seed)
    )
    sampled = predict_finite_shot(
        counts, selection.selected_models, tuple(config.evaluation.quantiles)
    )
    assert len(sampled) == 10
    assert set(sampled["budget"]) == {1000, 10000}
    assert sampled["ideal_shot_noise_only"].all()
    assert (~sampled["loss_modeled"]).all()


def test_benchmark_service_freezes_before_holdout_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")
    services_module: Any = importlib.import_module("pv_pilot.services")

    store = _make_service_run(tmp_path)
    frozen = False
    holdout_materializations = 0
    original_select = benchmark_module.select_photonic_models
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file
    original_dataset = services_module.pds.dataset

    def select_then_freeze(*args: Any, **kwargs: Any) -> object:
        nonlocal frozen
        result = original_select(*args, **kwargs)
        frozen = True
        return result

    def guarded_artifact_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            assert frozen, "artifact layer hashed holdout before selection freeze"
        return cast(str, original_artifact_hash(path))

    def guarded_service_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            assert frozen, "service hashed holdout before selection freeze"
        return cast(str, original_service_hash(path))

    def tracked_dataset(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal holdout_materializations
        if Path(path).name == "holdout_features.parquet":
            assert frozen
            holdout_materializations += 1
        return original_dataset(path, *args, **kwargs)

    monkeypatch.setattr(benchmark_module, "select_photonic_models", select_then_freeze)
    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", _fast_heads)
    monkeypatch.setattr(
        services_module.baseline_models,
        "DeadlineProcessFitExecutor",
        lambda **kwargs: DirectFitExecutor(kwargs["resource_checkpoint"]),
    )
    monkeypatch.setattr(artifacts_module, "sha256_file", guarded_artifact_hash)
    monkeypatch.setattr(services_module, "sha256_file", guarded_service_hash)
    monkeypatch.setattr(services_module.pds, "dataset", tracked_dataset)

    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.SUCCESS
    assert holdout_materializations == 1
    resumed = DefaultPilotServices().benchmark(store.root)
    assert resumed.status is StageStatus.SUCCESS
    assert holdout_materializations == 1
    exact = pd.read_parquet(store.root / "benchmark/exact_predictions.parquet")
    finite = pd.read_parquet(store.root / "benchmark/finite_shot_predictions.parquet")
    assert len(exact) == 10
    assert len(finite) == 10
    manifest = json.loads(
        (store.root / "benchmark/selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["holdout_audit"]["materialization_count"] == 1
    assert manifest["baseline_predictions"]["policy"] == ("immutable_reuse_no_retrain_no_repredict")
    assert manifest["scientific_decision"] == "NOT_EVALUATED"
    marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "SUCCESS"
    assert marker["details"]["selection_freeze_sha256"] == manifest["selection_freeze_sha256"]


def test_benchmark_contract_failure_is_durable_invalid_without_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")

    store = _make_service_run(tmp_path)
    original_hash = artifacts_module.sha256_file

    def reject_holdout(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("INVALID diagnostic touched holdout")
        return cast(str, original_hash(path))

    def reject_selection(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise PhotonicBenchmarkContractError("expected selection contract failure")

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_holdout)
    monkeypatch.setattr(benchmark_module, "select_photonic_models", reject_selection)
    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    resumed = DefaultPilotServices().benchmark(store.root)
    assert resumed.status is StageStatus.INVALID
    marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INVALID"
    assert marker["details"]["prerequisite_verification"] == "marker_only"
    assert pd.read_parquet(store.root / "benchmark/exact_predictions.parquet").empty
