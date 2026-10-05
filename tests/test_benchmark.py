from __future__ import annotations

import hashlib
import importlib
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

from pv_pilot.artifacts import RunStore
from pv_pilot.baselines import (
    BaselineModelBundle,
    DirectFitExecutor,
    PersistenceModel,
    QuantileHeads,
)
from pv_pilot.benchmark import (
    PhotonicBenchmarkContractError,
    PhotonicSelectionResult,
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
from pv_pilot.photonics import (
    CircuitDefinition,
    build_circuit,
    circuit_checksum,
    enumerate_fock_states,
    exact_probabilities,
    generate_circuit,
    state_order_checksum,
)
from pv_pilot.seeds import SeedManager
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import feature_dataframe_parquet_bytes, feature_schema_sha256


class _ConstantQuantilePredictor:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.linspace(1.0, 2.0, 7), (len(values), 1))


@lru_cache(maxsize=1)
def _config() -> PilotConfig:
    return load_config(Path("configs/pilot.synthetic.yaml")).config


def _development_frame() -> pd.DataFrame:
    return _cached_development_frame().copy(deep=True)


@lru_cache(maxsize=1)
def _cached_development_frame() -> pd.DataFrame:
    config = _config()
    rows: list[dict[str, object]] = []
    timestamps = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-16T00:00:00Z",
            "2025-01-17T00:00:00Z",
        ],
        utc=True,
    )
    for index, timestamp in enumerate(timestamps):
        split = "train" if index < 2 else "validation"
        target = float((index % 5) + 1)
        for modes in (4, 6):
            dimension = math.comb(modes + 1, 2)
            scaled = np.asarray([0.05 + 0.1 * index + 0.03 * offset for offset in range(modes)])
            for scale in (math.pi / 2, math.pi):
                for seed in config.seeds.circuit_seeds:
                    circuit = generate_circuit(circuit_seed=seed, modes=modes, photons=2)
                    for control in ("indistinguishable", "distinguishable"):
                        values = exact_probabilities(
                            build_circuit(circuit, scaled, scale),
                            photons=2,
                            control=control,
                        )
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


def _development_reference(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
    ]
    reference = frame.loc[:, metadata].drop_duplicates("sample_id").copy()
    for modes in (4, 6):
        for offset in range(modes):
            reference[f"matched{modes}_scaled_{offset}"] = [
                0.05 + 0.1 * int(sample_id.split("-")[-1]) + 0.03 * offset
                for sample_id in reference["sample_id"].astype(str)
            ]
    return reference


def _trusted_contracts(
    config: PilotConfig,
) -> tuple[
    dict[tuple[int, int], CircuitDefinition],
    dict[tuple[int, int], tuple[int, str]],
]:
    circuits = {
        (modes, seed): generate_circuit(circuit_seed=seed, modes=modes, photons=2)
        for modes in config.photonics.modes
        for seed in config.seeds.circuit_seeds
    }
    states = {
        (modes, 2): (
            math.comb(modes + 1, 2),
            state_order_checksum(enumerate_fock_states(modes, 2)),
        )
        for modes in config.photonics.modes
    }
    return circuits, states


def _select(frame: pd.DataFrame, config: PilotConfig) -> PhotonicSelectionResult:
    circuits, states = _trusted_contracts(config)
    return select_photonic_models(
        frame,
        config,
        development_reference=_development_reference(frame),
        trusted_circuits=circuits,
        trusted_state_spaces=states,
    )


def test_development_semantic_validation_honors_resource_checkpoint_before_slos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.benchmark as benchmark_module

    config = _config()
    frame = _development_frame()
    circuits, states = _trusted_contracts(config)
    slos_calls = 0

    def counted_exact(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        nonlocal slos_calls
        slos_calls += 1
        return exact_probabilities(*args, **kwargs)

    def expired(_declared: int) -> None:
        raise TimeoutError("expired before semantic validation")

    monkeypatch.setattr(benchmark_module, "exact_probabilities", counted_exact)
    with pytest.raises(TimeoutError, match="expired"):
        select_photonic_models(
            frame,
            config,
            development_reference=_development_reference(frame),
            trusted_circuits=circuits,
            trusted_state_spaces=states,
            resource_checkpoint=expired,
        )
    assert slos_calls == 0


def test_development_semantic_validation_stops_promptly_mid_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.benchmark as benchmark_module

    config = _config()
    frame = _development_frame()
    circuits, states = _trusted_contracts(config)
    slos_calls = 0
    checkpoints = 0

    def counted_exact(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        nonlocal slos_calls
        slos_calls += 1
        return exact_probabilities(*args, **kwargs)

    def expire_mid_grid(_declared: int) -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 7:
            raise TimeoutError("expired mid-grid")

    monkeypatch.setattr(benchmark_module, "exact_probabilities", counted_exact)
    with pytest.raises(TimeoutError, match="mid-grid"):
        select_photonic_models(
            frame,
            config,
            development_reference=_development_reference(frame),
            trusted_circuits=circuits,
            trusted_state_spaces=states,
            resource_checkpoint=expire_mid_grid,
        )
    assert 0 < slos_calls < len(frame)
    assert checkpoints == 7


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("state_order", "state-order"),
        ("distinguishable_circuit", "circuit checksum"),
        ("candidate_target", "sample metadata"),
        ("probability_length", "probability"),
        ("probability_sum", "probability"),
        ("probability_reorder", "probability semantics"),
    ],
)
def test_semantic_contract_rejects_before_any_fit(
    mutation: str, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    config = _config()
    frame = _development_frame()
    row = frame.index[0]
    if mutation == "state_order":
        frame.at[row, "state_order_sha256"] = "f" * 64
    elif mutation == "distinguishable_circuit":
        target = frame.index[frame["control"] == "distinguishable"][0]
        frame.at[target, "circuit_sha256"] = "e" * 64
    elif mutation == "candidate_target":
        frame.at[row, "target_ac_power_kw"] = float(frame.at[row, "target_ac_power_kw"]) + 1
    elif mutation == "probability_length":
        frame.at[row, "probabilities"] = list(frame.at[row, "probabilities"])[:-1]
    elif mutation == "probability_sum":
        changed = list(frame.at[row, "probabilities"])
        changed[0] += 0.1
        frame.at[row, "probabilities"] = changed
    else:
        frame.at[row, "probabilities"] = list(reversed(frame.at[row, "probabilities"]))

    def unexpected_fit(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("semantic validation must finish before model fit")

    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", unexpected_fit)
    circuits, states = _trusted_contracts(config)
    with pytest.raises(PhotonicBenchmarkContractError, match=match):
        select_photonic_models(
            frame,
            config,
            development_reference=_development_reference(_development_frame()),
            trusted_circuits=circuits,
            trusted_state_spaces=states,
        )


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


def test_photonic_candidate_scaling_fit_and_prediction_share_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.benchmark as benchmark_module

    operations: list[str] = []

    class RecordingExecutor(DirectFitExecutor):
        def run(
            self,
            operation: str,
            function: Any,
            *args: Any,
            declared_bytes: int = 0,
            **kwargs: Any,
        ) -> Any:
            operations.append(operation)
            return super().run(
                operation,
                function,
                *args,
                declared_bytes=declared_bytes,
                **kwargs,
            )

    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", _fast_heads)
    config = _config()
    seed = config.seeds.circuit_seeds[0]
    rows, scaler, checksum = benchmark_module._fit_and_score_candidate(
        _development_frame(),
        modes=4,
        phase_scale=math.pi / 2,
        alpha=0.01,
        control="indistinguishable",
        circuit_seed=seed,
        quantiles=tuple(config.evaluation.quantiles),
        executor=RecordingExecutor(lambda _declared: None),
        selection_role="primary_grid",
    )

    assert operations == [
        "photonic-candidate-batch:"
        f"indistinguishable:m=4:c={math.pi / 2}:seed={seed}:alpha=0.01"
    ]
    assert len(rows) == 2
    assert scaler.n_features_in_ == math.comb(4 + 1, 2)
    assert len(checksum) == 64


def test_photonic_model_scaling_and_head_prediction_share_one_batch() -> None:
    import pv_pilot.benchmark as benchmark_module

    operations: list[str] = []

    class RecordingExecutor(DirectFitExecutor):
        def run(
            self,
            operation: str,
            function: Any,
            *args: Any,
            declared_bytes: int = 0,
            **kwargs: Any,
        ) -> Any:
            operations.append(operation)
            return super().run(
                operation,
                function,
                *args,
                declared_bytes=declared_bytes,
                **kwargs,
            )

    train = np.arange(30, dtype=np.float64).reshape(3, 10)
    scaler = StandardScaler().fit(train)
    heads = QuantileHeads(
        quantiles=tuple(_config().evaluation.quantiles),
        estimators=tuple(_ConstantEstimator(float(index)) for index in range(7)),
    )
    model = benchmark_module.PhotonicSeedModel(
        control="indistinguishable",
        circuit_seed=17,
        modes=4,
        photons=2,
        phase_scale=math.pi / 2,
        alpha=0.01,
        scaler=scaler,
        heads=heads,
        circuit_sha256="a" * 64,
    )

    actual = model.raw_predict(train, RecordingExecutor(lambda _declared: None))

    assert operations == ["photonic:indistinguishable:seed=17:predict-batch"]
    assert np.array_equal(actual, heads.predict(scaler.transform(train)))


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
    quantiles = tuple(loaded.config.evaluation.quantiles)
    columns = ("power_lag_0m_kw", "power_lag_15m_kw", "ghi_w_m2", "air_temp_c")
    baseline_models = {
        "persistence": PersistenceModel(quantiles, np.zeros(7, dtype=np.float64)),
        "raw": BaselineModelBundle("raw", columns, _ConstantQuantilePredictor()),
        "rff": BaselineModelBundle("rff", columns, _ConstantQuantilePredictor()),
        "xgboost": BaselineModelBundle("xgboost", columns, _ConstantQuantilePredictor()),
    }
    baseline_model_bytes = pickle.dumps(baseline_models, protocol=5)
    baseline_freeze = "b" * 64
    with store.stage("train-baselines") as transaction:
        transaction.write_json(
            "selection_manifest.json",
            {
                "schema_version": "1.0",
                "selection_freeze_sha256": baseline_freeze,
                "holdout_access": "benchmark_only_after_photonic_freeze",
                "selected": {
                    branch: {"feature_columns": list(columns)}
                    for branch in ("raw", "rff", "xgboost")
                },
            },
        )
        transaction.write_bytes("models/baseline_models.pkl", baseline_model_bytes)
        transaction.write_json(
            "models/manifest.json",
            {
                "schema_version": "1.0",
                "models_sha256": hashlib.sha256(baseline_model_bytes).hexdigest(),
                "selection_freeze_sha256": baseline_freeze,
            },
        )
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
    result = _select(development, config)
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
        _select(frame.loc[mask].copy(), config)


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


def test_finite_shot_checks_resource_before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    exact = pd.DataFrame(
        [
            {
                "sample_id": "sample",
                "site_id": "site-a",
                "issued_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
                "target_at_utc": pd.Timestamp("2025-01-01T00:15:00Z"),
                "target_ac_power_kw": 1.0,
                "capacity_kw": 10.0,
                "modes": 4,
                "photons": 2,
                "phase_scale": math.pi / 2,
                "circuit_seed": 1,
                "control": "indistinguishable",
                "probabilities": [0.2, 0.3, 0.5],
                "dimension": 3,
                "state_order_sha256": "a" * 64,
                "circuit_sha256": "b" * 64,
            }
        ]
    )
    sampling_calls = 0

    def unexpected_rng(*args: Any, **kwargs: Any) -> object:
        nonlocal sampling_calls
        del args, kwargs
        sampling_calls += 1
        raise AssertionError("expired finite-shot screen sampled")

    def expired(_declared: int) -> None:
        raise TimeoutError("finite-shot resource expired")

    monkeypatch.setattr(np.random, "default_rng", unexpected_rng)
    with pytest.raises(TimeoutError, match="finite-shot"):
        finite_shot_screen(
            exact,
            (1000, 10000),
            SeedManager(42),
            resource_checkpoint=expired,
        )
    assert sampling_calls == 0


def test_real_slos_holdout_both_controls_predictions_and_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    import pv_pilot.benchmark as benchmark_module

    config = _config()
    selection = _select(_development_frame(), config)
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
    trusted_state_space = (
        math.comb(modes + 1, 2),
        state_order_checksum(enumerate_fock_states(modes, 2)),
    )
    slos_calls = 0
    checkpoints: list[int] = []

    def counted_exact(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        nonlocal slos_calls
        slos_calls += 1
        return exact_probabilities(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "exact_probabilities", counted_exact)
    validated = build_holdout_photonic_features(
        holdout,
        selected=selection.selected,
        circuits=circuits,
        trusted_state_space=trusted_state_space,
        resource_checkpoint=checkpoints.append,
    )
    exact = validated.frame
    assert len(exact) == 10
    assert slos_calls == len(exact)
    assert checkpoints
    assert set(exact["control"]) == {"indistinguishable", "distinguishable"}
    assert exact.groupby("control")["circuit_seed"].nunique().eq(5).all()
    assert exact["probabilities"].map(sum).to_numpy() == pytest.approx(np.ones(10))
    predictions, diagnostics = predict_exact(
        validated,
        selection.selected_models,
        tuple(config.evaluation.quantiles),
    )
    assert slos_calls == len(exact)
    assert len(predictions) == 10
    assert predictions.groupby("control")["circuit_seed"].nunique().eq(5).all()
    quantile_columns = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    values = predictions[quantile_columns].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (values >= 0).all() and (values <= 10).all()
    assert len(diagnostics) == 10

    restored = pickle.loads(model_pickle_bytes(selection.selected_models))
    assert set(restored) == set(selection.selected_models)
    repeated, _ = predict_exact(
        validated,
        restored,
        tuple(config.evaluation.quantiles),
    )
    assert slos_calls == len(exact)
    assert repeated[quantile_columns].to_numpy() == pytest.approx(values)

    for column, bad_value in (
        ("state_order_sha256", "f" * 64),
        ("circuit_sha256", "e" * 64),
        ("target_ac_power_kw", 9.0),
        ("probabilities", list(reversed(exact.iloc[0]["probabilities"]))),
    ):
        original = validated._frame.at[validated._frame.index[0], column]
        validated._frame.at[validated._frame.index[0], column] = bad_value
        with pytest.raises(PhotonicBenchmarkContractError, match="token integrity"):
            predict_exact(validated, selection.selected_models, tuple(config.evaluation.quantiles))
        validated._frame.at[validated._frame.index[0], column] = original
    assert slos_calls == len(exact)

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
    reject_holdout_hash = False
    holdout_materializations = 0
    original_select = benchmark_module.select_photonic_models
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file
    original_materialize = (
        services_module.DefaultPilotServices._materialize_benchmark_holdout_source
    )

    def select_then_freeze(*args: Any, **kwargs: Any) -> object:
        nonlocal frozen
        result = original_select(*args, **kwargs)
        frozen = True
        return result

    def guarded_artifact_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            assert frozen, "artifact layer hashed holdout before selection freeze"
            if reject_holdout_hash:
                raise AssertionError("terminal verification reopened the physical holdout")
        return cast(str, original_artifact_hash(path))

    def guarded_service_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            assert frozen, "service hashed holdout before selection freeze"
        return cast(str, original_service_hash(path))

    def tracked_materialize(self: Any, path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal holdout_materializations
        if Path(path).name == "holdout_features.parquet":
            assert frozen
            holdout_materializations += 1
        return original_materialize(self, path, *args, **kwargs)

    monkeypatch.setattr(benchmark_module, "select_photonic_models", select_then_freeze)
    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", _fast_heads)
    monkeypatch.setattr(
        services_module.baseline_models,
        "DeadlineProcessFitExecutor",
        lambda **kwargs: DirectFitExecutor(kwargs["resource_checkpoint"]),
    )
    monkeypatch.setattr(artifacts_module, "sha256_file", guarded_artifact_hash)
    monkeypatch.setattr(services_module, "sha256_file", guarded_service_hash)
    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_materialize_benchmark_holdout_source",
        tracked_materialize,
    )

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
    baseline = pd.read_parquet(store.root / "benchmark/baseline_predictions.parquet")
    assert len(baseline) == 4
    assert not (store.root / "train-baselines/predictions.parquet").exists()
    manifest = json.loads(
        (store.root / "benchmark/selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["holdout_audit"]["materialization_count"] == 1
    assert manifest["baseline_predictions"]["policy"] == (
        "same_authenticated_holdout_after_photonic_freeze"
    )
    assert manifest["scientific_decision"] == "NOT_EVALUATED"
    marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "SUCCESS"
    assert marker["details"]["selection_freeze_sha256"] == manifest["selection_freeze_sha256"]
    assert marker["input_hashes"][".benchmark-holdout-access/cache.parquet"] == (
        manifest["holdout_audit"]["source_sha256"]
    ) == manifest["holdout_audit"]["cache_sha256"]
    reject_holdout_hash = True
    store.verify_terminal_stage("benchmark")


def test_benchmark_parses_authenticated_development_snapshot_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    service = DefaultPilotServices()
    development_path = (
        store.root / "build-photonic-features/development_photonic_features.parquet"
    )
    original_digest = hashlib.sha256(development_path.read_bytes()).hexdigest()
    original_capture = service._capture_report_artifact
    replaced = False

    def capture_then_replace(*args: Any, **kwargs: Any) -> tuple[bytes, int]:
        nonlocal replaced
        payload, live = original_capture(*args, **kwargs)
        if kwargs.get("relative_path") == (
            "build-photonic-features/development_photonic_features.parquet"
        ):
            development_path.write_bytes(b"replacement-after-capture")
            replaced = True
        return payload, live

    monkeypatch.setattr(service, "_capture_report_artifact", capture_then_replace)
    result = service.benchmark(store.root)

    assert replaced is True
    assert result.status is StageStatus.SUCCESS
    marker = json.loads(
        (store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["input_hashes"][
        "build-photonic-features/development_photonic_features.parquet"
    ] == original_digest


def test_benchmark_holdout_materialization_rejects_mutation_of_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    holdout_path = store.root / "build-splits/holdout_features.parquet"
    original_open = services_module.os.open
    swapped = False

    def open_then_replace(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path).name == "holdout_features.parquet" and not swapped:  # type: ignore[arg-type]
            holdout_path.write_bytes(b"mutation-after-descriptor-open")
            swapped = True
        return descriptor

    monkeypatch.setattr(services_module.os, "open", open_then_replace)
    with pytest.raises(ValueError, match=r"snapshot input changed while (opening|captured)"):
        DefaultPilotServices().benchmark(store.root)

    assert swapped is True
    assert not (store.root / "benchmark/stage_manifest.json").exists()


def _install_fast_benchmark_service(
    monkeypatch: pytest.MonkeyPatch, services_module: Any, benchmark_module: Any
) -> None:
    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", _fast_heads)
    monkeypatch.setattr(
        services_module.baseline_models,
        "DeadlineProcessFitExecutor",
        lambda **kwargs: DirectFitExecutor(kwargs["resource_checkpoint"]),
    )


def test_holdout_intent_without_complete_never_reopens_source_and_becomes_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    original_materialize = (
        services_module.DefaultPilotServices._materialize_benchmark_holdout_source
    )
    source_opens = 0
    crash_once = True
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file

    def reject_source_artifact_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("physical holdout was hashed outside its exact-once access")
        return cast(str, original_artifact_hash(path))

    def reject_source_service_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("physical holdout was hashed outside its exact-once access")
        return cast(str, original_service_hash(path))

    def counted_materialize(self: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal source_opens
        source_opens += 1
        return cast(pd.DataFrame, original_materialize(self, *args, **kwargs))

    def crash_after_materialization(self: Any) -> None:
        nonlocal crash_once
        del self
        if crash_once:
            crash_once = False
            raise RuntimeError("simulated crash after physical holdout access")

    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_materialize_benchmark_holdout_source",
        counted_materialize,
    )
    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_after_benchmark_holdout_materialized",
        crash_after_materialization,
    )
    monkeypatch.setattr(artifacts_module, "sha256_file", reject_source_artifact_hash)
    monkeypatch.setattr(services_module, "sha256_file", reject_source_service_hash)

    with pytest.raises(RuntimeError, match="simulated crash"):
        DefaultPilotServices().benchmark(store.root)
    assert source_opens == 1
    assert not (store.root / "benchmark/stage_manifest.json").exists()

    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert source_opens == 1
    marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["details"]["gate"] == "holdout_access_indeterminate"


def test_holdout_complete_cache_resumes_success_without_reopening_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    original_materialize = (
        services_module.DefaultPilotServices._materialize_benchmark_holdout_source
    )
    source_opens = 0
    crash_once = True
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file

    def reject_source_artifact_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("physical holdout was hashed outside its exact-once access")
        return cast(str, original_artifact_hash(path))

    def reject_source_service_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("physical holdout was hashed outside its exact-once access")
        return cast(str, original_service_hash(path))

    def counted_materialize(self: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal source_opens
        source_opens += 1
        return cast(pd.DataFrame, original_materialize(self, *args, **kwargs))

    def crash_after_complete(self: Any) -> None:
        nonlocal crash_once
        del self
        if crash_once:
            crash_once = False
            raise RuntimeError("simulated crash after durable holdout cache")

    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_materialize_benchmark_holdout_source",
        counted_materialize,
    )
    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_after_benchmark_holdout_cache_complete",
        crash_after_complete,
    )
    monkeypatch.setattr(artifacts_module, "sha256_file", reject_source_artifact_hash)
    monkeypatch.setattr(services_module, "sha256_file", reject_source_service_hash)

    with pytest.raises(RuntimeError, match="simulated crash"):
        DefaultPilotServices().benchmark(store.root)
    assert source_opens == 1
    access_root = store.root / ".benchmark-holdout-access"
    assert (access_root / "intent.json").is_file()
    assert (access_root / "cache.parquet").is_file()
    assert (access_root / "complete.json").is_file()

    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.SUCCESS
    assert source_opens == 1
    resumed = DefaultPilotServices().benchmark(store.root)
    assert resumed.status is StageStatus.SUCCESS
    assert source_opens == 1


@pytest.mark.parametrize("tamper", ["intent", "cache", "complete", "cache_and_complete"])
@pytest.mark.parametrize("resume", [False, True])
def test_same_attempt_holdout_journal_mutation_cannot_rebind_consumed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str, resume: bool
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    service = DefaultPilotServices()

    def crash_after_complete() -> None:
        raise RuntimeError("stop after durable COMPLETE")

    if resume:
        monkeypatch.setattr(
            service, "_after_benchmark_holdout_cache_complete", crash_after_complete
        )
        with pytest.raises(RuntimeError, match="durable COMPLETE"):
            service.benchmark(store.root)

    access_root = store.root / ".benchmark-holdout-access"

    def mutate_journal() -> None:
        if tamper in {"cache", "cache_and_complete"}:
            cache_path = access_root / "cache.parquet"
            frame = pd.read_parquet(cache_path)
            frame.loc[0, "target_ac_power_kw"] += 0.125
            cache_bytes = feature_dataframe_parquet_bytes(frame, _config())
            cache_path.write_bytes(cache_bytes)
        if tamper == "cache_and_complete":
            complete_path = access_root / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["cache_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
            complete_path.write_text(json.dumps(complete), encoding="utf-8")
        elif tamper in {"intent", "complete"}:
            path = access_root / f"{tamper}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["access_id"] = "f" * 32
            path.write_text(json.dumps(payload), encoding="utf-8")

    if resume:
        original_load = service._load_or_materialize_benchmark_holdout

        def load_then_mutate(**kwargs: Any) -> Any:
            result = original_load(**kwargs)
            mutate_journal()
            return result

        monkeypatch.setattr(service, "_load_or_materialize_benchmark_holdout", load_then_mutate)
    else:
        monkeypatch.setattr(service, "_after_benchmark_holdout_cache_complete", mutate_journal)

    with pytest.raises(ValueError, match="checksum"):
        service.benchmark(store.root)
    assert not (store.root / "benchmark/stage_manifest.json").exists()


@pytest.mark.parametrize(
    "tamper", ["intent", "cache", "cache_and_complete", "source_marker"]
)
def test_holdout_access_tamper_fails_closed_without_source_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    import pv_pilot.benchmark as benchmark_module

    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")
    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    original_materialize = (
        services_module.DefaultPilotServices._materialize_benchmark_holdout_source
    )
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file
    source_opens = 0

    def counted_materialize(self: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal source_opens
        source_opens += 1
        return cast(pd.DataFrame, original_materialize(self, *args, **kwargs))

    def crash_after_complete(self: Any) -> None:
        del self
        raise RuntimeError("stop after COMPLETE")

    def reject_source_hash(path: Path, original: Any) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("tamper recovery reopened the physical holdout")
        return cast(str, original(path))

    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_materialize_benchmark_holdout_source",
        counted_materialize,
    )
    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_after_benchmark_holdout_cache_complete",
        crash_after_complete,
    )
    monkeypatch.setattr(
        artifacts_module,
        "sha256_file",
        lambda path: reject_source_hash(path, original_artifact_hash),
    )
    monkeypatch.setattr(
        services_module,
        "sha256_file",
        lambda path: reject_source_hash(path, original_service_hash),
    )
    with pytest.raises(RuntimeError, match="COMPLETE"):
        DefaultPilotServices().benchmark(store.root)
    assert source_opens == 1

    access_root = store.root / ".benchmark-holdout-access"
    if tamper == "intent":
        intent_path = access_root / "intent.json"
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        intent["access_id"] = "f" * 32
        intent_path.write_text(json.dumps(intent), encoding="utf-8")
    elif tamper == "cache":
        cache_path = access_root / "cache.parquet"
        cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    elif tamper == "cache_and_complete":
        cache_path = access_root / "cache.parquet"
        tampered_holdout = pd.read_parquet(cache_path)
        tampered_holdout.loc[0, "target_ac_power_kw"] += 0.125
        tampered_cache = feature_dataframe_parquet_bytes(tampered_holdout, _config())
        cache_path.write_bytes(tampered_cache)
        complete_path = access_root / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["cache_sha256"] = hashlib.sha256(tampered_cache).hexdigest()
        complete_path.write_text(json.dumps(complete), encoding="utf-8")
    else:
        marker_path = store.root / "build-splits/stage_manifest.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["details"]["tampered"] = True
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_after_benchmark_holdout_cache_complete",
        lambda self: None,
    )
    with pytest.raises(ValueError):
        DefaultPilotServices().benchmark(store.root)
    assert source_opens == 1
    assert not (store.root / "benchmark/stage_manifest.json").exists()


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


def test_benchmark_expired_resource_is_durable_invalid_before_slos_or_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    artifacts_module: Any = importlib.import_module("pv_pilot.artifacts")
    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = services_module.sha256_file
    slos_calls = 0
    fit_calls = 0

    def reject_holdout_artifact(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("expired benchmark touched holdout")
        return cast(str, original_artifact_hash(path))

    def reject_holdout_service(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("expired benchmark touched holdout")
        return cast(str, original_service_hash(path))

    def unexpected_slos(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        nonlocal slos_calls
        slos_calls += 1
        return exact_probabilities(*args, **kwargs)

    def unexpected_fit(*args: Any, **kwargs: Any) -> object:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        raise AssertionError("expired benchmark started a fit")

    def expired_checkpoint(self: Any, declared_bytes: int = 0) -> None:
        del self, declared_bytes
        raise services_module._ResourceLimitExceeded("wall_clock")

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_holdout_artifact)
    monkeypatch.setattr(services_module, "sha256_file", reject_holdout_service)
    monkeypatch.setattr(benchmark_module, "exact_probabilities", unexpected_slos)
    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", unexpected_fit)
    monkeypatch.setattr(services_module._ResourceGuard, "checkpoint", expired_checkpoint)

    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert slos_calls == 0
    assert fit_calls == 0
    resumed = DefaultPilotServices().benchmark(store.root)
    assert resumed.status is StageStatus.INVALID
    marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INVALID"
    assert marker["details"]["gate"] == "resource_wall_clock"
    assert pd.read_parquet(store.root / "benchmark/exact_predictions.parquet").empty
    assert not [path for path in store.root.rglob("*") if ".tmp" in path.name]


def test_benchmark_serializer_resource_expiry_stops_later_serializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.benchmark as benchmark_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    store = _make_service_run(tmp_path)
    _install_fast_benchmark_service(monkeypatch, services_module, benchmark_module)
    original = services_module.DefaultPilotServices._serialize_benchmark_payload
    labels: list[str] = []

    def expire_on_exact(
        self: Any,
        label: str,
        producer: Any,
        *,
        guard: Any,
        live_bytes: int,
    ) -> tuple[bytes, int]:
        labels.append(label)
        if label == "exact_predictions":
            raise services_module._ResourceLimitExceeded("wall_clock")
        return cast(
            tuple[bytes, int],
            original(
                self,
                label,
                producer,
                guard=guard,
                live_bytes=live_bytes,
            ),
        )

    monkeypatch.setattr(
        services_module.DefaultPilotServices,
        "_serialize_benchmark_payload",
        expire_on_exact,
    )
    result = DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert labels == ["validation_scores", "holdout_features", "exact_predictions"]
    assert pd.read_parquet(store.root / "benchmark/exact_predictions.parquet").empty
    assert not [path for path in store.root.glob(".stage-benchmark-*")]
