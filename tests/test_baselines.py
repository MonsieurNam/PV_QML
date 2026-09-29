from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from pv_pilot.artifacts import RunStore, sha256_file
from pv_pilot.baselines import (
    BaselineContractError,
    BaselineModelBundle,
    BaselineTrainingResult,
    DeadlineProcessFitExecutor,
    DirectFitExecutor,
    FitDeadlineExceeded,
    FitMemoryExceeded,
    PersistenceModel,
    QuantileHeads,
    candidate_sort_key,
    fit_rff_transform,
    normalized_wis,
    postprocess_quantiles,
    prediction_parquet_bytes,
    quantile_columns,
    train_baseline_models,
    train_persistence,
    validation_scores_parquet_bytes,
    xgboost_estimator,
)
from pv_pilot.config import PilotConfig, load_config
from pv_pilot.domain import StageResult, StageStatus, deterministic_data
from pv_pilot.seeds import SeedManager
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import feature_dataframe_parquet_bytes, feature_schema_sha256

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.value)


def _raise_fit_error() -> None:
    raise ValueError("expected child failure")


@pytest.fixture
def fast_baseline_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_module: Any = importlib.import_module("pv_pilot.baselines")

    monkeypatch.setattr(
        baseline_module,
        "DeadlineProcessFitExecutor",
        lambda **kwargs: DirectFitExecutor(kwargs["resource_checkpoint"]),
    )
    import pv_pilot.baselines as module

    def fit_linear(
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
                ConstantEstimator(center + index) for index in range(len(quantiles))
            ),
        )

    def fit_xgb(
        features: np.ndarray,
        target: np.ndarray,
        quantiles: tuple[float, ...],
        *,
        max_depth: int,
        learning_rate: float,
        seed_manager: SeedManager,
        executor: object,
    ) -> QuantileHeads:
        del features, seed_manager, executor
        center = float(np.mean(target)) + max_depth + learning_rate
        return QuantileHeads(
            quantiles=quantiles,
            estimators=tuple(
                ConstantEstimator(center + index) for index in range(len(quantiles))
            ),
        )

    monkeypatch.setattr(module, "_fit_linear_heads", fit_linear)
    monkeypatch.setattr(module, "_fit_xgboost_heads", fit_xgb)


def pilot_config() -> PilotConfig:
    root = Path(__file__).parents[1]
    return load_config(root / "configs/pilot.synthetic.yaml", project_root=root).config


def feature_frame(test_target_offset: float = 0.0) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    for split_index, split in enumerate(("train", "validation", "test")):
        for row in range(12):
            value = float(split_index * 20 + row)
            target = (
                value
                + 1.0
                + (3.0 if split == "validation" else 0.0)
                + (test_target_offset if split == "test" else 0.0)
            )
            records.append(
                {
                    "sample_id": f"{split}-{row}",
                    "site_id": "site-a" if row % 2 == 0 else "site-b",
                    "issued_at_utc": start + pd.Timedelta(minutes=5 * len(records)),
                    "target_at_utc": start + pd.Timedelta(minutes=5 * len(records) + 15),
                    "split": split,
                    "target_ac_power_kw": target,
                    "capacity_kw": 100.0,
                    "is_daylight": True,
                    "power_lag_0m_kw": value,
                    "power_lag_5m_kw": value - 1,
                    "power_lag_10m_kw": value - 2,
                    "power_lag_15m_kw": value - 3,
                    "power_lag_30m_kw": value - 6,
                    "power_lag_60m_kw": value - 12,
                    "ghi_w_m2": value * 10,
                    "air_temp_c": 20 + value / 10,
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def development_and_holdout(
    *, test_target_offset: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = feature_frame(test_target_offset=test_target_offset)
    return (
        frame.loc[frame["split"].isin(("train", "validation"))].copy(),
        frame.loc[frame["split"] == "test"].copy(),
    )


def train_fixture(*, test_target_offset: float = 0.0) -> BaselineTrainingResult:
    development, holdout = development_and_holdout(
        test_target_offset=test_target_offset
    )
    return train_baseline_models(
        development,
        pilot_config(),
        holdout_loader=lambda _selection_hash: holdout,
    )


def make_feature_run(tmp_path: Path) -> RunStore:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    source = Path(__file__).parents[1] / "configs/pilot.synthetic.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"] = {
        "input_data": "../data/input.csv",
        "artifact_dir": "../artifacts",
    }
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (project / "data/input.csv").write_text("placeholder\n", encoding="utf-8")
    loaded = load_config(config_path, project_root=project)
    store = RunStore.create(loaded, now=datetime(2026, 9, 29, tzinfo=UTC))
    result = StageResult(stage="validate-data", status=StageStatus.SUCCESS)
    with store.stage("validate-data") as transaction:
        transaction.write_bytes("canonical.parquet", b"canonical")
        transaction.write_json("stage_result.json", deterministic_data(result))
    split_result = StageResult(stage="build-splits", status=StageStatus.SUCCESS)
    complete_features = feature_frame()
    development_features, holdout_features = development_and_holdout()

    def sample_hash(frame: pd.DataFrame) -> str:
        payload = json.dumps(
            frame["sample_id"].astype(str).tolist(), separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def membership_hash(frame: pd.DataFrame) -> str:
        payload = json.dumps(
            sorted(set(frame["sample_id"].astype(str))), separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with store.stage(
        "build-splits", input_paths=(Path("validate-data/canonical.parquet"),)
    ) as transaction:
        transaction.write_bytes(
            "feature_matrix.parquet",
            feature_dataframe_parquet_bytes(complete_features, loaded.config),
        )
        transaction.write_bytes(
            "development_features.parquet",
            feature_dataframe_parquet_bytes(development_features, loaded.config),
        )
        transaction.write_bytes(
            "holdout_features.parquet",
            feature_dataframe_parquet_bytes(holdout_features, loaded.config),
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
                "xgb_columns": [
                    "power_lag_0m_kw",
                    "power_lag_5m_kw",
                    "power_lag_10m_kw",
                    "power_lag_15m_kw",
                    "power_lag_30m_kw",
                    "power_lag_60m_kw",
                    "ghi_w_m2",
                    "air_temp_c",
                ],
                "split_counts": {"train": 12, "validation": 12, "test": 12},
                "partitions": {
                    "development": {
                        "path": "build-splits/development_features.parquet",
                        "splits": ["train", "validation"],
                        "row_count": 24,
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "sample_id_order_sha256": sample_hash(development_features),
                        "order_policy": ["site_id", "issued_at_utc", "sample_id"],
                    },
                    "holdout": {
                        "path": "build-splits/holdout_features.parquet",
                        "splits": ["test"],
                        "row_count": 12,
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "sample_id_order_sha256": sample_hash(holdout_features),
                        "order_policy": ["site_id", "issued_at_utc", "sample_id"],
                    },
                    "combined": {
                        "row_count": 36,
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "sample_id_membership_sha256": membership_hash(complete_features),
                        "membership_policy": "sorted_unique_sample_id",
                    },
                },
            },
        )
        transaction.write_json("minmax_scalers.json", {})
        transaction.write_json("stage_result.json", deterministic_data(split_result))
    return store


def make_pipeline_run(tmp_path: Path, sites: tuple[str, ...]) -> RunStore:
    project = tmp_path / "pipeline-project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    source = Path(__file__).parents[1] / "configs/pilot.synthetic.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"] = {
        "input_data": "../data/input.csv",
        "artifact_dir": "../artifacts",
    }
    payload["split"] = {
        "train_end_utc": "2025-01-01T02:00:00Z",
        "validation_end_utc": "2025-01-01T03:00:00Z",
        "test_end_utc": "2025-01-01T04:00:00Z",
    }
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    rows: list[dict[str, object]] = []
    times = pd.date_range("2025-01-01T00:00:00Z", periods=55, freq="5min")
    for site_index, site in enumerate(sites):
        for index, timestamp in enumerate(times):
            rows.append(
                {
                    "site_id": site,
                    "timestamp": timestamp,
                    "power_kw": float(site_index * 1000 + index),
                    "capacity_kw": 2000.0,
                    "ghi_w_m2": float(100 + index),
                    "temperature_c": float(20 + index / 10),
                    "is_daylight": True,
                }
            )
    pd.DataFrame.from_records(rows).to_csv(project / "data/input.csv", index=False)
    loaded = load_config(config_path, project_root=project)
    return RunStore.create(loaded, now=datetime(2026, 9, 29, tzinfo=UTC))


def test_wis_matches_hand_calculation_and_normalizes_by_capacity() -> None:
    forecasts = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

    # y=10: median term=2.5; weighted interval terms are 4.5, 3.4 and 2.3.
    score = normalized_wis(
        y_true=np.array([10.0]),
        quantile_predictions=forecasts,
        capacity=np.array([20.0]),
        quantiles=QUANTILES,
    )

    assert score.tolist() == pytest.approx([(2.5 + 4.5 + 3.4 + 2.3) / 3.5 / 20.0])


def test_wis_rejects_nonpositive_capacity_and_wrong_quantile_contract() -> None:
    predictions = np.ones((1, 7))
    with pytest.raises(BaselineContractError, match="capacity"):
        normalized_wis(np.array([1.0]), predictions, np.array([0.0]), QUANTILES)
    with pytest.raises(BaselineContractError, match="quantile"):
        normalized_wis(np.array([1.0]), predictions[:, :2], np.array([1.0]), (0.1, 0.9))


def test_postprocess_clips_before_stable_row_sort_and_reports_diagnostics() -> None:
    raw = np.array([[11.0, -2.0, 8.0, 4.0, 3.0, 7.0, 6.0]])
    processed, diagnostics = postprocess_quantiles(raw, np.array([10.0]))

    assert processed.tolist() == [[0.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0]]
    assert diagnostics == {
        "element_count": 7,
        "lower_clip_count": 1,
        "lower_clip_rate": pytest.approx(1 / 7),
        "pre_sort_crossing_count": 1,
        "pre_sort_crossing_rate": 1.0,
        "row_count": 1,
        "upper_clip_count": 1,
        "upper_clip_rate": pytest.approx(1 / 7),
    }


@pytest.mark.parametrize(("modes", "expected"), [(4, 10), (6, 21)])
def test_rff_dimension_and_seed_are_reproducible(modes: int, expected: int) -> None:
    values = np.arange(48, dtype=float).reshape(8, 6)[:, :modes]
    seed = SeedManager(20260928).seed("rff", f"matched{modes}", "1")
    first = fit_rff_transform(values[:5], values[5:], gamma=1.0, modes=modes, photons=2, seed=seed)
    second = fit_rff_transform(values[:5], values[5:], gamma=1.0, modes=modes, photons=2, seed=seed)

    assert first.train.shape == (5, expected)
    assert first.other.shape == (3, expected)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.other, second.other)
    assert np.array_equal(first.scaler.mean_, second.scaler.mean_)


def test_changing_rff_seed_changes_only_rff_projection() -> None:
    values = np.arange(40, dtype=float).reshape(10, 4)
    first = fit_rff_transform(values[:6], values[6:], gamma=0.1, modes=4, photons=2, seed=11)
    second = fit_rff_transform(values[:6], values[6:], gamma=0.1, modes=4, photons=2, seed=12)

    assert np.array_equal(first.scaler.mean_, second.scaler.mean_)
    assert not np.array_equal(first.train, second.train)


def test_rff_projection_is_reproducible_across_fresh_processes() -> None:
    code = (
        "import hashlib,numpy as np;"
        "from pv_pilot.baselines import fit_rff_transform;"
        "x=np.arange(40,dtype=float).reshape(10,4);"
        "z=fit_rff_transform(x[:6],x[6:],gamma=1.0,modes=4,photons=2,seed=19);"
        "print(hashlib.sha256(z.train.tobytes()+z.other.tobytes()).hexdigest())"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("branch", "left", "right"),
    [
        ("raw", {"modes": 4, "alpha": 0.0001}, {"modes": 6, "alpha": 0.1}),
        ("raw", {"modes": 4, "alpha": 0.1}, {"modes": 4, "alpha": 0.001}),
        (
            "rff",
            {"modes": 4, "alpha": 0.1, "gamma": 0.1},
            {"modes": 4, "alpha": 0.1, "gamma": 1.0},
        ),
        (
            "xgboost",
            {"max_depth": 3, "learning_rate": 0.03},
            {"max_depth": 6, "learning_rate": 0.03},
        ),
    ],
)
def test_locked_tie_break_order(
    branch: str, left: dict[str, float], right: dict[str, float]
) -> None:
    assert candidate_sort_key(branch, left, "a") < candidate_sort_key(branch, right, "z")


def test_persistence_uses_train_residuals_then_can_refit_train_validation() -> None:
    train_point = np.array([1.0, 2.0, 3.0])
    train_target = np.array([2.0, 3.0, 4.0])
    validation_point = np.array([10.0, 11.0])
    validation_target = np.array([14.0, 15.0])

    validation_model = train_persistence(train_point, train_target, QUANTILES)
    final_model = train_persistence(
        np.concatenate([train_point, validation_point]),
        np.concatenate([train_target, validation_target]),
        QUANTILES,
    )

    assert np.all(validation_model.residual_quantiles == 1.0)
    assert np.all(final_model.residual_quantiles >= validation_model.residual_quantiles)


def test_quantile_columns_are_canonical_and_stable() -> None:
    assert quantile_columns(QUANTILES) == (
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
    )


def test_model_bundle_pickle_round_trip_predictions_identically(tmp_path: Path) -> None:
    heads = QuantileHeads(
        quantiles=QUANTILES,
        estimators=tuple(ConstantEstimator(float(index)) for index in range(7)),
    )
    bundle = BaselineModelBundle(branch="raw", feature_columns=("a", "b"), predictor=heads)
    path = tmp_path / "bundle.pkl"
    path.write_bytes(pickle.dumps(bundle, protocol=5))
    restored = pickle.loads(path.read_bytes())
    values = np.ones((3, 2))

    assert isinstance(restored, BaselineModelBundle)
    assert np.array_equal(bundle.predict(values), restored.predict(values))


def test_real_xgboost_quantile_objective_smoke() -> None:
    estimator = xgboost_estimator(
        quantile=0.5,
        max_depth=3,
        learning_rate=0.03,
        seed=7,
        n_estimators=2,
    )
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = np.linspace(0.0, 1.0, 12)
    estimator.fit(x, y)

    params = estimator.get_params()
    assert estimator.predict(x).shape == (12,)
    assert params["objective"] == "reg:quantileerror"
    assert params["tree_method"] == "hist"
    assert params["n_jobs"] == 1
    assert params["subsample"] == 1.0
    assert params["colsample_bytree"] == 1.0
    assert params["reg_lambda"] == 1.0


def test_training_selects_one_shared_configuration_and_never_uses_test_for_selection(
    fast_baseline_fits: None,
) -> None:
    first = train_fixture()
    changed_test = train_fixture(test_target_offset=10_000.0)

    assert first.selection_manifest["selected"] == changed_test.selection_manifest["selected"]
    assert first.selection_manifest["selection_data"] == ["train", "validation"]
    assert first.selection_manifest["test_access"] == "after_selection_freeze_once"
    assert set(first.models) == {"persistence", "raw", "rff", "xgboost"}
    persistence = first.models["persistence"]
    assert isinstance(persistence, PersistenceModel)
    assert persistence.residual_quantiles[3] == pytest.approx(2.5)
    for branch in ("raw", "rff", "xgboost"):
        selected = first.selection_manifest["selected"][branch]
        assert isinstance(selected, dict)
        assert "validation_mean_normalized_wis" in selected
    assert set(first.predictions["split"]) == {"test"}
    assert set(first.predictions["model"]) == {"persistence", "raw", "rff", "xgboost"}
    assert len(first.predictions) == 4 * 12


def test_holdout_is_opened_once_only_after_selection_freezes_and_fits_complete(
    fast_baseline_fits: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.baselines as module

    development, holdout = development_and_holdout()
    poisoned = holdout.copy()
    poisoned["target_ac_power_kw"] = np.nan
    original_fit = module._fit_linear_heads
    fit_count = 0
    loader_calls: list[str] = []

    def counted_fit(*args: object, **kwargs: object) -> QuantileHeads:
        nonlocal fit_count
        fit_count += 1
        return original_fit(*args, **kwargs)  # type: ignore[arg-type]

    def load_holdout(selection_freeze_sha256: str) -> pd.DataFrame:
        loader_calls.append(selection_freeze_sha256)
        assert fit_count == 34  # all 32 candidates plus two selected refits
        assert len(selection_freeze_sha256) == 64
        return poisoned

    monkeypatch.setattr(module, "_fit_linear_heads", counted_fit)
    with pytest.raises(BaselineContractError, match="holdout"):
        train_baseline_models(
            development,
            pilot_config(),
            holdout_loader=load_holdout,
        )

    assert len(loader_calls) == 1


def test_resource_checkpoint_runs_through_candidate_and_refit_boundaries(
    fast_baseline_fits: None,
) -> None:
    development, holdout = development_and_holdout()
    checkpoints: list[int] = []

    train_baseline_models(
        development,
        pilot_config(),
        holdout_loader=lambda _selection_hash: holdout,
        resource_checkpoint=checkpoints.append,
    )

    assert len(checkpoints) >= 250
    assert all(value >= 0 for value in checkpoints)


def test_deadline_fit_executor_terminates_long_fit_without_lingering_child(
    tmp_path: Path,
) -> None:
    executor = DeadlineProcessFitExecutor(
        deadline_monotonic=time.monotonic() + 0.25,
        resource_checkpoint=lambda _declared: None,
        private_directory=tmp_path,
    )
    started = time.monotonic()

    with pytest.raises(FitDeadlineExceeded, match="deadline"):
        executor.run("sleeping-fit", time.sleep, 5.0)

    assert time.monotonic() - started < 2.0
    assert executor.active_child_pid is None
    assert not list(tmp_path.iterdir())


def test_deadline_fit_executor_enforces_child_rss_and_cleans_private_files(
    tmp_path: Path,
) -> None:
    executor = DeadlineProcessFitExecutor(
        deadline_monotonic=time.monotonic() + 5.0,
        resource_checkpoint=lambda _declared: None,
        private_directory=tmp_path,
        max_memory_bytes=1,
    )

    with pytest.raises(FitMemoryExceeded, match="RSS"):
        executor.run("memory-capped-fit", time.sleep, 5.0)

    assert executor.active_child_pid is None
    assert not list(tmp_path.iterdir())


def test_fit_executor_counts_each_child_descendant_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_module: Any = importlib.import_module("pv_pilot.baselines")
    processes: dict[int, tuple[int, list[int]]] = {
        10: (200, [11, 12, 11]),
        11: (300, []),
        12: (400, []),
    }

    class MemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakePsutilProcess:
        def __init__(self, pid: int | None = None) -> None:
            self.pid = 1 if pid is None else pid

        def memory_info(self) -> MemoryInfo:
            if self.pid == 1:
                return MemoryInfo(100)
            return MemoryInfo(processes[self.pid][0])

        def children(self, *, recursive: bool = False) -> list[FakePsutilProcess]:
            assert recursive
            return [FakePsutilProcess(pid) for pid in processes[self.pid][1]]

    monkeypatch.setattr(baseline_module.psutil, "Process", FakePsutilProcess)
    executor = DeadlineProcessFitExecutor(
        deadline_monotonic=time.monotonic() + 5.0,
        resource_checkpoint=lambda _declared: None,
        private_directory=tmp_path,
        max_memory_bytes=2_000,
    )

    executor._sample_memory(10)

    assert executor.peak_parent_rss_bytes == 100
    assert executor.peak_child_tree_rss_bytes == 900
    assert executor.peak_combined_rss_bytes == 1_000


def test_train_baselines_aggregate_fit_rss_exceedance_is_invalid_and_telemetried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_module: Any = importlib.import_module("pv_pilot.baselines")

    store = make_feature_run(tmp_path)
    per_process_rss = 5 * 1024**3

    class MemoryInfo:
        rss = per_process_rss

    class FakePsutilProcess:
        def __init__(self, _pid: int | None = None) -> None:
            pass

        def memory_info(self) -> MemoryInfo:
            return MemoryInfo()

        def children(self, *, recursive: bool = False) -> list[FakePsutilProcess]:
            del recursive
            return []

    monkeypatch.setattr(baseline_module.psutil, "Process", FakePsutilProcess)

    result = DefaultPilotServices().train_baselines(store.root)

    assert result.status is StageStatus.INVALID
    ledger = json.loads(
        (store.root / "train-baselines/resource_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["resource_gate"] == "fit_memory_rss"
    assert ledger["resource_status"] == "EXCEEDED"
    assert ledger["max_memory_bytes"] == 8 * 1024**3
    assert ledger["peak_parent_rss_bytes"] == per_process_rss
    assert ledger["peak_child_tree_rss_bytes"] == per_process_rss
    assert ledger["peak_combined_rss_bytes"] == 2 * per_process_rss
    assert ledger["peak_parent_rss_bytes"] < 8 * 1024**3
    assert ledger["peak_child_tree_rss_bytes"] < 8 * 1024**3
    assert ledger["peak_combined_rss_bytes"] > 8 * 1024**3
    assert not list(store.root.glob(".fit-operation-*"))


def test_deadline_fit_executor_propagates_child_error_and_cleans_private_files(
    tmp_path: Path,
) -> None:
    executor = DeadlineProcessFitExecutor(
        deadline_monotonic=time.monotonic() + 30.0,
        resource_checkpoint=lambda _declared: None,
        private_directory=tmp_path,
    )

    with pytest.raises(RuntimeError, match="ValueError: expected child failure"):
        executor.run("failing-fit", _raise_fit_error)

    assert executor.active_child_pid is None
    assert not list(tmp_path.iterdir())


def test_selected_raw_and_rff_scalers_are_fit_on_train_only(fast_baseline_fits: None) -> None:
    result = train_fixture()
    raw = result.models["raw"]
    rff = result.models["rff"]
    assert isinstance(raw, BaselineModelBundle)
    assert isinstance(rff, BaselineModelBundle)
    assert raw.scaler is not None
    assert rff.scaler is not None

    selected_raw = result.selection_manifest["selected"]["raw"]
    selected_rff = result.selection_manifest["selected"]["rff"]
    raw_columns = tuple(selected_raw["feature_columns"])
    rff_columns = tuple(selected_rff["feature_columns"])
    train = feature_frame().loc[lambda frame: frame["split"] == "train"]
    assert raw.scaler.mean_ == pytest.approx(train.loc[:, raw_columns].mean().to_numpy())
    assert rff.scaler.mean_ == pytest.approx(train.loc[:, rff_columns].mean().to_numpy())


def test_candidate_records_use_common_alpha_and_xgboost_tuple_for_all_heads(
    fast_baseline_fits: None,
) -> None:
    result = train_fixture()

    for candidate in result.selection_manifest["candidates"]:
        if candidate["branch"] in {"raw", "rff"}:
            assert candidate["head_alphas"] == [candidate["alpha"]] * 7
        if candidate["branch"] == "xgboost":
            assert candidate["head_max_depths"] == [candidate["max_depth"]] * 7
            assert candidate["head_learning_rates"] == [candidate["learning_rate"]] * 7


def test_prediction_and_validation_schemas_are_stable_for_empty_frames(
    fast_baseline_fits: None,
) -> None:
    predictions = train_fixture().predictions
    scores = pd.DataFrame.from_records(
        train_fixture().validation_scores
    )

    prediction_schema = pq.read_schema(pa.BufferReader(prediction_parquet_bytes(predictions)))
    empty_prediction_schema = pq.read_schema(
        pa.BufferReader(prediction_parquet_bytes(predictions.iloc[0:0]))
    )
    score_schema = pq.read_schema(pa.BufferReader(validation_scores_parquet_bytes(scores)))
    empty_score_schema = pq.read_schema(
        pa.BufferReader(validation_scores_parquet_bytes(scores.iloc[0:0]))
    )

    assert prediction_schema == empty_prediction_schema
    assert score_schema == empty_score_schema
    assert prediction_schema.field("issued_at_utc").type == pa.timestamp("ns", tz="UTC")
    assert prediction_schema.names[-7:] == list(quantile_columns(QUANTILES))


def test_train_baselines_service_publishes_complete_artifacts_and_resumes(
    tmp_path: Path,
    fast_baseline_fits: None,
) -> None:
    store = make_feature_run(tmp_path)
    service = DefaultPilotServices()

    first = service.train_baselines(store.root)
    second = service.train_baselines(store.root)

    assert first.status is StageStatus.SUCCESS
    assert second.status is StageStatus.SUCCESS
    assert "already successful" in (second.message or "")
    expected = {
        "selection_manifest.json",
        "predictions.parquet",
        "validation_scores.parquet",
        "diagnostics.json",
        "stage_result.json",
        "resource_ledger.json",
        "stage_manifest.json",
    }
    assert expected <= {path.name for path in (store.root / "train-baselines").iterdir()}
    store.verify_terminal_stage("train-baselines")
    predictions = pd.read_parquet(store.root / "train-baselines/predictions.parquet")
    assert set(predictions["model"]) == {"persistence", "raw", "rff", "xgboost"}
    assert len(predictions) == 48
    model_dir = store.root / "train-baselines/models"
    assert {"baseline_models.pkl", "manifest.json"} <= {
        path.name for path in model_dir.iterdir()
    }
    models = pickle.loads((model_dir / "baseline_models.pkl").read_bytes())
    assert set(models) == {"persistence", "raw", "rff", "xgboost"}
    manifest = json.loads(
        (model_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["serialization_roundtrip_verified"] is True
    assert manifest["scientific_decision"] == "NOT_EVALUATED"


@pytest.mark.parametrize(
    "sites",
    [("site-a",), ("site-b", "site-a")],
    ids=["single-site", "exact-two-site"],
)
def test_validate_build_train_pipeline_preserves_partition_membership(
    tmp_path: Path,
    fast_baseline_fits: None,
    sites: tuple[str, ...],
) -> None:
    store = make_pipeline_run(tmp_path, sites)
    service = DefaultPilotServices()

    assert service.validate_data(store.root).status is StageStatus.SUCCESS
    assert service.build_splits(store.root).status is StageStatus.SUCCESS
    assert service.train_baselines(store.root).status is StageStatus.SUCCESS

    development = pd.read_parquet(
        store.root / "build-splits/development_features.parquet"
    )
    holdout = pd.read_parquet(store.root / "build-splits/holdout_features.parquet")
    assert set(development["sample_id"]).isdisjoint(set(holdout["sample_id"]))
    assert set(development["site_id"]) == set(sites)
    assert set(holdout["site_id"]) == set(sites)
    stage_marker = json.loads(
        (store.root / "train-baselines/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert {
        "build-splits/development_features.parquet",
        "build-splits/holdout_features.parquet",
    } <= set(stage_marker["input_hashes"])


def test_service_scans_development_before_lazy_holdout_and_opens_holdout_once(
    tmp_path: Path,
    fast_baseline_fits: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.baselines as baseline_module
    services_module: Any = importlib.import_module("pv_pilot.services")

    store = make_feature_run(tmp_path)
    original = baseline_module.train_baseline_models
    original_dataset = services_module.pds.dataset
    original_sha256 = services_module.sha256_file
    events: list[str] = []
    partition_reads: list[str] = []

    def audited_dataset(path: object, *args: object, **kwargs: object) -> object:
        partition_reads.append(Path(path).name)  # type: ignore[arg-type]
        return original_dataset(path, *args, **kwargs)

    def audited_sha256(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            partition_reads.append("holdout_checksum")
        return str(original_sha256(path))

    def audited_train(
        development_frame: pd.DataFrame,
        config: PilotConfig,
        *,
        holdout_loader: Callable[[str], pd.DataFrame],
        **_kwargs: object,
    ) -> BaselineTrainingResult:
        assert set(development_frame["split"]) == {"train", "validation"}
        assert partition_reads == ["development_features.parquet"]
        events.append("development_received")

        def audited_loader(selection_hash: str) -> pd.DataFrame:
            events.append("holdout_opened")
            assert len(selection_hash) == 64
            return holdout_loader(selection_hash)

        return original(
            development_frame,
            config,
            holdout_loader=audited_loader,
        )

    monkeypatch.setattr(baseline_module, "train_baseline_models", audited_train)
    monkeypatch.setattr(services_module.pds, "dataset", audited_dataset)
    monkeypatch.setattr(services_module, "sha256_file", audited_sha256)
    result = DefaultPilotServices().train_baselines(store.root)

    assert result.status is StageStatus.SUCCESS
    assert events == ["development_received", "holdout_opened"]
    assert partition_reads[:3] == [
        "development_features.parquet",
        "holdout_checksum",
        "holdout_features.parquet",
    ]
    selection = json.loads(
        (store.root / "train-baselines/selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    audit = selection["holdout_audit"]
    assert audit["event_order"]["selection_frozen"] < audit["event_order"]["holdout_opened"]
    assert audit["prediction_passes"] == 1
    assert audit["opened_at_utc"].endswith("Z")
    assert len(audit["row_content_sha256"]) == 64


def test_train_baselines_rejects_feature_input_tamper(
    tmp_path: Path,
    fast_baseline_fits: None,
) -> None:
    store = make_feature_run(tmp_path)
    feature_path = store.root / "build-splits/development_features.parquet"
    feature_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        DefaultPilotServices().train_baselines(store.root)

    assert not (store.root / "train-baselines/stage_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["missing-and-extra", "duplicate"])
def test_train_baselines_rejects_invalid_combined_partition_membership(
    tmp_path: Path,
    fast_baseline_fits: None,
    mutation: str,
) -> None:
    store = make_feature_run(tmp_path)
    holdout_path = store.root / "build-splits/holdout_features.parquet"
    manifest_path = store.root / "build-splits/feature_manifest.json"
    marker_path = store.root / "build-splits/stage_manifest.json"
    holdout = pd.read_parquet(holdout_path)
    if mutation == "missing-and-extra":
        holdout.loc[holdout.index[0], "sample_id"] = "unexpected-extra-sample"
    else:
        holdout = pd.concat([holdout, holdout.iloc[[0]]], ignore_index=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout_ids = holdout["sample_id"].astype(str).tolist()
    manifest["partitions"]["holdout"]["row_count"] = len(holdout)
    manifest["partitions"]["holdout"]["sample_id_order_sha256"] = hashlib.sha256(
        json.dumps(holdout_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if mutation == "duplicate":
        manifest["partitions"]["combined"]["row_count"] += 1
    holdout_path.write_bytes(feature_dataframe_parquet_bytes(holdout, pilot_config()))
    manifest_path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    replacement_hashes = {
        "build-splits/holdout_features.parquet": sha256_file(holdout_path),
        "build-splits/feature_manifest.json": sha256_file(manifest_path),
    }
    for output in marker["outputs"]:
        if output["path"] in replacement_hashes:
            output["sha256"] = replacement_hashes[output["path"]]
    marker_path.write_text(
        json.dumps(
            marker,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = DefaultPilotServices().train_baselines(store.root)

    assert result.status is StageStatus.INVALID
    assert "contract" in (result.message or "")
    store.verify_terminal_stage(
        "train-baselines", accepted_statuses=frozenset({"INVALID"})
    )


def test_train_baselines_resource_limit_publishes_bounded_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.services as services_module

    store = make_feature_run(tmp_path)

    def exceed(self: object, declared_bytes: int = 0) -> None:
        del self, declared_bytes
        raise services_module._ResourceLimitExceeded("wall_clock")

    monkeypatch.setattr(services_module._ResourceGuard, "checkpoint", exceed)
    result = DefaultPilotServices().train_baselines(store.root)

    assert result.status is StageStatus.INVALID
    store.verify_terminal_stage("train-baselines", accepted_statuses=frozenset({"INVALID"}))
    ledger = json.loads(
        (store.root / "train-baselines/resource_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["resource_status"] == "EXCEEDED"
    assert pd.read_parquet(store.root / "train-baselines/predictions.parquet").empty


def test_train_baselines_hard_deadline_is_durable_without_pre_freeze_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.baselines as baseline_module

    services_module: Any = importlib.import_module("pv_pilot.services")
    store = make_feature_run(tmp_path)
    original_sha256 = services_module.sha256_file
    executor_created = False

    class AlwaysDeadlineExecutor:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal executor_created
            executor_created = True

        def run(self, *_args: object, **_kwargs: object) -> object:
            raise FitDeadlineExceeded("injected hard deadline")

    def reject_holdout_hash(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("deadline INVALID publication read the holdout")
        return str(original_sha256(path))

    monkeypatch.setattr(
        baseline_module, "DeadlineProcessFitExecutor", AlwaysDeadlineExecutor
    )
    monkeypatch.setattr(services_module, "sha256_file", reject_holdout_hash)

    result = DefaultPilotServices().train_baselines(store.root)

    assert executor_created
    assert result.status is StageStatus.INVALID
    marker = json.loads(
        (store.root / "train-baselines/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "INVALID"
    assert marker["details"]["resource_gate"] == "fit_wall_clock_deadline"
    assert marker["details"]["prerequisite_verification"] == "marker_only"
    assert not list(store.root.glob(".fit-operation-*"))


def test_train_baselines_contract_error_publishes_invalid_not_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pv_pilot.baselines as baseline_module

    store = make_feature_run(tmp_path)

    def reject(
        frame: pd.DataFrame,
        config: object,
        *,
        holdout_loader: object,
        **_kwargs: object,
    ) -> object:
        del frame, config, holdout_loader
        raise BaselineContractError("bad split schema")

    monkeypatch.setattr(baseline_module, "train_baseline_models", reject)
    result = DefaultPilotServices().train_baselines(store.root)

    assert result.status is StageStatus.INVALID
    assert "bad split schema" in (result.message or "")
    store.verify_terminal_stage("train-baselines", accepted_statuses=frozenset({"INVALID"}))


def test_train_baselines_resource_limit_during_commit_rolls_back_then_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_baseline_fits: None,
) -> None:
    import inspect

    import pv_pilot.services as services_module

    store = make_feature_run(tmp_path)
    original = services_module._ResourceGuard.checkpoint
    raised = False

    def exceed_in_commit(self: object, declared_bytes: int = 0) -> None:
        nonlocal raised
        in_commit = any(
            frame.function == "__exit__"
            and getattr(frame.frame.f_locals.get("self"), "stage", None) == "train-baselines"
            for frame in inspect.stack()
        )
        if in_commit and not raised:
            raised = True
            raise services_module._ResourceLimitExceeded("wall_clock")
        original(self, declared_bytes)  # type: ignore[arg-type]

    monkeypatch.setattr(services_module._ResourceGuard, "checkpoint", exceed_in_commit)
    result = DefaultPilotServices().train_baselines(store.root)

    assert raised
    assert result.status is StageStatus.INVALID
    store.verify_terminal_stage("train-baselines", accepted_statuses=frozenset({"INVALID"}))
