from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pv_pilot.domain import (
    CanonicalRow,
    CompletionKind,
    DataProvenance,
    DaylightSource,
    DetailEntry,
    FrozenJsonObject,
    PhotonicFeatureSet,
    PredictionRecord,
    QuantileValue,
    StageResult,
    StageStatus,
    deterministic_json,
    freeze_json,
)


def test_domain_enums_use_stable_serializable_values() -> None:
    assert [status.value for status in StageStatus] == [
        "success",
        "blocked",
        "invalid",
        "failed",
    ]
    assert [kind.value for kind in DataProvenance] == ["real", "synthetic"]
    assert [source.value for source in DaylightSource] == ["solar_elevation", "column"]


def test_domain_records_are_versioned_frozen_and_serializable() -> None:
    result = StageResult(stage="validate-data", status=StageStatus.SUCCESS)
    row = CanonicalRow(
        site_id="site-a",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        power=10.0,
        capacity=20.0,
        ghi=500.0,
        temperature=25.0,
    )

    assert result.schema_version == "1.0"
    assert result.model_dump(mode="json")["status"] == "success"
    assert row.model_dump_json()
    with pytest.raises(ValidationError):
        result.status = StageStatus.FAILED


def test_stage_result_completion_kind_is_typed_and_backwards_compatible() -> None:
    legacy = StageResult.model_validate_json(
        '{"schema_version":"1.0","stage":"report","status":"blocked"}'
    )
    marked = legacy.model_copy(
        update={"completion_kind": CompletionKind.SYNTHETIC_SOFTWARE_ACCEPTANCE}
    )

    assert legacy.completion_kind is None
    assert marked.model_dump(mode="json")["completion_kind"] == (
        "synthetic_software_acceptance"
    )
    assert StageResult.model_validate_json(marked.model_dump_json()) == marked


def test_domain_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StageResult.model_validate(
            {"stage": "validate-data", "status": StageStatus.SUCCESS, "surprise": True}
        )


def test_domain_records_do_not_coerce_scalar_payloads() -> None:
    with pytest.raises(ValidationError):
        CanonicalRow.model_validate(
            {
                "site_id": "site-a",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "power": "10.0",
                "capacity": 20.0,
                "ghi": 500.0,
                "temperature": 25.0,
            }
        )


def test_domain_payloads_are_deeply_immutable_and_deterministic() -> None:
    result = StageResult(
        stage="validate-data",
        status=StageStatus.SUCCESS,
        details=(DetailEntry(key="counts", value=(1, ("valid", True))),),
    )
    first = deterministic_json(result)
    details: Any = result.details
    first_detail: Any = result.details[0]

    with pytest.raises((ValidationError, FrozenInstanceError)):
        first_detail.key = "changed"
    with pytest.raises(AttributeError):
        details.append(DetailEntry(key="x", value=1))
    assert deterministic_json(result) == first


def test_detail_objects_are_recursively_frozen_sorted_and_canonical() -> None:
    first = StageResult(
        stage="validate-data",
        status=StageStatus.SUCCESS,
        details=(
            DetailEntry(key="z", value={"b": 2, "a": [1, {"y": True, "x": None}]}),
            DetailEntry(key="a", value="first"),
        ),
    )
    second = StageResult(
        stage="validate-data",
        status=StageStatus.SUCCESS,
        details=(
            DetailEntry(key="a", value="first"),
            DetailEntry(key="z", value={"a": [1, {"x": None, "y": True}], "b": 2}),
        ),
    )

    assert isinstance(first.details[1].value, FrozenJsonObject)
    assert freeze_json({"b": 2, "a": 1}) == freeze_json({"a": 1, "b": 2})
    assert deterministic_json(first) == deterministic_json(second)
    nested: Any = first.details[1].value
    with pytest.raises((FrozenInstanceError, AttributeError)):
        nested.entries = ()


def test_duplicate_detail_and_nested_object_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        StageResult(
            stage="validate-data",
            status=StageStatus.SUCCESS,
            details=(DetailEntry("same", 1), DetailEntry("same", 2)),
        )
    with pytest.raises(ValueError, match="duplicate"):
        FrozenJsonObject((DetailEntry("same", 1), DetailEntry("same", 2)))


def test_stage_result_standard_and_deterministic_json_round_trip_details() -> None:
    record = StageResult(
        stage="validate-data",
        status=StageStatus.SUCCESS,
        details=(
            DetailEntry(
                "summary",
                {
                    "count": 2,
                    "items": [1, {"valid": True, "reason": None}],
                },
            ),
        ),
    )

    for serialized in (record.model_dump_json(), record.deterministic_json()):
        payload = json.loads(serialized)

        assert payload["details"] == {
            "summary": {
                "count": 2,
                "items": [1, {"reason": None, "valid": True}],
            }
        }
        assert "entries" not in serialized
        assert StageResult.model_validate_json(serialized) == record


def test_artifact_paths_use_portable_posix_json_and_round_trip() -> None:
    windows_style = Path(r"artifacts\circuits\features.parquet")
    stage = StageResult(
        stage="build-photonic-features",
        status=StageStatus.SUCCESS,
        artifacts=(windows_style,),
    )
    features = PhotonicFeatureSet(
        artifact_path=windows_style,
        checksum_sha256="0" * 64,
        backend="SLOS",
        modes=4,
        photons=2,
        phase_scale=1.0,
        circuit_seed=1,
        feature_names=("p0",),
        row_count=1,
        feature_count=1,
    )

    for record in (stage, features):
        for serialized in (record.model_dump_json(), record.deterministic_json()):
            assert "artifacts/circuits/features.parquet" in serialized
            assert "\\\\" not in serialized
        assert type(record).model_validate_json(record.model_dump_json()) == record
        assert type(record).model_validate_json(record.deterministic_json()) == record


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_domain_numeric_payloads_reject_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValidationError):
        QuantileValue(q=0.5, value=invalid)
    with pytest.raises(ValidationError):
        QuantileValue(q=invalid, value=1.0)
    with pytest.raises(ValidationError):
        CanonicalRow(
            site_id="site-a",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            power=invalid,
            capacity=20.0,
            ghi=500.0,
            temperature=25.0,
        )


def test_domain_timestamps_must_be_aware_and_are_normalized_to_utc() -> None:
    local = timezone(timedelta(hours=7))
    row = CanonicalRow(
        site_id="site-a",
        timestamp=datetime(2026, 1, 1, 7, tzinfo=local),
        power=10.0,
        capacity=20.0,
        ghi=500.0,
        temperature=25.0,
        power_available_at=datetime(2026, 1, 1, 6, tzinfo=local),
    )
    assert row.timestamp == datetime(2026, 1, 1, tzinfo=UTC)
    assert row.power_available_at == datetime(2025, 12, 31, 23, tzinfo=UTC)

    with pytest.raises(ValidationError):
        CanonicalRow(
            site_id="site-a",
            timestamp=datetime(2026, 1, 1),
            power=10.0,
            capacity=20.0,
            ghi=500.0,
            temperature=25.0,
        )


def test_prediction_timing_contract_and_quantile_records() -> None:
    issued = datetime(2026, 1, 1, tzinfo=UTC)
    record = PredictionRecord(
        site_id="site-a",
        issued_at=issued,
        target_interval_start=issued + timedelta(minutes=10),
        target_interval_end=issued + timedelta(minutes=15),
        model_version="v1",
        quantiles=(QuantileValue(q=0.5, value=10.0),),
        status=StageStatus.SUCCESS,
    )
    assert record.quantiles[0].q == 0.5

    with pytest.raises(ValidationError):
        PredictionRecord(
            site_id="site-a",
            issued_at=issued,
            target_interval_start=issued + timedelta(minutes=9),
            target_interval_end=issued + timedelta(minutes=15),
            model_version="v1",
            quantiles=(),
            status=StageStatus.SUCCESS,
        )


def test_prediction_quantiles_are_nonempty_unique_sorted_and_nondecreasing() -> None:
    issued = datetime(2026, 1, 1, tzinfo=UTC)

    def prediction(quantiles: tuple[QuantileValue, ...]) -> PredictionRecord:
        return PredictionRecord(
            site_id="site-a",
            issued_at=issued,
            target_interval_start=issued + timedelta(minutes=10),
            target_interval_end=issued + timedelta(minutes=15),
            model_version="v1",
            quantiles=quantiles,
            status=StageStatus.SUCCESS,
        )

    ordered = prediction(
        (QuantileValue(q=0.1, value=1.0), QuantileValue(q=0.9, value=9.0))
    )
    permuted = prediction(
        (QuantileValue(q=0.9, value=9.0), QuantileValue(q=0.1, value=1.0))
    )
    assert permuted.quantiles == ordered.quantiles
    assert deterministic_json(permuted) == deterministic_json(ordered)

    with pytest.raises(ValidationError):
        prediction(())
    with pytest.raises(ValidationError, match="unique"):
        prediction((QuantileValue(q=0.5, value=1.0), QuantileValue(q=0.5, value=2.0)))
    with pytest.raises(ValidationError, match="nondecreasing"):
        prediction((QuantileValue(q=0.1, value=2.0), QuantileValue(q=0.9, value=1.0)))


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_quantile_probabilities_are_strictly_interior(q: float) -> None:
    with pytest.raises(ValidationError):
        QuantileValue(q=q, value=1.0)
