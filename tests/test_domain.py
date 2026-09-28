from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from pv_pilot.domain import (
    CanonicalRow,
    DataProvenance,
    DaylightSource,
    DetailEntry,
    PredictionRecord,
    QuantileValue,
    StageResult,
    StageStatus,
    deterministic_json,
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

    with pytest.raises(ValidationError):
        result.details[0].key = "changed"
    with pytest.raises(AttributeError):
        details.append(DetailEntry(key="x", value=1))
    assert deterministic_json(result) == first


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
