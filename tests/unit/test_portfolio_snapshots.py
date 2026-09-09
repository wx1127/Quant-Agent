"""Immutable account valuation, quantity partitions, hashes, and JSON round trips."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

from quant_agent.backtest import TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.portfolio import (
    AccountSnapshot,
    CashSnapshot,
    PortfolioPositionSnapshot,
    ValuationStatus,
)
from quant_agent.regime.contracts import stable_hash

VALUATION_AT = datetime.combine(date(2026, 8, 28), time(15), tzinfo=SHANGHAI_TZ)
AS_OF = datetime.combine(date(2026, 8, 28), time(18), tzinfo=SHANGHAI_TZ)
MARK_POLICY_HASH = stable_hash({"mark_policy": "position-mark-v1"})


def _position(
    instrument_id: str = "CN.SSE.600000",
    *,
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    available: str = "700",
    frozen: str = "100",
    unsettled: str = "200",
    average_cost: str = "10",
    valuation_price: str = "12",
    stale: bool = False,
) -> PortfolioPositionSnapshot:
    observed_at = VALUATION_AT - timedelta(days=1) if stale else VALUATION_AT
    return PortfolioPositionSnapshot.build(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        available_quantity=Decimal(available),
        frozen_quantity=Decimal(frozen),
        unsettled_quantity=Decimal(unsettled),
        average_cost=Decimal(average_cost),
        valuation_price=Decimal(valuation_price),
        price_observed_at=observed_at,
        price_available_at=observed_at + timedelta(hours=1),
        position_as_of=AS_OF,
        price_data_version="snapshot-v1",
        price_source_hash=stable_hash({"instrument_id": instrument_id, "observed_at": observed_at}),
        valuation_status=(ValuationStatus.STALE if stale else ValuationStatus.FRESH),
        mark_policy_hash=MARK_POLICY_HASH,
    )


def _cash(total: str = "5000", frozen: str = "500") -> CashSnapshot:
    total_value = Decimal(total)
    frozen_value = Decimal(frozen)
    return CashSnapshot(
        as_of=AS_OF,
        currency="CNY",
        total_cash=total_value,
        available_cash=total_value - frozen_value,
        frozen_cash=frozen_value,
    )


def _account(
    *,
    positions: tuple[PortfolioPositionSnapshot, ...] | None = None,
    cash: CashSnapshot | None = None,
) -> AccountSnapshot:
    position_values = (
        positions
        if positions is not None
        else (
            _position(
                "CN.SSE.510300",
                instrument_type=TradableInstrumentType.ETF,
                available="100",
                frozen="0",
                unsettled="0",
                average_cost="2",
                valuation_price="2.5",
            ),
            _position(),
        )
    )
    return AccountSnapshot.build(
        snapshot_id="account-snapshot-20260828",
        account_id="paper-account-1",
        runtime_mode=RuntimeMode.PAPER,
        as_of=AS_OF,
        valuation_at=VALUATION_AT,
        data_version="snapshot-v1",
        currency="CNY",
        cash=cash or _cash(),
        positions=position_values,
        previous_snapshot_hash="a" * 64,
        source_event_log_hash="b" * 64,
    )


def test_account_snapshot_separates_cash_and_quantity_states_and_revalues_exactly() -> None:
    account = _account()
    positions = {item.instrument_id: item for item in account.positions}
    stock = positions["CN.SSE.600000"]
    etf = positions["CN.SSE.510300"]

    assert stock.total_quantity == Decimal(1000)
    assert stock.available_quantity == Decimal(700)
    assert stock.frozen_quantity == Decimal(100)
    assert stock.unsettled_quantity == Decimal(200)
    assert stock.market_value == Decimal(12000)
    assert stock.cost_basis == Decimal(10000)
    assert stock.unrealized_pnl == Decimal(2000)
    assert etf.market_value == Decimal(250)
    assert account.cash.available_cash == Decimal(4500)
    assert account.cash.frozen_cash == Decimal(500)
    assert account.position_market_value == Decimal(12250)
    assert account.total_equity == Decimal(17250)
    assert account.gross_exposure + account.cash_weight == Decimal(1)
    assert len(account.content_hash) == 64
    assert account.identity_payload()["content_hash"] == account.content_hash


def test_snapshot_is_deeply_immutable_and_repeated_build_is_identical() -> None:
    first = _account()
    repeated = _account()

    assert first == repeated
    with pytest.raises(FrozenInstanceError):
        first.total_equity = Decimal(1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.cash.available_cash = Decimal(1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.positions[0].available_quantity = Decimal(1)  # type: ignore[misc]


def test_canonical_json_round_trip_revalidates_content_hash() -> None:
    account = _account()
    encoded = account.to_json()
    decoded = AccountSnapshot.from_json(encoded)

    assert decoded == account
    assert decoded.to_json() == encoded
    assert " " not in encoded
    payload = json.loads(encoded)
    payload["snapshot_id"] = "tampered-snapshot"
    with pytest.raises(ValueError, match="content_hash"):
        AccountSnapshot.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="JSON is invalid"):
        AccountSnapshot.from_json("{")
    with pytest.raises(ValueError, match="must contain an object"):
        AccountSnapshot.from_json("[]")


def test_json_decoder_rejects_unknown_root_cash_and_position_fields() -> None:
    payload = json.loads(_account().to_json())
    payload["unexpected"] = "not hashed"
    with pytest.raises(ValueError, match="fields do not match"):
        AccountSnapshot.from_json(json.dumps(payload))

    payload = json.loads(_account().to_json())
    payload["cash"]["unexpected"] = "not hashed"
    with pytest.raises(ValueError, match="cash fields"):
        AccountSnapshot.from_json(json.dumps(payload))

    payload = json.loads(_account().to_json())
    payload["positions"][0]["unexpected"] = "not hashed"
    with pytest.raises(ValueError, match="position fields"):
        AccountSnapshot.from_json(json.dumps(payload))


def test_stale_or_suspended_marks_are_explicit_and_future_prices_fail_closed() -> None:
    stale = _position(stale=True)
    account = _account(positions=(stale,))
    assert account.positions[0].valuation_status is ValuationStatus.STALE
    assert account.positions[0].price_observed_at < account.valuation_at

    with pytest.raises(ValueError, match="FRESH position marks"):
        _account(positions=(replace(stale, valuation_status=ValuationStatus.FRESH),))
    with pytest.raises(ValueError, match="future valuation prices"):
        _account(positions=(replace(stale, price_available_at=AS_OF + timedelta(seconds=1)),))
    with pytest.raises(ValueError, match="position_as_of"):
        _account(positions=(replace(stale, position_as_of=AS_OF - timedelta(seconds=1)),))


def test_position_and_cash_contracts_reject_incoherent_partitions_or_derived_values() -> None:
    position = _position()
    with pytest.raises(ValueError, match="must equal available"):
        replace(position, total_quantity=Decimal(999))
    fractional = _position(available="1.5", frozen="0", unsettled="0")
    assert fractional.total_quantity == Decimal("1.5")
    with pytest.raises(ValueError, match="non-negative"):
        _position(available="-1", frozen="0", unsettled="0")
    with pytest.raises(ValueError, match="must be positive"):
        _position(available="0", frozen="0", unsettled="0")
    with pytest.raises(ValueError, match="market_value"):
        replace(position, market_value=position.market_value + Decimal(1))
    with pytest.raises(ValueError, match="must equal available_cash"):
        CashSnapshot(
            as_of=AS_OF,
            currency="CNY",
            total_cash=Decimal(100),
            available_cash=Decimal(80),
            frozen_cash=Decimal(10),
        )


def test_account_rejects_unsorted_duplicate_or_misaligned_sources() -> None:
    stock = _position()
    etf = _position(
        "CN.SSE.510300",
        instrument_type=TradableInstrumentType.ETF,
        available="100",
        frozen="0",
        unsettled="0",
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        _account(positions=(stock, etf))
    with pytest.raises(ValueError, match="unique and sorted"):
        _account(positions=(stock, stock))
    with pytest.raises(ValueError, match="cash time and currency"):
        _account(cash=replace(_cash(), as_of=AS_OF - timedelta(seconds=1)))
    with pytest.raises(ValueError, match="RuntimeMode"):
        AccountSnapshot.build(
            snapshot_id="bad-mode",
            account_id="paper-account-1",
            runtime_mode="PAPER",  # type: ignore[arg-type]
            as_of=AS_OF,
            valuation_at=VALUATION_AT,
            data_version="snapshot-v1",
            currency="CNY",
            cash=_cash(),
            positions=(),
        )


def test_cash_only_account_is_valid_and_economic_or_hash_tampering_is_rejected() -> None:
    empty = _account(positions=(), cash=_cash("1000", "0"))
    assert empty.position_market_value == 0
    assert empty.total_equity == Decimal(1000)
    assert empty.gross_exposure == 0
    assert empty.cash_weight == 1

    with pytest.raises(ValueError, match="total_equity"):
        replace(empty, total_equity=Decimal(999))
    with pytest.raises(ValueError, match="content_hash"):
        replace(empty, content_hash="f" * 64)
    with pytest.raises(ValueError, match="total_equity must be positive"):
        AccountSnapshot.build(
            snapshot_id="zero-account",
            account_id="paper-account-1",
            runtime_mode=RuntimeMode.PAPER,
            as_of=AS_OF,
            valuation_at=VALUATION_AT,
            data_version="snapshot-v1",
            currency="CNY",
            cash=_cash("0", "0"),
            positions=(),
        )
