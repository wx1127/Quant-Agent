"""Contract tests for immutable, non-executable order drafts."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.execution.order_drafts import (
    ORDER_DRAFT_GENERATOR_VERSION,
    OrderDraftBatch,
    OrderDraftGeneratorConfig,
    OrderDraftLine,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import RiskCheckStatus

AS_OF = datetime(2026, 8, 28, 10, 0, tzinfo=SHANGHAI_TZ)
DATA_VERSION = "cn-market-20260828-v1"


def _digest(name: str) -> str:
    return stable_hash({"fixture": name})


def _buy_line() -> OrderDraftLine:
    return OrderDraftLine.build(
        instrument_id="CN.SSE.510300",
        instrument_type=TradableInstrumentType.ETF,
        side=Side.BUY,
        current_quantity=Decimal(100),
        target_quantity=Decimal(200),
        quantity=Decimal(100),
        is_full_liquidation=False,
        reference_price=Decimal("10.00"),
        estimated_execution_price=Decimal("10.01"),
        price_observed_at=AS_OF - timedelta(minutes=2),
        price_available_at=AS_OF - timedelta(minutes=1),
        data_version=DATA_VERSION,
        state_revision="state-etf-r1",
        state_hash=_digest("state-etf"),
        participation_rate=Decimal("0.02"),
        commission=Decimal("1.00"),
        stamp_duty=Decimal("0"),
        transfer_fee=Decimal("0.02"),
        other_fee=Decimal("0.03"),
        market_rule_version="cn-market-rule-v1",
        market_rule_hash=_digest("market-rule-etf"),
        fee_rule_version="cn-fee-rule-v1",
        fee_rule_hash=_digest("fee-rule-etf"),
        slippage_model_version="impact-model-v1",
        slippage_model_hash=_digest("slippage-etf"),
        rationale="increase the core ETF sleeve",
    )


def _sell_line() -> OrderDraftLine:
    # Thirteen shares intentionally exercise the A-share odd-lot full-exit path.
    return OrderDraftLine.build(
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        current_quantity=Decimal(13),
        target_quantity=Decimal(0),
        quantity=Decimal(13),
        is_full_liquidation=True,
        reference_price=Decimal("20.00"),
        estimated_execution_price=Decimal("19.98"),
        price_observed_at=AS_OF - timedelta(minutes=3),
        price_available_at=AS_OF - timedelta(minutes=1),
        data_version=DATA_VERSION,
        state_revision="state-stock-r7",
        state_hash=_digest("state-stock"),
        participation_rate=Decimal("0.01"),
        commission=Decimal("1.00"),
        stamp_duty=Decimal("0.26"),
        transfer_fee=Decimal("0.01"),
        other_fee=Decimal("0"),
        market_rule_version="cn-market-rule-v1",
        market_rule_hash=_digest("market-rule-stock"),
        fee_rule_version="cn-fee-rule-v1",
        fee_rule_hash=_digest("fee-rule-stock"),
        slippage_model_version="impact-model-v1",
        slippage_model_hash=_digest("slippage-stock"),
        rationale="fully exit the odd-lot stock holding",
    )


def _config(*, validity_seconds: int = 900) -> OrderDraftGeneratorConfig:
    return OrderDraftGeneratorConfig(
        version="order-draft-test-v1",
        validity_seconds=validity_seconds,
    )


def _batch(
    *,
    status: RiskCheckStatus = RiskCheckStatus.PASS,
    available_cash: Decimal = Decimal(2000),
    lines: tuple[OrderDraftLine, ...] | None = None,
    config: OrderDraftGeneratorConfig | None = None,
) -> OrderDraftBatch:
    return OrderDraftBatch.build(
        decision_id="portfolio-decision-20260828",
        draft_as_of=AS_OF,
        data_version=DATA_VERSION,
        currency="CNY",
        account_snapshot_id="account:paper:20260828T100000",
        account_snapshot_hash=_digest("account-snapshot"),
        account_snapshot_as_of=AS_OF - timedelta(minutes=1),
        runtime_mode=RuntimeMode.PAPER,
        portfolio_proposal_hash=_digest("target-portfolio"),
        risk_request_hash=_digest("risk-request"),
        risk_result_hash=_digest(f"risk-result-{status.value}"),
        risk_status=status,
        risk_checked_at=AS_OF - timedelta(seconds=30),
        risk_engine_version="portfolio-risk-engine-v1",
        risk_policy_version="portfolio-risk-policy-v1",
        risk_policy_hash=_digest("risk-policy"),
        config=config or _config(),
        available_cash=available_cash,
        lines=lines if lines is not None else (_buy_line(), _sell_line()),
    )


def _rebuild_line(line: OrderDraftLine, **changes: object) -> OrderDraftLine:
    payload = asdict(line)
    payload.pop("line_hash")
    payload.update(changes)
    return OrderDraftLine(**payload, line_hash=stable_hash(payload))


def test_buy_sell_lines_enforce_amount_fee_slippage_and_cash_identities() -> None:
    buy = _buy_line()
    sell = _sell_line()

    assert buy.gross_amount == Decimal("1001.00")
    assert buy.estimated_slippage_amount == Decimal("1.00")
    assert buy.total_fee == Decimal("1.05")
    assert buy.estimated_cash_change == Decimal("-1002.05")
    assert buy.estimated_cash_change == -(buy.gross_amount + buy.total_fee)

    assert sell.quantity == Decimal(13)
    assert sell.is_full_liquidation
    assert sell.gross_amount == Decimal("259.74")
    assert sell.estimated_slippage_amount == Decimal("0.26")
    assert sell.total_fee == Decimal("1.27")
    assert sell.estimated_cash_change == Decimal("258.47")
    assert sell.estimated_cash_change == sell.gross_amount - sell.total_fee


@pytest.mark.parametrize("status", [RiskCheckStatus.PASS, RiskCheckStatus.WARN])
def test_pass_and_warn_batches_aggregate_lines_and_remain_review_only(
    status: RiskCheckStatus,
) -> None:
    batch = _batch(status=status)

    assert batch.risk_status is status
    assert batch.estimated_buy_cash_required == Decimal("1002.05")
    assert batch.reserved_cash_required == Decimal("1002.05")
    assert batch.estimated_sell_cash_proceeds == Decimal("258.47")
    assert batch.estimated_total_fees == Decimal("2.32")
    assert batch.estimated_total_slippage == Decimal("1.26")
    assert batch.expires_at == AS_OF + timedelta(seconds=900)
    assert batch.batch_id == f"draft:{batch.batch_hash}"
    assert not batch.is_executable
    assert batch.requires_human_approval


def test_build_is_deterministic_and_canonicalizes_line_order_and_identity_sets() -> None:
    buy = _buy_line()
    sell = _sell_line()

    ordered = _batch(lines=(buy, sell))
    reversed_input = _batch(lines=(sell, buy))

    assert reversed_input == ordered
    assert reversed_input.input_hash == ordered.input_hash
    assert reversed_input.batch_hash == ordered.batch_hash
    assert tuple(line.instrument_id for line in ordered.lines) == (
        "CN.SSE.510300",
        "CN.SSE.600000",
    )
    assert ordered.state_hashes == tuple(sorted({buy.state_hash, sell.state_hash}))
    assert ordered.market_rule_hashes == tuple(
        sorted({buy.market_rule_hash, sell.market_rule_hash})
    )
    assert ordered.fee_rule_hashes == tuple(sorted({buy.fee_rule_hash, sell.fee_rule_hash}))
    assert ordered.slippage_model_hashes == tuple(
        sorted({buy.slippage_model_hash, sell.slippage_model_hash})
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gross_amount": Decimal("1001.01")}, "gross_amount"),
        ({"estimated_slippage_amount": Decimal("1.01")}, "slippage amount"),
        ({"total_fee": Decimal("1.06")}, "fee component sum"),
        ({"estimated_cash_change": Decimal("-1002.04")}, "cash change"),
    ],
)
def test_line_rejects_rehashed_economic_identity_tampering(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _rebuild_line(_buy_line(), **changes)


def test_line_hash_and_batch_hash_chains_reject_content_tampering() -> None:
    line = _buy_line()
    batch = _batch()

    with pytest.raises(ValueError, match="line_hash"):
        replace(line, line_hash="f" * 64)
    with pytest.raises(ValueError, match="input_hash"):
        replace(batch, input_hash="f" * 64)
    with pytest.raises(ValueError, match="batch_hash"):
        replace(batch, batch_hash="f" * 64)
    with pytest.raises(ValueError, match="batch_hash"):
        replace(batch, currency="USD")


def test_side_quantity_full_exit_and_adverse_slippage_are_consistent() -> None:
    buy = _buy_line()
    sell = _sell_line()

    with pytest.raises(ValueError, match="side and quantity"):
        _rebuild_line(buy, quantity=Decimal(99))
    with pytest.raises(ValueError, match="full-liquidation flag"):
        _rebuild_line(sell, is_full_liquidation=False)
    with pytest.raises(ValueError, match="BUY slippage"):
        _rebuild_line(buy, estimated_execution_price=Decimal("9.99"))
    with pytest.raises(ValueError, match="SELL slippage"):
        _rebuild_line(sell, estimated_execution_price=Decimal("20.01"))
    with pytest.raises(ValueError, match="positive whole number"):
        replace(buy, quantity=Decimal("100.5"))
    expensive_exit = _rebuild_line(
        sell,
        commission=Decimal(300),
        total_fee=Decimal("300.27"),
        cash_reservation_fee=Decimal("300.27"),
        reserved_cash=Decimal("40.53"),
        estimated_cash_change=Decimal("-40.53"),
    )
    assert expensive_exit.reserved_cash == Decimal("40.53")


def test_pit_times_trading_day_and_data_version_fail_closed() -> None:
    buy = _buy_line()
    batch = _batch()

    with pytest.raises(ValueError, match="timezone information"):
        replace(buy, price_observed_at=AS_OF.replace(tzinfo=None))
    with pytest.raises(ValueError, match="cannot precede"):
        replace(
            buy,
            price_observed_at=AS_OF - timedelta(seconds=1),
            price_available_at=AS_OF - timedelta(seconds=2),
        )
    future_line = _rebuild_line(buy, price_available_at=AS_OF + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="PIT price"):
        _batch(lines=(future_line, _sell_line()))
    with pytest.raises(ValueError, match="data version"):
        _batch(lines=(_rebuild_line(buy, data_version="future-v2"), _sell_line()))
    with pytest.raises(ValueError, match="risk result cannot be checked after"):
        replace(batch, risk_checked_at=AS_OF + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="trading_day"):
        replace(batch, trading_day=batch.trading_day - timedelta(days=1))


def test_batch_rejects_duplicate_unsorted_and_unbound_identity_sets() -> None:
    buy = _buy_line()
    sell = _sell_line()
    batch = _batch()

    with pytest.raises(ValueError, match="unique and sorted"):
        replace(batch, lines=(sell, buy))
    with pytest.raises(ValueError, match="unique and sorted"):
        _batch(lines=(buy, buy))
    with pytest.raises(ValueError, match="identity hashes"):
        replace(batch, state_hashes=("f" * 64,))
    with pytest.raises(ValueError, match="identity hashes"):
        replace(batch, market_rule_hashes=tuple(reversed(batch.market_rule_hashes)))
    with pytest.raises(ValueError, match="identity hashes"):
        replace(batch, fee_rule_hashes=())
    with pytest.raises(ValueError, match="identity hashes"):
        replace(batch, slippage_model_hashes=("a" * 64,))


def test_batch_root_schema_totals_and_non_negative_amounts_are_recomputed() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="schema_version"):
        replace(batch, schema_version="2")
    with pytest.raises(ValueError, match="at least one draft line"):
        _batch(lines=())
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(batch, available_cash=Decimal(-1))
    with pytest.raises(ValueError, match="estimated buy cash"):
        replace(batch, estimated_buy_cash_required=Decimal("1002.06"))
    with pytest.raises(ValueError, match="estimated sell proceeds"):
        replace(batch, estimated_sell_cash_proceeds=Decimal("258.48"))
    with pytest.raises(ValueError, match="estimated total fees"):
        replace(batch, estimated_total_fees=Decimal("2.33"))
    with pytest.raises(ValueError, match="estimated total slippage"):
        replace(batch, estimated_total_slippage=Decimal("1.27"))


@pytest.mark.parametrize("status", [RiskCheckStatus.REJECT, RiskCheckStatus.ERROR])
def test_reject_and_error_risk_results_cannot_create_drafts(status: RiskCheckStatus) -> None:
    with pytest.raises(ValueError, match="only PASS or WARN"):
        _batch(status=status)


def test_available_cash_cannot_be_supplemented_by_expected_sell_proceeds() -> None:
    assert _sell_line().estimated_cash_change > Decimal(250)

    with pytest.raises(ValueError, match="exceed snapshot available cash"):
        _batch(available_cash=Decimal(1000))


@pytest.mark.parametrize("validity_seconds", [0, -1, 86_401, True])
def test_validity_window_has_strict_bounded_integer_config(validity_seconds: int) -> None:
    with pytest.raises(ValueError, match=r"within 1\.\.86400"):
        _config(validity_seconds=validity_seconds)


def test_expiry_and_generator_configuration_are_content_bound() -> None:
    config = _config(validity_seconds=1)
    batch = _batch(config=config)

    assert batch.expires_at == AS_OF + timedelta(seconds=1)
    assert batch.validity_seconds == 1
    assert batch.generator_version == ORDER_DRAFT_GENERATOR_VERSION
    assert batch.generator_config_hash == config.config_hash
    with pytest.raises(ValueError, match="expiry"):
        replace(batch, expires_at=batch.expires_at + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="unknown order draft generator"):
        replace(batch, generator_version="order-draft-generator-v2")
    with pytest.raises(ValueError, match="generator_config_hash"):
        replace(batch, generator_config_hash="f" * 64)


@pytest.mark.parametrize("value", [1.0, Decimal("NaN"), Decimal("Infinity")])
def test_line_numeric_fields_require_finite_exact_decimal(value: object) -> None:
    with pytest.raises(ValueError, match=r"exact Decimal|finite"):
        replace(_buy_line(), reference_price=value)  # type: ignore[arg-type]


def test_batch_numeric_enum_and_digest_boundaries_are_strict() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="exact Decimal"):
        replace(batch, available_cash=2000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"RiskCheckStatus|only PASS or WARN"):
        replace(batch, risk_status="PASS")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DraftFundingPolicy"):
        replace(
            batch,
            funding_policy="SNAPSHOT_AVAILABLE_CASH_ONLY",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="PAPER or LIVE_ASSISTED"):
        replace(batch, runtime_mode=RuntimeMode.LIVE_AUTO)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(batch, account_snapshot_hash="A" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(batch, risk_request_hash="a" * 63)


def test_line_enum_decimal_hash_and_range_boundaries_are_strict() -> None:
    buy = _buy_line()

    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(buy, instrument_type="ETF")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Side"):
        replace(buy, side="BUY")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(buy, state_hash="A" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(buy, fee_rule_hash="a" * 63)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(buy, commission=Decimal("-0.01"))
    with pytest.raises(ValueError, match=r"within \(0, 1\]"):
        replace(buy, participation_rate=Decimal(0))
    with pytest.raises(ValueError, match="positive"):
        replace(buy, reference_price=Decimal(0))
    with pytest.raises(ValueError, match="must be a bool"):
        _rebuild_line(buy, is_full_liquidation=0)


def test_generator_config_rejects_lossy_enum_and_empty_version() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OrderDraftGeneratorConfig(version="  ")
    with pytest.raises(ValueError, match="DraftFundingPolicy"):
        OrderDraftGeneratorConfig(
            funding_policy="SNAPSHOT_AVAILABLE_CASH_ONLY"  # type: ignore[arg-type]
        )


def test_draft_has_no_execution_authority_or_embedded_approval_token() -> None:
    batch = _batch()
    batch_fields = {field.name for field in fields(batch)}
    line_fields = {field.name for field in fields(batch.lines[0])}
    forbidden = {
        "approval_token",
        "approved_by",
        "broker_order_id",
        "execution_token",
        "submitted_at",
    }

    assert forbidden.isdisjoint(batch_fields | line_fields)
    assert not hasattr(batch, "execute")
    assert not hasattr(batch, "submit")
    assert not batch.is_executable
    assert batch.requires_human_approval
