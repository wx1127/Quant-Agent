"""Frozen entry, exit, weighting, and turnover rules for stock targets."""

from dataclasses import replace
from decimal import Decimal

import pytest
from tests.unit import test_leaders as leader_fixtures
from tests.unit.test_candidates import _rank, _scenario

from quant_agent.features.mainline import MainlineState
from quant_agent.leaders import LeaderType
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.mainline_leader import (
    MainlineLeaderAction,
    MainlineLeaderConfig,
    MainlineLeaderDecisionStatus,
    MainlineLeaderInputError,
    MainlineLeaderPortfolioPosition,
    MainlineLeaderPortfolioState,
    MainlineLeaderReasonCode,
    MainlineLeaderRequest,
    MainlineLeaderStrategy,
    MainlineLeaderWeightingMode,
)


def _request(session_index: int = 0) -> MainlineLeaderRequest:
    return MainlineLeaderRequest(
        signal_date=leader_fixtures.SESSION,
        as_of=leader_fixtures.AS_OF,
        data_version="snapshot-v1",
        session_index=session_index,
    )


def _portfolio(
    weights: dict[str, str] | None = None,
) -> MainlineLeaderPortfolioState:
    values = weights or {}
    positions = tuple(
        MainlineLeaderPortfolioPosition(
            instrument_id=instrument_id,
            industry_id=leader_fixtures.INDUSTRY_A,
            target_weight=Decimal(weight),
        )
        for instrument_id, weight in sorted(values.items())
    )
    return MainlineLeaderPortfolioState(
        as_of=leader_fixtures.AS_OF,
        data_version="snapshot-v1",
        positions=positions,
        source_decision_hash=stable_hash({"weights": values}) if positions else None,
    )


def _inputs(
    state: MarketRegime = MarketRegime.UPTREND,
):  # type: ignore[no-untyped-def]
    regime, mainline, leaders, *_rest = _scenario(state)
    return regime, mainline, leaders, _rank(state=state)


def _baseline_config(**changes: object) -> MainlineLeaderConfig:
    values: dict[str, object] = {
        "maximum_positions": 2,
        "maximum_instrument_weight": Decimal("0.50"),
        "maximum_discretionary_turnover": Decimal("2"),
        "rebalance_frequency_sessions": 1,
    }
    values.update(changes)
    return MainlineLeaderConfig(**values)  # type: ignore[arg-type]


def _decide(
    *,
    config: MainlineLeaderConfig | None = None,
    request: MainlineLeaderRequest | None = None,
    portfolio: MainlineLeaderPortfolioState | None = None,
    state: MarketRegime = MarketRegime.UPTREND,
):  # type: ignore[no-untyped-def]
    regime, mainline, leaders, candidates = _inputs(state)
    return MainlineLeaderStrategy(config or _baseline_config()).decide(
        request=request or _request(),
        regime=regime,
        mainline=mainline,
        leaders=leaders,
        candidates=candidates,
        portfolio=portfolio or _portfolio(),
    )


def test_equal_weight_entry_emits_complete_targets_without_orders() -> None:
    first = _decide()
    repeated = _decide()

    assert first == repeated
    assert first.status is MainlineLeaderDecisionStatus.REBALANCE
    assert first.exposure_cap == Decimal("0.60")
    assert first.gross_target_weight == Decimal("0.60")
    assert first.cash_target_weight == Decimal("0.40")
    targets = {item.instrument_id: item.target_weight for item in first.targets}
    assert targets == {
        "BLOCKED": Decimal(0),
        "CORE": Decimal("0.30"),
        "SECOND": Decimal("0.30"),
        "WATCH": Decimal(0),
    }
    selections = {item.instrument_id: item for item in first.selections}
    assert selections["CORE"].action is MainlineLeaderAction.ENTER
    assert selections["SECOND"].action is MainlineLeaderAction.ENTER
    assert MainlineLeaderReasonCode.CANDIDATE_INELIGIBLE in selections["BLOCKED"].reason_codes
    assert MainlineLeaderReasonCode.OUTSIDE_TOP_N in selections["WATCH"].reason_codes
    assert all(
        item.leader_type in {LeaderType.TREND, LeaderType.CAPACITY}
        for item in first.selections
        if item.desired
    )
    assert not hasattr(first, "orders")
    assert len(first.input_hash) == len(first.result_hash) == 64


def test_risk_adjusted_weighting_uses_frozen_candidate_score_and_penalty() -> None:
    config = _baseline_config(
        weighting_mode=MainlineLeaderWeightingMode.RISK_ADJUSTED,
        maximum_instrument_weight=Decimal(1),
    )
    decision = _decide(config=config)
    targets = {item.instrument_id: item.target_weight for item in decision.targets}

    assert targets["CORE"] > targets["SECOND"] > 0
    assert targets["CORE"] + targets["SECOND"] == Decimal("0.60")
    assert decision.config_hash == config.config_hash


def test_turnover_cap_interpolates_targets_and_records_constraint() -> None:
    decision = _decide(config=_baseline_config(maximum_discretionary_turnover=Decimal("0.20")))
    selections = {item.instrument_id: item for item in decision.selections}

    assert decision.discretionary_turnover == Decimal("0.20")
    assert decision.forced_exit_turnover == 0
    assert decision.gross_target_weight == Decimal("0.20")
    assert MainlineLeaderReasonCode.TURNOVER_CAP in selections["CORE"].reason_codes
    assert selections["CORE"].target_weight < selections["CORE"].unconstrained_weight


def test_rebalance_schedule_holds_valid_positions_and_blocks_new_entries() -> None:
    config = _baseline_config(rebalance_frequency_sessions=5)
    held = _decide(
        config=config,
        request=_request(session_index=1),
        portfolio=_portfolio({"CORE": "0.30", "SECOND": "0.30"}),
    )
    blocked_entry = _decide(
        config=config,
        request=_request(session_index=1),
        portfolio=_portfolio(),
    )

    assert held.status is MainlineLeaderDecisionStatus.HOLD
    assert held.total_turnover == 0
    assert {item.instrument_id: item.target_weight for item in held.targets}["CORE"] == Decimal(
        "0.30"
    )
    assert blocked_entry.status is MainlineLeaderDecisionStatus.CASH
    assert all(item.target_weight == 0 for item in blocked_entry.targets)
    assert any(
        MainlineLeaderReasonCode.REBALANCE_SCHEDULE in item.reason_codes
        for item in blocked_entry.selections
    )


def test_invalid_stock_exit_bypasses_schedule_and_discretionary_turnover_cap() -> None:
    decision = _decide(
        config=_baseline_config(
            rebalance_frequency_sessions=5,
            maximum_discretionary_turnover=Decimal(0),
        ),
        request=_request(session_index=1),
        portfolio=_portfolio({"BLOCKED": "0.20"}),
    )
    blocked = next(item for item in decision.selections if item.instrument_id == "BLOCKED")

    assert blocked.action is MainlineLeaderAction.EXIT
    assert blocked.target_weight == 0
    assert decision.forced_exit_turnover == Decimal("0.20")
    assert decision.discretionary_turnover == 0
    assert MainlineLeaderReasonCode.CANDIDATE_INELIGIBLE in blocked.reason_codes


def test_fading_mainline_forces_prior_position_exit() -> None:
    regime, mainline, leaders, candidates = _inputs()
    industries = tuple(
        replace(
            item,
            state=MainlineState.FADING,
            previous_state=MainlineState.CONFIRMED,
            transition_reason="confirmed mainline failed persistence and is fading",
        )
        if item.industry_id == leader_fixtures.INDUSTRY_A
        else item
        for item in mainline.industries
    )
    mainline_hash = stable_hash(
        {"base_result_hash": mainline.result_hash, "state": MainlineState.FADING}
    )
    faded_mainline = replace(mainline, industries=industries, result_hash=mainline_hash)
    empty_leaders = replace(
        leaders,
        mainline_result_hash=mainline_hash,
        leaders=(),
        exclusions=(),
        input_hash=stable_hash({"mainline_result_hash": mainline_hash, "stage": "leaders"}),
        result_hash=stable_hash({"mainline_result_hash": mainline_hash, "stage": "leaders-result"}),
    )
    empty_candidates = replace(
        candidates,
        candidates=(),
        exclusions=(),
        input_hash=stable_hash({"leader_result_hash": empty_leaders.result_hash}),
        result_hash=stable_hash(
            {"leader_result_hash": empty_leaders.result_hash, "stage": "candidate-result"}
        ),
    )

    decision = MainlineLeaderStrategy(_baseline_config(rebalance_frequency_sessions=5)).decide(
        request=_request(session_index=1),
        regime=regime,
        mainline=faded_mainline,
        leaders=empty_leaders,
        candidates=empty_candidates,
        portfolio=_portfolio({"CORE": "0.30"}),
    )
    core = decision.selections[0]

    assert decision.status is MainlineLeaderDecisionStatus.CASH
    assert decision.forced_exit_turnover == Decimal("0.30")
    assert core.action is MainlineLeaderAction.EXIT
    assert MainlineLeaderReasonCode.MAINLINE_FADING in core.reason_codes


def test_downtrend_forces_cash_even_when_prior_target_exists() -> None:
    decision = _decide(
        state=MarketRegime.DOWNTREND,
        portfolio=_portfolio({"CORE": "0.30"}),
    )

    assert decision.market_regime is MarketRegime.DOWNTREND
    assert decision.status is MainlineLeaderDecisionStatus.CASH
    assert decision.gross_target_weight == 0
    assert decision.forced_exit_turnover == Decimal("0.30")
    assert all(item.target_weight == 0 for item in decision.targets)


def test_alignment_and_future_portfolio_inputs_fail_closed() -> None:
    regime, mainline, leaders, candidates = _inputs()
    strategy = MainlineLeaderStrategy(_baseline_config())
    kwargs = {
        "request": _request(),
        "regime": regime,
        "mainline": mainline,
        "leaders": leaders,
        "candidates": candidates,
        "portfolio": _portfolio(),
    }
    with pytest.raises(MainlineLeaderInputError, match="bind the current mainline"):
        strategy.decide(**{**kwargs, "leaders": replace(leaders, mainline_result_hash="f" * 64)})
    candidate = candidates.candidates[0]
    changed_identity = replace(candidate.upstream_identity, leader_result_hash="f" * 64)
    changed_candidate = replace(candidate, upstream_identity=changed_identity)
    changed_candidates = replace(
        candidates,
        candidates=(changed_candidate, *candidates.candidates[1:]),
    )
    with pytest.raises(MainlineLeaderInputError, match="upstream identities"):
        strategy.decide(**{**kwargs, "candidates": changed_candidates})
    future = replace(
        _portfolio(),
        as_of=leader_fixtures.AS_OF.replace(year=leader_fixtures.AS_OF.year + 1),
    )
    with pytest.raises(MainlineLeaderInputError, match="future prior portfolio"):
        strategy.decide(**{**kwargs, "portfolio": future})
    oversized = _portfolio({"CORE": "0.60"})
    with pytest.raises(MainlineLeaderInputError, match="per-instrument limit"):
        strategy.decide(**{**kwargs, "portfolio": oversized})


def test_config_and_decision_contracts_reject_parameter_or_hash_tampering() -> None:
    with pytest.raises(ValueError, match="TREND and CAPACITY"):
        MainlineLeaderConfig(eligible_leader_types=(LeaderType.ELASTICITY,))
    with pytest.raises(ValueError, match=r"within 0\.\.2"):
        MainlineLeaderConfig(maximum_discretionary_turnover=Decimal("2.1"))
    with pytest.raises(ValueError, match="canonical order"):
        MainlineLeaderConfig(eligible_tiers=())

    decision = _decide()
    with pytest.raises(ValueError, match="result hash does not match"):
        replace(decision, result_hash="f" * 64)
    with pytest.raises(ValueError, match="input hash does not match"):
        replace(decision, portfolio_input_hash="e" * 64)
