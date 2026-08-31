"""Fixed PIT samples for uncalibrated stock candidate scoring and filtering."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from tests.unit import test_etf_rotation as etf_fixtures
from tests.unit import test_leaders as leader_fixtures

from quant_agent.leaders import LeaderEngine
from quant_agent.leaders.candidates import (
    CandidateComponent,
    CandidateConfig,
    CandidateEngine,
    CandidateExclusionCode,
    CandidateInputError,
    CandidateRiskCode,
    CandidateSignalKind,
    CandidateSupplementalSignal,
    CandidateTier,
)
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash


def _aligned_regime_and_mainline(
    state: MarketRegime = MarketRegime.UPTREND,
):  # type: ignore[no-untyped-def]
    regime = etf_fixtures._regime(
        state,
        as_of=leader_fixtures.AS_OF,
        data_version="snapshot-v1",
    )
    base = leader_fixtures._mainline()
    identity = replace(
        base.input_identity,
        regime_transition_version=regime.transition_config_version,
        regime_transition_config_hash=regime.transition_config_hash,
        regime_transition_result_hash=regime.result_hash,
    )
    input_hash = stable_hash({"base": base.input_hash, "regime_result_hash": regime.result_hash})
    mainline = replace(
        base,
        market_regime=state,
        input_identity=identity,
        input_hash=input_hash,
        result_hash=stable_hash({"input_hash": input_hash, "stage": "candidate-mainline"}),
    )
    return regime, mainline


def _scenario(
    state: MarketRegime = MarketRegime.UPTREND,
):  # type: ignore[no-untyped-def]
    regime, mainline = _aligned_regime_and_mainline(state)
    stocks = [
        leader_fixtures._stock(
            "CORE",
            short_return="0.30",
            within_theme="95",
            benchmark_resilience="90",
            trend="95",
            volume_price="90",
        ),
        leader_fixtures._stock(
            "SECOND",
            short_return="0.22",
            within_theme="80",
            benchmark_resilience="75",
            trend="80",
            volume_price="75",
        ),
        leader_fixtures._stock(
            "WATCH",
            short_return="0.12",
            within_theme="65",
            benchmark_resilience="60",
            trend="65",
            volume_price="55",
        ),
        leader_fixtures._stock(
            "BLOCKED",
            short_return="0.18",
            within_theme="75",
            benchmark_resilience="70",
            trend="75",
            volume_price="70",
        ),
        leader_fixtures._stock(
            "OUTSIDE",
            industry_id=leader_fixtures.INDUSTRY_B,
        ),
    ]
    trades = [
        leader_fixtures._tradeability("CORE", liquidity="100"),
        leader_fixtures._tradeability("SECOND", liquidity="85"),
        leader_fixtures._tradeability("WATCH", liquidity="75"),
        leader_fixtures._tradeability("BLOCKED", liquidity="90", eligible=False),
        leader_fixtures._tradeability("OUTSIDE", liquidity="80"),
    ]
    fundamentals = [
        leader_fixtures._fundamental("CORE"),
        leader_fixtures._fundamental("SECOND"),
        leader_fixtures._fundamental("WATCH"),
        leader_fixtures._fundamental("BLOCKED"),
        leader_fixtures._fundamental("OUTSIDE"),
    ]
    leaders = LeaderEngine().rank(
        mainline=mainline,
        stock_strength=stocks,
        tradeability=trades,
        fundamentals=fundamentals,
    )
    ranked_ids = {item.instrument_id for item in leaders.leaders}
    return (
        regime,
        mainline,
        leaders,
        [item for item in stocks if item.instrument_id in ranked_ids],
        [item for item in trades if item.request.instrument_id in ranked_ids],
        [item for item in fundamentals if item.instrument_id in ranked_ids],
    )


def _signal(
    instrument_id: str,
    kind: CandidateSignalKind,
    *,
    score: str,
    risk: str = "0",
    as_of=leader_fixtures.AS_OF,  # type: ignore[no-untyped-def]
) -> CandidateSupplementalSignal:
    return CandidateSupplementalSignal.build(
        instrument_id=instrument_id,
        kind=kind,
        session_date=leader_fixtures.SESSION,
        as_of=as_of,
        data_version="snapshot-v1",
        model_version=f"{kind.value.lower()}-v1",
        score=Decimal(score),
        risk_penalty=Decimal(risk),
        source_refs=(f"{kind.value.lower()}:{instrument_id}",),
    )


def _rank(
    *,
    state: MarketRegime = MarketRegime.UPTREND,
    signals: tuple[CandidateSupplementalSignal, ...] = (),
    config: CandidateConfig | None = None,
):  # type: ignore[no-untyped-def]
    regime, mainline, leaders, stocks, trades, fundamentals = _scenario(state)
    return CandidateEngine(config).rank(
        regime=regime,
        mainline=mainline,
        leaders=leaders,
        stock_strength=stocks,
        tradeability=trades,
        fundamentals=fundamentals,
        supplemental_signals=signals,
    )


def test_candidate_pool_is_ranked_filtered_traceable_and_explicitly_uncalibrated() -> None:
    signals = (
        _signal("CORE", CandidateSignalKind.EVENT, score="95"),
        _signal("CORE", CandidateSignalKind.VALUATION, score="80"),
        _signal("SECOND", CandidateSignalKind.VALUATION, score="50"),
    )
    first = _rank(signals=signals)
    repeated = _rank(signals=tuple(reversed(signals)))

    assert first == repeated
    assert not first.calibrated
    assert first.calibration_version is None
    assert first.candidates[0].instrument_id == "CORE"
    assert tuple(item.research_rank for item in first.candidates) == tuple(
        range(1, len(first.candidates) + 1)
    )
    assert all(
        tuple(item.component for item in candidate.components) == tuple(CandidateComponent)
        for candidate in first.candidates
    )
    blocked = next(item for item in first.candidates if item.instrument_id == "BLOCKED")
    assert blocked.tier is CandidateTier.EXCLUDED
    assert blocked.eligible_rank is None
    assert CandidateExclusionCode.LEADER_FILTER in blocked.exclusion_codes
    assert any("SUSPENDED" in reason for reason in blocked.exclusion_reasons)
    assert first.exclusions[0].instrument_id == "OUTSIDE"
    assert first.exclusions[0].code is CandidateExclusionCode.UPSTREAM_LEADER_EXCLUSION
    assert all(item.observation_conditions and item.invalidations for item in first.candidates)
    assert all(
        len(value) == 64 for value in (first.config_hash, first.input_hash, first.result_hash)
    )
    rendered = repr(first).lower()
    assert all(term not in rendered for term in ("guaranteed", "certain rise", "稳赚", "必买"))


def test_tiers_are_score_layers_and_missing_valuation_is_not_neutral() -> None:
    result = _rank(
        signals=(
            _signal("CORE", CandidateSignalKind.VALUATION, score="100"),
            _signal("SECOND", CandidateSignalKind.VALUATION, score="40"),
        ),
        config=CandidateConfig(tier_a_minimum=Decimal("90"), tier_b_minimum=Decimal("80")),
    )
    by_id = {item.instrument_id: item for item in result.candidates}

    assert by_id["CORE"].tier is CandidateTier.A
    assert by_id["SECOND"].tier is CandidateTier.B
    assert by_id["WATCH"].tier is CandidateTier.WATCH
    valuation = next(
        item for item in by_id["WATCH"].components if item.component is CandidateComponent.VALUATION
    )
    assert valuation.score == 0
    assert valuation.source == "MISSING_PIT_VALUATION"
    assert any(
        item.feature == CandidateComponent.VALUATION.value
        for item in by_id["WATCH"].counter_evidence
    )


def test_event_support_cannot_override_hard_tradeability_or_downtrend_filters() -> None:
    supported = _rank(signals=(_signal("BLOCKED", CandidateSignalKind.EVENT, score="100"),))
    blocked = next(item for item in supported.candidates if item.instrument_id == "BLOCKED")
    assert blocked.tier is CandidateTier.EXCLUDED

    risky = _rank(signals=(_signal("CORE", CandidateSignalKind.EVENT, score="100", risk="10"),))
    core = next(item for item in risky.candidates if item.instrument_id == "CORE")
    assert CandidateRiskCode.EVENT in {item.code for item in core.risks}
    assert any(item.feature == "RISK:EVENT" for item in core.counter_evidence)

    downtrend = _rank(
        state=MarketRegime.DOWNTREND,
        signals=(_signal("CORE", CandidateSignalKind.EVENT, score="100"),),
    )
    assert downtrend.regime is MarketRegime.DOWNTREND
    assert downtrend.candidates
    assert all(item.tier is CandidateTier.EXCLUDED for item in downtrend.candidates)
    assert all(
        CandidateExclusionCode.MARKET_DOWNTREND in item.exclusion_codes
        for item in downtrend.candidates
    )


def test_future_duplicate_and_unranked_supplemental_signals_fail_closed() -> None:
    future = _signal(
        "CORE",
        CandidateSignalKind.EVENT,
        score="90",
        as_of=leader_fixtures.AS_OF + timedelta(seconds=1),
    )
    with pytest.raises(CandidateInputError, match="future"):
        _rank(signals=(future,))

    signal = _signal("CORE", CandidateSignalKind.EVENT, score="90")
    with pytest.raises(CandidateInputError, match="must be unique"):
        _rank(signals=(signal, signal))
    with pytest.raises(CandidateInputError, match="unranked"):
        _rank(signals=(_signal("UNKNOWN", CandidateSignalKind.EVENT, score="90"),))


def test_upstream_identity_and_exact_feature_coverage_are_enforced() -> None:
    regime, mainline, leaders, stocks, trades, fundamentals = _scenario()
    changed_stock = replace(stocks[0], result_hash="f" * 64)
    with pytest.raises(CandidateInputError, match="upstream identities"):
        CandidateEngine().rank(
            regime=regime,
            mainline=mainline,
            leaders=leaders,
            stock_strength=(changed_stock, *stocks[1:]),
            tradeability=trades,
            fundamentals=fundamentals,
        )
    wrong_industry = replace(stocks[0], current_industry_id="SW2021:OTHER")
    with pytest.raises(CandidateInputError, match="industry membership"):
        CandidateEngine().rank(
            regime=regime,
            mainline=mainline,
            leaders=leaders,
            stock_strength=(wrong_industry, *stocks[1:]),
            tradeability=trades,
            fundamentals=fundamentals,
        )
    with pytest.raises(CandidateInputError, match="cover exactly"):
        CandidateEngine().rank(
            regime=regime,
            mainline=mainline,
            leaders=leaders,
            stock_strength=stocks[:-1],
            tradeability=trades,
            fundamentals=fundamentals,
        )

    mismatched_identity = replace(
        mainline.input_identity,
        regime_transition_result_hash="f" * 64,
    )
    with pytest.raises(CandidateInputError, match="does not bind"):
        CandidateEngine().rank(
            regime=regime,
            mainline=replace(mainline, input_identity=mismatched_identity),
            leaders=leaders,
            stock_strength=stocks,
            tradeability=trades,
            fundamentals=fundamentals,
        )


def test_config_and_signal_contracts_are_versioned_and_hash_bound() -> None:
    baseline = CandidateConfig()
    assert len(baseline.config_hash) == 64
    assert replace(baseline, tier_a_minimum=Decimal("76")).config_hash != baseline.config_hash
    with pytest.raises(ValueError, match="sum to one"):
        CandidateConfig(market_regime_weight=Decimal("0.21"))
    with pytest.raises(ValueError, match="ordered"):
        CandidateConfig(tier_a_minimum=Decimal("60"), tier_b_minimum=Decimal("60"))

    signal = _signal("CORE", CandidateSignalKind.EVENT, score="90")
    assert signal == _signal("CORE", CandidateSignalKind.EVENT, score="90")
    assert signal.identity_payload()["result_hash"] == signal.result_hash
    with pytest.raises(ValueError, match="result_hash"):
        replace(signal, score=Decimal(91))
    with pytest.raises(ValueError, match="input_hash"):
        replace(signal, source_refs=("different-source",))
    with pytest.raises(ValueError, match="source references"):
        CandidateSupplementalSignal.build(
            instrument_id="CORE",
            kind=CandidateSignalKind.EVENT,
            session_date=leader_fixtures.SESSION,
            as_of=leader_fixtures.AS_OF,
            data_version="snapshot-v1",
            model_version="event-v1",
            score=Decimal(90),
            source_refs=(),
        )
