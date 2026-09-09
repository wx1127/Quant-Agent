"""Research-pipeline tests over coherent point-in-time engine inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import pytest
from tests.unit import test_candidates as candidate_fixtures
from tests.unit import test_leaders as leader_fixtures
from tests.unit import test_mainline as mainline_fixtures
from tests.unit import test_quality as quality_fixtures
from tests.unit import test_regime_classifier as regime_fixtures

from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools.research.inputs import (
    AccountBoundLeaderInputs,
    DataBoundQualityContext,
    RegimeFeaturePair,
    ResearchInputInvalid,
    ResearchInputUnavailable,
)
from quant_agent.agent.tools.research.service import ResearchPipeline
from quant_agent.config import RuntimeMode
from quant_agent.data.quality import QualityContext, QualityEngine, QualityReport
from quant_agent.data.snapshots import SnapshotFile, SnapshotManifest
from quant_agent.features.industry_strength import IndustryStrengthSnapshot
from quant_agent.features.mainline import MainlineConfig, MainlineState, apply_mainline_states
from quant_agent.leaders import LeaderEngine
from quant_agent.leaders.candidates import (
    CandidateEngine,
    CandidateSignalKind,
    CandidateSupplementalSignal,
)
from quant_agent.regime import MarketRegime, MarketRegimeClassifier, apply_regime_transitions
from quant_agent.regime.contracts import stable_hash

DATA_VERSION = "snapshot-v1"
DATA_CONTENT_HASH = "3" * 64
ACCOUNT_ID = "research-account-1"
ACCOUNT_SNAPSHOT_ID = "research-account-1:20260828T160000"
ACCOUNT_SNAPSHOT_HASH = "4" * 64


@dataclass
class _ResearchSource:
    manifest: SnapshotManifest
    quality: DataBoundQualityContext
    regimes: tuple[RegimeFeaturePair, ...]
    industries: tuple[IndustryStrengthSnapshot, ...]
    leader_inputs: AccountBoundLeaderInputs
    signals: tuple[CandidateSupplementalSignal, ...]
    unavailable_method: str | None = None

    def _available(self, method: str) -> None:
        if self.unavailable_method == method:
            raise ResearchInputUnavailable(f"{method} is unavailable")

    def load_verified_manifest(self, decision: DecisionSnapshot) -> SnapshotManifest:
        self._available("manifest")
        return self.manifest

    def load_quality_context(self, decision: DecisionSnapshot) -> DataBoundQualityContext:
        self._available("quality")
        return self.quality

    def load_regime_history(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[RegimeFeaturePair, ...]:
        self._available("regimes")
        return self.regimes

    def load_industry_history(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[IndustryStrengthSnapshot, ...]:
        self._available("industries")
        return self.industries

    def load_leader_inputs(self, decision: DecisionSnapshot) -> AccountBoundLeaderInputs:
        self._available("leaders")
        return self.leader_inputs

    def load_candidate_signals(
        self,
        decision: DecisionSnapshot,
    ) -> tuple[CandidateSupplementalSignal, ...]:
        self._available("signals")
        return self.signals


def _decision() -> DecisionSnapshot:
    reference = StrategySnapshotRef(
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        config_hash="1" * 64,
        parameter_version="research-parameters-v1",
        parameter_hash="2" * 64,
        registered_at=leader_fixtures.AS_OF - timedelta(days=30),
    )
    return DecisionSnapshot.build(
        decision_id="dec_20260828_0123456789abcdef0123456789abcdef",
        mode=RuntimeMode.RESEARCH,
        market="CN_A",
        as_of=leader_fixtures.AS_OF,
        data_version=DATA_VERSION,
        data_content_hash=DATA_CONTENT_HASH,
        strategy_refs=(reference,),
        risk_policy_version="research-risk-v1",
        risk_policy_hash="5" * 64,
        account_id=ACCOUNT_ID,
        account_snapshot_id=ACCOUNT_SNAPSHOT_ID,
        account_snapshot_hash=ACCOUNT_SNAPSHOT_HASH,
        code_commit="6" * 40,
        code_artifact_hash="7" * 64,
        agent_version="quant-agent-harness-v1",
        model_version="research-model-v1",
    )


def _regime_history() -> tuple[RegimeFeaturePair, ...]:
    pairs: list[RegimeFeaturePair] = []
    for index, days_before in enumerate((2, 1, 0)):
        as_of = leader_fixtures.AS_OF - timedelta(days=days_before)
        session_date = as_of.date()
        trend = replace(
            regime_fixtures._trend("70"),
            session_date=session_date,
            as_of=as_of,
            cache_key=stable_hash({"kind": "trend", "index": index}),
        )
        breadth = replace(
            regime_fixtures._breadth(
                advance_decline="0.70",
                above_average="0.75",
                new_high="0.20",
                new_low="0.01",
                turnover="0.80",
                downside_volatility="0.005",
            ),
            session_date=session_date,
            as_of=as_of,
            cache_key=stable_hash({"kind": "breadth", "index": index}),
        )
        pairs.append(RegimeFeaturePair(trend=trend, breadth=breadth))
    return tuple(pairs)


def _industry_history() -> tuple[IndustryStrengthSnapshot, ...]:
    results: list[IndustryStrengthSnapshot] = []
    for days_before, rank in zip((2, 1, 0), (1, 2, 5), strict=True):
        as_of = leader_fixtures.AS_OF - timedelta(days=days_before)
        snapshot = mainline_fixtures._industry_snapshot(2 - days_before, rank)
        results.append(replace(snapshot, session_date=as_of.date(), as_of=as_of))
    return tuple(results)


def _features() -> AccountBoundLeaderInputs:
    _, _, _, stocks, trades, fundamentals = candidate_fixtures._scenario()
    rebound_fundamentals = tuple(replace(item, data_version=DATA_VERSION) for item in fundamentals)
    return AccountBoundLeaderInputs(
        account_id=ACCOUNT_ID,
        account_snapshot_id=ACCOUNT_SNAPSHOT_ID,
        account_snapshot_hash=ACCOUNT_SNAPSHOT_HASH,
        stock_strength=tuple(stocks),
        tradeability=tuple(trades),
        fundamentals=rebound_fundamentals,
    )


def _source(
    *,
    signals: tuple[CandidateSupplementalSignal, ...] = (),
) -> _ResearchSource:
    decision = _decision()
    manifest = SnapshotManifest(
        data_version=DATA_VERSION,
        created_at=decision.as_of.isoformat(),
        content_hash=DATA_CONTENT_HASH,
        files=(
            SnapshotFile(
                name="bars",
                relative_path="bars.parquet",
                row_count=1,
                sha256="8" * 64,
            ),
        ),
        metadata={"market": "CN_A"},
    )
    quality_context = QualityContext(
        bars=(quality_fixtures._bar(trade_date=leader_fixtures.SESSION),),
        expected_instrument_ids=("CN.SZ.000001",),
        expected_open_dates=(leader_fixtures.SESSION,),
    )
    return _ResearchSource(
        manifest=manifest,
        quality=DataBoundQualityContext(
            data_version=DATA_VERSION,
            data_content_hash=DATA_CONTENT_HASH,
            context=quality_context,
        ),
        regimes=_regime_history(),
        industries=_industry_history(),
        leader_inputs=_features(),
        signals=signals,
    )


def _expected_outputs(source: _ResearchSource) -> tuple[Any, Any, Any, Any]:
    classified = tuple(
        MarketRegimeClassifier().classify(
            trend_snapshot=pair.trend,
            breadth_snapshot=pair.breadth,
        )
        for pair in source.regimes
    )
    transitions = apply_regime_transitions(classified)
    mainlines = apply_mainline_states(
        zip(source.industries, transitions, strict=True),
        MainlineConfig(),
    )
    leaders = LeaderEngine().rank(
        mainline=mainlines[-1],
        stock_strength=source.leader_inputs.stock_strength,
        tradeability=source.leader_inputs.tradeability,
        fundamentals=source.leader_inputs.fundamentals,
    )
    ranked_ids = frozenset(item.instrument_id for item in leaders.leaders)
    candidates = CandidateEngine().rank(
        regime=transitions[-1],
        mainline=mainlines[-1],
        leaders=leaders,
        stock_strength=tuple(
            item for item in source.leader_inputs.stock_strength if item.instrument_id in ranked_ids
        ),
        tradeability=tuple(
            item
            for item in source.leader_inputs.tradeability
            if item.request.instrument_id in ranked_ids
        ),
        fundamentals=tuple(
            item for item in source.leader_inputs.fundamentals if item.instrument_id in ranked_ids
        ),
        supplemental_signals=source.signals,
    )
    return transitions[-1], mainlines[-1], leaders, candidates


def test_pipeline_matches_actual_engines_and_is_deterministic() -> None:
    decision = _decision()
    source = _source(
        signals=(
            candidate_fixtures._signal(
                "CORE",
                CandidateSignalKind.EVENT,
                score="90",
            ),
        )
    )
    pipeline = ResearchPipeline(source)
    expected_regime, expected_mainline, expected_leaders, expected_candidates = _expected_outputs(
        source
    )

    assert pipeline.detect_market_regime(decision) == expected_regime
    assert pipeline.rank_market_themes(decision) == expected_mainline
    assert pipeline.rank_theme_leaders(decision) == expected_leaders
    assert pipeline.rank_stock_candidates(decision) == expected_candidates
    assert pipeline.detect_market_regime(decision) == expected_regime
    assert pipeline.rank_stock_candidates(decision) == expected_candidates
    assert expected_regime.final_regime is MarketRegime.UPTREND
    assert expected_candidates.candidates
    assert expected_candidates.calibrated is False
    assert expected_candidates.as_of == decision.as_of
    assert expected_candidates.data_version == decision.data_version


def test_full_history_confirms_mainline_that_one_session_cannot_confirm() -> None:
    decision = _decision()
    source = _source()
    pipeline = ResearchPipeline(source)

    final = pipeline.rank_market_themes(decision)
    industry = next(
        item for item in final.industries if item.industry_id == mainline_fixtures.INDUSTRIES[0]
    )
    one_session_source = _source()
    one_session_source.regimes = source.regimes[-1:]
    one_session_source.industries = source.industries[-1:]
    one_session = ResearchPipeline(one_session_source).rank_market_themes(decision)
    one_session_industry = next(
        item
        for item in one_session.industries
        if item.industry_id == mainline_fixtures.INDUSTRIES[0]
    )

    assert industry.state is MainlineState.CONFIRMED
    assert industry.persistence.recent_ranks == (1, 2, 5)
    assert one_session_industry.state is MainlineState.EMERGING


@pytest.mark.parametrize(
    ("field", "value"),
    (("data_version", "other-v1"), ("content_hash", "f" * 64)),
)
def test_manifest_identity_mismatch_fails_closed(field: str, value: str) -> None:
    source = _source()
    source.manifest = source.manifest.model_copy(update={field: value})

    with pytest.raises(ResearchInputInvalid, match="manifest does not match"):
        ResearchPipeline(source).load_manifest(_decision())


@pytest.mark.parametrize("failure", ["future", "misaligned"])
def test_future_or_misaligned_regime_history_fails_closed(failure: str) -> None:
    source = _source()
    final = source.regimes[-1]
    if failure == "future":
        shifted = leader_fixtures.AS_OF + timedelta(seconds=1)
        changed = replace(
            final,
            trend=replace(final.trend, as_of=shifted),
            breadth=replace(final.breadth, as_of=shifted),
        )
        message = "future observation"
    else:
        changed = replace(
            final,
            breadth=replace(final.breadth, as_of=final.breadth.as_of - timedelta(seconds=1)),
        )
        message = "not aligned"
    source.regimes = (*source.regimes[:-1], changed)

    with pytest.raises(ResearchInputInvalid, match=message):
        ResearchPipeline(source).detect_market_regime(_decision())


class _CapturingQualityEngine:
    def __init__(self) -> None:
        self.session: object = "not-called"
        self.observed_at: datetime | None = None

    def run(
        self,
        data_version: str,
        context: QualityContext,
        *,
        session: object = None,
        observed_at: datetime | None = None,
    ) -> QualityReport:
        self.session = session
        self.observed_at = observed_at
        return QualityEngine().run(
            data_version,
            context,
            session=None,
            observed_at=observed_at,
        )


def test_quality_is_fixed_to_decision_time_and_never_receives_a_persistence_session() -> None:
    decision = _decision()
    pipeline = ResearchPipeline(_source())
    spy = _CapturingQualityEngine()
    pipeline._quality_engine = spy  # type: ignore[assignment]

    report = pipeline.validate_market_data(decision)

    assert report.qualified
    assert report.observed_at == decision.as_of
    assert spy.observed_at == decision.as_of
    assert spy.session is None


def test_account_snapshot_hash_mismatch_fails_closed() -> None:
    source = _source()
    source.leader_inputs = replace(
        source.leader_inputs,
        account_snapshot_hash="f" * 64,
    )

    with pytest.raises(ResearchInputInvalid, match="decision account"):
        ResearchPipeline(source).rank_theme_leaders(_decision())


def test_fundamental_data_version_mismatch_fails_closed_before_ranking() -> None:
    source = _source()
    source.leader_inputs = replace(
        source.leader_inputs,
        fundamentals=(
            replace(source.leader_inputs.fundamentals[0], data_version="future-v1"),
            *source.leader_inputs.fundamentals[1:],
        ),
    )

    with pytest.raises(ResearchInputInvalid, match="fundamental input is not decision-bound"):
        ResearchPipeline(source).rank_theme_leaders(_decision())


def test_future_supplemental_signal_fails_closed() -> None:
    future = candidate_fixtures._signal(
        "CORE",
        CandidateSignalKind.EVENT,
        score="90",
        as_of=leader_fixtures.AS_OF + timedelta(seconds=1),
    )

    with pytest.raises(ResearchInputInvalid, match="candidate signal is not point-in-time"):
        ResearchPipeline(_source(signals=(future,))).rank_stock_candidates(_decision())


@pytest.mark.parametrize("empty_input", ["regimes", "leaders"])
def test_empty_required_inputs_fail_closed(empty_input: str) -> None:
    source = _source()
    if empty_input == "regimes":
        source.regimes = ()
        message = "regime history cannot be empty"
    else:
        source.leader_inputs = replace(source.leader_inputs, stock_strength=())
        message = "stock strength inputs have an invalid size"

    with pytest.raises(ResearchInputInvalid, match=message):
        if empty_input == "regimes":
            ResearchPipeline(source).detect_market_regime(_decision())
        else:
            ResearchPipeline(source).rank_theme_leaders(_decision())


def test_unavailable_authoritative_input_is_not_converted_to_an_empty_result() -> None:
    source = _source()
    source.unavailable_method = "manifest"

    with pytest.raises(ResearchInputUnavailable, match="manifest is unavailable"):
        ResearchPipeline(source).detect_market_regime(_decision())


def test_pipeline_rejects_incomplete_source_and_non_decision_objects() -> None:
    with pytest.raises(TypeError, match="complete read-only protocol"):
        ResearchPipeline(object())  # type: ignore[arg-type]

    with pytest.raises(ResearchInputInvalid, match="exact DecisionSnapshot"):
        ResearchPipeline(_source()).load_manifest(object())  # type: ignore[arg-type]


def test_mutated_decision_and_wrong_manifest_type_fail_strict_revalidation() -> None:
    decision = _decision()
    object.__setattr__(decision, "schema_version", "2")
    with pytest.raises(ResearchInputInvalid, match="decision snapshot failed"):
        ResearchPipeline(_source()).load_manifest(decision)

    source = _source()
    source.manifest = object()  # type: ignore[assignment]
    with pytest.raises(ResearchInputInvalid, match="manifest has an unsupported type"):
        ResearchPipeline(source).load_manifest(_decision())


@pytest.mark.parametrize("shape", ["list", "oversized"])
def test_regime_history_requires_an_exact_resource_bounded_tuple(shape: str) -> None:
    source = _source()
    if shape == "list":
        source.regimes = list(source.regimes)  # type: ignore[assignment]
        message = "regime history must be an exact tuple"
    else:
        source.regimes = (source.regimes[0],) * 10_001
        message = "regime history exceeds its resource limit"

    with pytest.raises(ResearchInputInvalid, match=message):
        ResearchPipeline(source).detect_market_regime(_decision())


def test_regime_history_rejects_version_order_and_incomplete_boundary() -> None:
    source = _source()
    first = source.regimes[0]
    source.regimes = (
        replace(first, trend=replace(first.trend, data_version="other-v1")),
        *source.regimes[1:],
    )
    with pytest.raises(ResearchInputInvalid, match="data version"):
        ResearchPipeline(source).detect_market_regime(_decision())

    source = _source()
    source.regimes = (source.regimes[1], source.regimes[0], source.regimes[2])
    with pytest.raises(ResearchInputInvalid, match="strictly chronological"):
        ResearchPipeline(source).detect_market_regime(_decision())

    source = _source()
    source.regimes = source.regimes[:-1]
    with pytest.raises(ResearchInputInvalid, match="end at the decision time"):
        ResearchPipeline(source).detect_market_regime(_decision())


def test_domain_regime_failure_and_industry_mismatch_are_typed_input_errors() -> None:
    source = _source()
    final = source.regimes[-1]
    source.regimes = (
        *source.regimes[:-1],
        replace(final, breadth=replace(final.breadth, above_average_ratio=None)),
    )
    with pytest.raises(ResearchInputInvalid, match="regime history failed domain alignment"):
        ResearchPipeline(source).detect_market_regime(_decision())

    source = _source()
    source.industries = source.industries[:-1]
    with pytest.raises(ResearchInputInvalid, match="same sessions"):
        ResearchPipeline(source).rank_market_themes(_decision())

    source = _source()
    source.industries = (
        replace(source.industries[0], data_version="other-v1"),
        *source.industries[1:],
    )
    with pytest.raises(ResearchInputInvalid, match="industry history row"):
        ResearchPipeline(source).rank_market_themes(_decision())


class _DriftingQualityEngine:
    def run(
        self,
        data_version: str,
        context: QualityContext,
        *,
        session: object = None,
        observed_at: datetime | None = None,
    ) -> QualityReport:
        del context, session
        assert observed_at is not None
        return QualityReport(
            data_version=f"{data_version}-drift",
            observed_at=observed_at,
            issues=(),
        )


def test_quality_context_identity_and_generated_report_cannot_drift() -> None:
    source = _source()
    source.quality = replace(source.quality, data_content_hash="f" * 64)
    with pytest.raises(ResearchInputInvalid, match="quality context does not match"):
        ResearchPipeline(source).validate_market_data(_decision())

    pipeline = ResearchPipeline(_source())
    pipeline._quality_engine = _DriftingQualityEngine()  # type: ignore[assignment]
    with pytest.raises(ResearchInputInvalid, match="quality report drifted"):
        pipeline.validate_market_data(_decision())


@pytest.mark.parametrize("collection", ["tradeability", "fundamentals", "coverage"])
def test_leader_feature_collections_are_nonempty_and_exactly_aligned(collection: str) -> None:
    source = _source()
    inputs = source.leader_inputs
    if collection == "tradeability":
        source.leader_inputs = replace(inputs, tradeability=())
        message = "tradeability inputs have an invalid size"
    elif collection == "fundamentals":
        source.leader_inputs = replace(inputs, fundamentals=())
        message = "fundamental inputs have an invalid size"
    else:
        source.leader_inputs = replace(inputs, tradeability=inputs.tradeability[:-1])
        message = "must cover identical unique instruments"

    with pytest.raises(ResearchInputInvalid, match=message):
        ResearchPipeline(source).rank_theme_leaders(_decision())


def test_leader_domain_mismatch_and_candidate_domain_mismatch_are_typed() -> None:
    source = _source()
    inputs = source.leader_inputs
    source.leader_inputs = replace(
        inputs,
        stock_strength=(
            replace(inputs.stock_strength[0], classification_version="OTHER"),
            *inputs.stock_strength[1:],
        ),
    )
    with pytest.raises(ResearchInputInvalid, match="leader inputs failed domain alignment"):
        ResearchPipeline(source).rank_theme_leaders(_decision())

    unranked = candidate_fixtures._signal(
        "UNRANKED",
        candidate_fixtures.CandidateSignalKind.EVENT,
        score="90",
    )
    with pytest.raises(ResearchInputInvalid, match="candidate inputs failed domain alignment"):
        ResearchPipeline(_source(signals=(unranked,))).rank_stock_candidates(_decision())


def test_candidate_signal_collection_requires_an_exact_tuple() -> None:
    source = _source()
    source.signals = []  # type: ignore[assignment]

    with pytest.raises(ResearchInputInvalid, match="signals must be an exact tuple"):
        ResearchPipeline(source).rank_stock_candidates(_decision())
