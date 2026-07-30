from datetime import UTC, datetime

from quant_agent.backtest.contracts import AssetType
from quant_agent.portfolio.builder import PortfolioBuildResult, PortfolioLine
from quant_agent.portfolio.snapshots import AccountSnapshot
from quant_agent.risk.contracts import RiskDecision, RiskRequest
from quant_agent.risk.portfolio import PortfolioRiskEngine, PortfolioRiskPolicy

NOW = datetime(2026, 1, 1, 16, tzinfo=UTC)


def account() -> AccountSnapshot:
    return AccountSnapshot("s1", "a1", NOW, 100_000, 0, (), "paper", "v1")


def portfolio(weight: float = 0.05, turnover: float = 0.05) -> PortfolioBuildResult:
    return PortfolioBuildResult(
        "d1",
        "s1",
        NOW,
        (
            PortfolioLine(
                "S1",
                AssetType.STOCK,
                "TECH",
                0,
                weight,
                weight,
                ("strategy",),
                "signal",
            ),
        ),
        1 - weight,
        turnover,
        {"TECH": weight},
        {"strategy": 0.1},
        "snap",
        "builder-v1",
        (),
    )


def request(
    target: PortfolioBuildResult | None = None,
    *,
    drawdown: float = 0,
    data_complete: bool = True,
    healthy: bool = True,
) -> RiskRequest:
    return RiskRequest(
        "r1",
        "d1",
        account(),
        target or portfolio(),
        drawdown,
        0.8,
        data_complete,
        healthy,
    )


def test_risk_allows_valid_warns_drawdown_and_fails_closed() -> None:
    engine = PortfolioRiskEngine()
    passed = engine.check(request(), checked_at=NOW)
    assert passed.passed and not passed.violations
    warning = engine.check(request(drawdown=0.12), checked_at=NOW)
    assert warning.passed and warning.warnings[0].rule_id == "DRAWDOWN_WARNING"
    stopped = engine.check(request(drawdown=0.16), checked_at=NOW)
    assert not stopped.passed
    assert any(item.rule_id == "DRAWDOWN_STOP" for item in stopped.violations)
    failed = RiskDecision.fail_closed(
        "r1", checked_at=NOW, policy_version="v1", reason="service timeout"
    )
    assert not failed.passed
    assert failed.violations[0].rule_id == "RISK_SERVICE_FAILURE"


def test_tighter_policy_never_increases_allowed_risk_and_locates_violation() -> None:
    target = portfolio(weight=0.08, turnover=0.08)
    loose = PortfolioRiskEngine(
        PortfolioRiskPolicy(
            maximum_stock_weight=0.10,
            maximum_turnover=0.10,
        )
    ).check(request(target), checked_at=NOW)
    tight = PortfolioRiskEngine().check(request(target), checked_at=NOW)
    assert loose.passed
    assert not tight.passed
    finding = next(item for item in tight.violations if item.rule_id == "INSTRUMENT_WEIGHT")
    assert finding.instrument_id == "S1"
    unhealthy = PortfolioRiskEngine().check(
        request(data_complete=False, healthy=False), checked_at=NOW
    )
    assert {item.rule_id for item in unhealthy.violations} == {
        "DATA_INCOMPLETE",
        "DEPENDENCY_UNHEALTHY",
    }
