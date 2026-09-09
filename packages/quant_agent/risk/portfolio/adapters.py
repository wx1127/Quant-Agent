"""Typed adapters that bind account and proposal identities into risk requests."""

from quant_agent.portfolio import AccountSnapshot, TargetPortfolio
from quant_agent.risk.contracts import RiskCheckRequest

from .contracts import PortfolioRiskContext, PortfolioRiskInputError, PortfolioRiskPolicy


def build_portfolio_risk_request(
    *,
    account: AccountSnapshot,
    proposal: TargetPortfolio,
    context: PortfolioRiskContext,
    policy: PortfolioRiskPolicy,
) -> RiskCheckRequest:
    """Build the exact P5-T03 request consumed by the portfolio risk evaluator."""

    expected = (
        (proposal.account_snapshot_id, account.snapshot_id, "proposal account id"),
        (proposal.account_snapshot_hash, account.content_hash, "proposal account hash"),
        (proposal.as_of, account.as_of, "proposal as_of"),
        (proposal.data_version, account.data_version, "proposal data version"),
        (proposal.total_equity, account.total_equity, "proposal equity"),
        (context.account_snapshot_hash, account.content_hash, "context account hash"),
        (context.as_of, account.as_of, "context as_of"),
        (context.current_equity, account.total_equity, "context equity"),
    )
    for left, right, name in expected:
        if left != right:
            raise PortfolioRiskInputError(f"{name} does not align")
    return RiskCheckRequest.build(
        decision_id=proposal.decision_id,
        account_snapshot_id=account.snapshot_id,
        account_snapshot_hash=account.content_hash,
        account_snapshot_as_of=account.as_of,
        portfolio_proposal_hash=proposal.result_hash,
        data_version=account.data_version,
        valuation_version=context.valuation_version,
        policy_version=policy.version,
        policy_hash=policy.policy_hash,
        risk_context_hash=context.context_hash,
    )


__all__ = ["build_portfolio_risk_request"]
