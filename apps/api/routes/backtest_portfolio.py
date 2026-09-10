"""Controlled backtest and portfolio proposal endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from core.app import Principal, _principal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy_id: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    start: datetime
    end: datetime
    initial_capital: float = Field(gt=0)


class BacktestTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    status: TaskStatus
    strategy_id: str
    strategy_version: str
    submitted_at: datetime
    result: dict[str, object] | None = None


class PortfolioProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str
    as_of: datetime
    strategy_id: str
    strategy_version: str
    risk_status: str
    positions: tuple[dict[str, object], ...] = ()
    executable: bool = False


class ProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy_id: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    as_of: datetime
    risk_status: str = Field(min_length=1)
    positions: tuple[dict[str, object], ...] = ()


class BacktestPortfolioService:
    def __init__(self, published_strategies: frozenset[tuple[str, str]] = frozenset()) -> None:
        self._published = published_strategies
        self._tasks: dict[str, BacktestTask] = {}
        self._proposals: dict[str, PortfolioProposal] = {}

    def submit(self, request: BacktestRequest) -> BacktestTask:
        if (request.strategy_id, request.strategy_version) not in self._published:
            raise HTTPException(status_code=422, detail="strategy is not published")
        if request.end <= request.start:
            raise HTTPException(status_code=422, detail="backtest end must be after start")
        task = BacktestTask(
            task_id=str(uuid4()),
            status=TaskStatus.QUEUED,
            strategy_id=request.strategy_id,
            strategy_version=request.strategy_version,
            submitted_at=datetime.now(UTC),
        )
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> BacktestTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="backtest task not found")
        return task

    def complete(self, task_id: str, result: dict[str, object]) -> BacktestTask:
        task = self.get_task(task_id)
        updated = task.model_copy(update={"status": TaskStatus.SUCCEEDED, "result": result})
        self._tasks[task_id] = updated
        return updated

    def create_proposal(self, request: ProposalRequest) -> PortfolioProposal:
        if (request.strategy_id, request.strategy_version) not in self._published:
            raise HTTPException(status_code=422, detail="strategy is not published")
        proposal = PortfolioProposal(
            proposal_id=str(uuid4()),
            as_of=request.as_of,
            strategy_id=request.strategy_id,
            strategy_version=request.strategy_version,
            risk_status=request.risk_status,
            positions=request.positions,
        )
        return self.add_proposal(proposal)

    def add_proposal(self, proposal: PortfolioProposal) -> PortfolioProposal:
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def get_proposal(self, proposal_id: str) -> PortfolioProposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="portfolio proposal not found")
        return proposal


def build_backtest_portfolio_router(service: BacktestPortfolioService | None = None) -> APIRouter:
    backend = service or BacktestPortfolioService()
    router = APIRouter(prefix="/backtest-portfolio", tags=["backtest-portfolio"])

    @router.post("/backtests", response_model=BacktestTask, status_code=202)
    async def create_backtest(
        request: BacktestRequest,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> BacktestTask:
        del principal
        return backend.submit(request)

    @router.get("/backtests/{task_id}", response_model=BacktestTask)
    async def get_backtest(
        task_id: str,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> BacktestTask:
        del principal
        return backend.get_task(task_id)

    @router.get("/proposals/{proposal_id}", response_model=PortfolioProposal)
    async def get_proposal(
        proposal_id: str,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> PortfolioProposal:
        del principal
        return backend.get_proposal(proposal_id)

    @router.post("/proposals", response_model=PortfolioProposal, status_code=201)
    async def create_proposal(
        request: ProposalRequest,
        principal: Principal = Depends(_principal),  # noqa: B008
    ) -> PortfolioProposal:
        del principal
        return backend.create_proposal(request)

    return router
