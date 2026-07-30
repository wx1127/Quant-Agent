"""System-level golden replay contracts and executor."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from quant_agent.agent.evaluation import GOLDEN_SCENARIOS, ExpectedOutcome


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    name: str
    passed: bool
    observed: str


@dataclass(frozen=True, slots=True)
class ScenarioExecution:
    scenario_id: str
    actual: ExpectedOutcome
    evidence: tuple[ScenarioEvidence, ...]


@dataclass(frozen=True, slots=True)
class GoldenReplayResult:
    scenario_id: str
    expected: ExpectedOutcome
    actual: ExpectedOutcome | None
    evidence: tuple[ScenarioEvidence, ...]
    passed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GoldenReplayReport:
    results: tuple[GoldenReplayResult, ...]

    @property
    def passed(self) -> bool:
        return len(self.results) == 12 and all(item.passed for item in self.results)


ScenarioExecutor = Callable[[], ScenarioExecution]


class SystemGoldenReplay:
    """Execute all twelve real component scenarios and compare observed outcomes."""

    def __init__(self, executors: Mapping[str, ScenarioExecutor]) -> None:
        required = {scenario.scenario_id for scenario in GOLDEN_SCENARIOS}
        supplied = set(executors)
        if supplied != required:
            missing = sorted(required - supplied)
            extra = sorted(supplied - required)
            raise ValueError(f"golden executors mismatch: missing={missing}, extra={extra}")
        self._executors = dict(executors)

    def run(self) -> GoldenReplayReport:
        results = []
        for scenario in GOLDEN_SCENARIOS:
            try:
                execution = self._executors[scenario.scenario_id]()
                evidence_passed = bool(execution.evidence) and all(
                    item.passed for item in execution.evidence
                )
                passed = (
                    execution.scenario_id == scenario.scenario_id
                    and execution.actual is scenario.expected
                    and evidence_passed
                )
                results.append(
                    GoldenReplayResult(
                        scenario.scenario_id,
                        scenario.expected,
                        execution.actual,
                        execution.evidence,
                        passed,
                    )
                )
            except Exception as exc:
                results.append(
                    GoldenReplayResult(
                        scenario.scenario_id,
                        scenario.expected,
                        None,
                        (),
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        return GoldenReplayReport(tuple(results))
