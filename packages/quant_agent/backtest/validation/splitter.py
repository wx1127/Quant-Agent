"""Chronological holdout and walk-forward split construction."""

from collections.abc import Iterable
from datetime import date

from quant_agent.backtest.validation.contracts import (
    WalkForwardFold,
    WalkForwardPlan,
    WalkForwardSplitConfig,
    WalkForwardValidationError,
)
from quant_agent.regime.contracts import stable_hash


def build_walk_forward_plan(
    sessions: Iterable[date],
    config: WalkForwardSplitConfig | None = None,
) -> WalkForwardPlan:
    """Create complete sequential folds without sorting or shuffling caller input."""

    active_config = config or WalkForwardSplitConfig()
    frozen = tuple(sessions)
    if tuple(sorted(set(frozen))) != frozen:
        raise WalkForwardValidationError(
            "trading sessions must already be unique and strictly chronological"
        )
    required = (
        active_config.train_sessions
        + active_config.validation_sessions
        + active_config.out_of_sample_sessions
    )
    if len(frozen) < required:
        raise WalkForwardValidationError(
            f"walk-forward calendar has {len(frozen)} sessions; requires at least {required}"
        )

    folds: list[WalkForwardFold] = []
    offset = 0
    while True:
        train_start = 0 if active_config.expanding_train else offset
        train_end = active_config.train_sessions + offset
        validation_end = train_end + active_config.validation_sessions
        oos_end = validation_end + active_config.out_of_sample_sessions
        if oos_end > len(frozen):
            break
        fold_id = f"WF-{len(folds) + 1:03d}"
        train = frozen[train_start:train_end]
        validation = frozen[train_end:validation_end]
        out_of_sample = frozen[validation_end:oos_end]
        fold_hash = stable_hash(
            {
                "fold_id": fold_id,
                "out_of_sample_sessions": out_of_sample,
                "split_config_hash": active_config.config_hash,
                "train_sessions": train,
                "validation_sessions": validation,
            }
        )
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_sessions=train,
                validation_sessions=validation,
                out_of_sample_sessions=out_of_sample,
                split_config_hash=active_config.config_hash,
                fold_hash=fold_hash,
            )
        )
        offset += active_config.step_sessions
    input_hash = stable_hash({"config_hash": active_config.config_hash, "sessions": frozen})
    result_hash = stable_hash(
        {"folds": [item.fold_hash for item in folds], "input_hash": input_hash}
    )
    return WalkForwardPlan(
        sessions=frozen,
        split_config_version=active_config.version,
        split_config_hash=active_config.config_hash,
        folds=tuple(folds),
        input_hash=input_hash,
        result_hash=result_hash,
    )


__all__ = ["build_walk_forward_plan"]
