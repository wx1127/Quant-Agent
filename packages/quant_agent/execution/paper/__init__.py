"""Credential-free paper account, matching, ledger, and idempotency contracts."""

from .contracts import (
    PAPER_EXECUTION_ENGINE_VERSION,
    PaperAccountState,
    PaperExecutionConfig,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperFill,
    PaperIdempotencyConflict,
    PaperMatchAttempt,
    PaperNoFillReason,
    PaperOrder,
    PaperOrderPolicy,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionLot,
)
from .engine import PaperExecutionEngine
from .repository import (
    InMemoryPaperRepository,
    PaperAccountConflict,
    PaperAccountNotFound,
    PaperConcurrentUpdate,
    PaperRepository,
    PaperRepositoryError,
)
from .service import PaperExecutionService, PaperNewOrderGate
from .sqlite_repository import SQLitePaperRepository

__all__ = [
    "PAPER_EXECUTION_ENGINE_VERSION",
    "InMemoryPaperRepository",
    "PaperAccountConflict",
    "PaperAccountNotFound",
    "PaperAccountState",
    "PaperConcurrentUpdate",
    "PaperExecutionConfig",
    "PaperExecutionEngine",
    "PaperExecutionInputError",
    "PaperExecutionReceipt",
    "PaperExecutionRequest",
    "PaperExecutionService",
    "PaperFill",
    "PaperIdempotencyConflict",
    "PaperMatchAttempt",
    "PaperNewOrderGate",
    "PaperNoFillReason",
    "PaperOrder",
    "PaperOrderPolicy",
    "PaperOrderStatus",
    "PaperPosition",
    "PaperPositionLot",
    "PaperRepository",
    "PaperRepositoryError",
    "SQLitePaperRepository",
]
