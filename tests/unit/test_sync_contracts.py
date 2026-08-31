"""Unit tests for synchronization state contracts."""

import pytest

from quant_agent.data.sync.contracts import (
    InvalidStateTransition,
    SyncPageState,
    SyncRunState,
    ensure_page_transition,
    ensure_run_transition,
)


def test_run_state_machine_accepts_resume_and_rejects_terminal_escape() -> None:
    ensure_run_transition(SyncRunState.PENDING, SyncRunState.RUNNING)
    ensure_run_transition(SyncRunState.RUNNING, SyncRunState.FAILED)
    ensure_run_transition(SyncRunState.FAILED, SyncRunState.RUNNING)
    ensure_run_transition(SyncRunState.RUNNING, SyncRunState.SUCCEEDED)

    with pytest.raises(InvalidStateTransition, match="illegal run transition"):
        ensure_run_transition(SyncRunState.SUCCEEDED, SyncRunState.RUNNING)
    with pytest.raises(InvalidStateTransition, match="illegal run transition"):
        ensure_run_transition(SyncRunState.PENDING, SyncRunState.SUCCEEDED)


def test_page_state_machine_requires_fetch_decode_and_commit_order() -> None:
    ensure_page_transition(SyncPageState.PLANNED, SyncPageState.FETCHING)
    ensure_page_transition(SyncPageState.FETCHING, SyncPageState.RETRY_WAIT)
    ensure_page_transition(SyncPageState.RETRY_WAIT, SyncPageState.FETCHING)
    ensure_page_transition(SyncPageState.FETCHING, SyncPageState.FETCHED)
    ensure_page_transition(SyncPageState.FETCHED, SyncPageState.DECODED)
    ensure_page_transition(SyncPageState.DECODED, SyncPageState.COMMITTED)

    with pytest.raises(InvalidStateTransition, match="illegal page transition"):
        ensure_page_transition(SyncPageState.FETCHED, SyncPageState.COMMITTED)
    with pytest.raises(InvalidStateTransition, match="illegal page transition"):
        ensure_page_transition(SyncPageState.COMMITTED, SyncPageState.FETCHING)
