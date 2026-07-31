"""Append-only, hash-chained JSONL ledger for shadow-run daily evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from quant_agent.shadow.io import shadow_day_from_mapping, shadow_day_to_mapping
from quant_agent.shadow.models import ShadowDayEvidence

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class ShadowLedgerRecord:
    sequence: int
    previous_record_hash: str
    record_hash: str
    evidence: ShadowDayEvidence


class ShadowEvidenceLedger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = RLock()

    def append(self, evidence: ShadowDayEvidence) -> ShadowLedgerRecord:
        with self._lock:
            records = self.read_all()
            duplicate = next(
                (
                    record
                    for record in records
                    if record.evidence.trading_date == evidence.trading_date
                ),
                None,
            )
            if duplicate is not None:
                if duplicate.evidence == evidence:
                    return duplicate
                raise ValueError("shadow evidence already exists for trading date")
            if records and evidence.trading_date <= records[-1].evidence.trading_date:
                raise ValueError("shadow evidence must be appended in trading-date order")
            previous_hash = records[-1].record_hash if records else _GENESIS_HASH
            payload = shadow_day_to_mapping(evidence)
            record_hash = _record_hash(len(records) + 1, previous_hash, payload)
            sequence = len(records) + 1
            record = {
                "sequence": sequence,
                "previous_record_hash": previous_hash,
                "record_hash": record_hash,
                "evidence": payload,
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            return ShadowLedgerRecord(
                sequence=sequence,
                previous_record_hash=previous_hash,
                record_hash=record_hash,
                evidence=evidence,
            )

    def read_all(self) -> tuple[ShadowLedgerRecord, ...]:
        with self._lock:
            if not self._path.exists():
                return ()
            records: list[ShadowLedgerRecord] = []
            expected_previous = _GENESIS_HASH
            for line_number, line in enumerate(
                self._path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                payload = json.loads(line)
                evidence_payload = payload["evidence"]
                expected_hash = _record_hash(
                    line_number,
                    expected_previous,
                    evidence_payload,
                )
                if payload["sequence"] != line_number:
                    raise ValueError("shadow ledger sequence is invalid")
                if payload["previous_record_hash"] != expected_previous:
                    raise ValueError("shadow ledger hash chain is broken")
                if payload["record_hash"] != expected_hash:
                    raise ValueError("shadow ledger record hash is invalid")
                evidence = shadow_day_from_mapping(evidence_payload)
                records.append(
                    ShadowLedgerRecord(
                        sequence=line_number,
                        previous_record_hash=expected_previous,
                        record_hash=expected_hash,
                        evidence=evidence,
                    )
                )
                expected_previous = expected_hash
            return tuple(records)


def _record_hash(sequence: int, previous_hash: str, evidence: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "sequence": sequence,
            "previous_record_hash": previous_hash,
            "evidence": evidence,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
