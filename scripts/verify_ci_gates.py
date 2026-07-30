"""Prove that format, type and test faults are rejected by the configured tools."""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    rejected_fault: bool
    return_code: int


def _run(arguments: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def verify() -> tuple[GateResult, ...]:
    with TemporaryDirectory(prefix="quant-agent-ci-") as directory:
        root = Path(directory)
        format_fault = root / "format_fault.py"
        format_fault.write_text(
            "def deliberately_bad( value:int)->int:\n return value\n",
            encoding="utf-8",
        )
        type_fault = root / "type_fault.py"
        type_fault.write_text(
            "def deliberately_bad(value: int) -> str:\n    return value\n",
            encoding="utf-8",
        )
        test_fault = root / "test_fault.py"
        test_fault.write_text(
            "def test_deliberate_failure() -> None:\n    assert False\n",
            encoding="utf-8",
        )
        checks = (
            ("format", _run(["ruff", "format", "--check", str(format_fault)], root)),
            ("type", _run(["mypy", "--strict", str(type_fault)], root)),
            (
                "test",
                _run(
                    [
                        "pytest",
                        "-q",
                        "-o",
                        "addopts=",
                        str(test_fault),
                    ],
                    root,
                ),
            ),
        )
    return tuple(
        GateResult(gate, process.returncode != 0, process.returncode) for gate, process in checks
    )


def main() -> int:
    results = verify()
    print(json.dumps([asdict(item) for item in results], indent=2))
    return 0 if all(item.rejected_fault for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
