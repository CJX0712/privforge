"""Top-level verification gate for PrivForge.

Runs, in order:
  1. ``ruff check .``
  2. ``ruff format --check .``
  3. ``pytest -q -W ignore::UserWarning``

Each step is reported as PASS / FAIL / SKIP. ``ruff`` is optional: if it is not
installed the step is SKIPped rather than failed. ``pytest`` returning 5 (no
tests collected) is treated as a PASS because the project may not yet ship
tests. Prints a summary and exits with code 1 only when a required step failed.
Author: 晨星
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ruff_cmd() -> list[str] | None:
    """Return a ruff command list, or None when ruff is not installed."""
    sibling = os.path.join(os.path.dirname(sys.executable), "ruff.exe")
    if os.path.exists(sibling):
        return [sibling]
    if shutil.which("ruff") is not None:
        return ["ruff"]
    return None


def _run(cmd: list[str], cwd: str) -> int:
    """Run ``cmd`` in ``cwd``; return the process return code."""
    proc = subprocess.run(cmd, cwd=cwd)
    return int(proc.returncode)


def main() -> int:
    root = _repo_root()
    print(f"PrivForge verification (repo: {root})\n")

    overall_ok = True

    # 1. ruff check
    ruff = _ruff_cmd()
    if ruff is None:
        print("[SKIP] ruff check: ruff not installed")
    else:
        rc = _run([*ruff, "check", "."], root)
        if rc == 0:
            print(f"[PASS] ruff check (rc={rc})")
        else:
            print(f"[FAIL] ruff check (rc={rc})")
            overall_ok = False

    # 2. ruff format --check
    if ruff is None:
        print("[SKIP] ruff format --check: ruff not installed")
    else:
        rc = _run([*ruff, "format", "--check", "."], root)
        if rc == 0:
            print(f"[PASS] ruff format --check (rc={rc})")
        else:
            print(f"[FAIL] ruff format --check (rc={rc})")
            overall_ok = False

    # 3. pytest
    rc = _run([sys.executable, "-m", "pytest", "-q", "-W", "ignore::UserWarning"], root)
    if rc == 0 or rc == 5:
        print(f"[PASS] pytest (rc={rc})")
    else:
        print(f"[FAIL] pytest (rc={rc})")
        overall_ok = False

    print()
    if overall_ok:
        print("VERIFY PASS")
        return 0
    print("VERIFY FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
