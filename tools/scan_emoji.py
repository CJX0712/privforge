"""Scan the source tree for non-ASCII emoji / symbol characters in ``.py`` files.

The project intentionally ships Chinese docstrings, so the scanner flags only
characters outside the "normal text" ranges:
  * ASCII (ord <= 0x7F)
  * CJK unified ideographs and extensions (0x3400-0x9FFF)
  * CJK symbols / punctuation (0x3000-0x303F)
  * Hiragana / Katakana (0x3040-0x30FF)
  * Hangul syllables (0xAC00-0xD7A3)
  * Fullwidth / halfwidth forms (0xFF00-0xFFEF)

Any other non-ASCII code point (emoji, dingbats, arrows, mathematical symbols,
etc.) is reported as a suspect. ``scan(path)`` returns a list of
``"file:line: <char>"`` strings; ``main()`` scans ``privforge/`` and exits 1 if
any suspect character is found.
Author: 晨星
"""

from __future__ import annotations

import os
import sys


def _is_normal_text(ch: str) -> bool:
    """True for characters that are acceptable in a source file."""
    o = ord(ch)
    return (
        o <= 0x7F
        or 0x3400 <= o <= 0x9FFF  # CJK ideographs + extension A
        or 0x3000 <= o <= 0x303F  # CJK symbols and punctuation
        or 0x3040 <= o <= 0x30FF  # Hiragana / Katakana
        or 0xAC00 <= o <= 0xD7A3  # Hangul syllables
        or 0xFF00 <= o <= 0xFFEF  # fullwidth / halfwidth forms
    )


def scan(path: str) -> list[str]:
    """Walk ``.py`` files under ``path`` and return suspect ``file:line`` strings."""
    findings: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [
            d
            for d in dirs
            if d
            not in (
                ".git",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                "build",
                "dist",
                ".venv",
                "venv",
            )
        ]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for ch in line:
                            if not _is_normal_text(ch):
                                findings.append(f"{fp}:{lineno}: {ch} (U+{ord(ch):04X})")
            except Exception:
                continue
    return findings


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    target = os.path.join(root, "privforge")
    findings = scan(target)
    if findings:
        print(f"Found {len(findings)} suspect non-ASCII character(s) in {target}:")
        for item in findings:
            print(f"  {item}")
        return 1
    print(f"No suspect non-ASCII characters found in {target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
