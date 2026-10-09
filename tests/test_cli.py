"""tests/test_cli.py — CLI 冒烟。"""

from __future__ import annotations

import json
import os

from cli import main


def test_selftest(capsys):
    rc = main(["selftest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "selftest OK" in out


def test_train(capsys):
    rc = main(
        [
            "train",
            "--method",
            "DPFuse",
            "--seed",
            "0",
            "--epochs",
            "2",
            "--n-samples",
            "400",
            "--n-features",
            "12",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "test_accuracy" in out


def test_benchmark_writes_file(tmp_path, capsys):
    out_dir = str(tmp_path)
    rc = main(
        [
            "benchmark",
            "--methods",
            "NonPrivate",
            "DPFuse",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "3",
            "--n-samples",
            "600",
            "--n-features",
            "12",
            "--out-dir",
            out_dir,
        ]
    )
    assert rc == 0
    path = os.path.join(out_dir, "benchmark.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
