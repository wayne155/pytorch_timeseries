# tests/leaderboard/test_build_script.py
"""Smoke test for the build script CLI entrypoint."""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys

import yaml


def test_build_script_runs(tmp_path):
    """Build script runs without error when dirs are empty."""
    cfg = {
        "leaderboard_results_dir": str(tmp_path / "lr"),
        "results_dir": str(tmp_path / "results"),
        "views_dir": str(tmp_path / "views"),
        "entries_dir": str(tmp_path / "entries"),
        "out": str(tmp_path / "out.json"),
    }
    cfg_path = tmp_path / "leaderboard.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    result = subprocess.run(
        [sys.executable, "leaderboard/build_leaderboard.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "out.json").read_text())
    assert "views" in data
    assert "generated_at" in data
