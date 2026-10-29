from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import ModelConfig
from .paths import OUTPUTS, ensure_directories


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def write_manifest(cfg: ModelConfig, command: str, started: float, results: dict, path: Path | None = None) -> Path:
    ensure_directories()
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    payload = {
        "run_id": cfg.run_id, "command": command, "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(), "python": platform.python_version(), "packages": packages,
        "config": cfg.to_dict(), "elapsed_seconds": time.perf_counter() - started, "results": results,
    }
    target = path or OUTPUTS / "manifest.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
