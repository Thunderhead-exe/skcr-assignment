"""
Logging setup, per-stage timing and run metadata (environment, git commit).
"""

import logging
import platform
import random
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics

# Format of every log line: time | level | module | message.
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    """
    Log to the console and to the run's log file with the same format.

    Existing handlers are removed first, so calling it again (e.g. a second run in the same
    process) does not duplicate every line.

    Args:
        log_file: path of pipeline.log in the run folder.
        level: minimum level (DEBUG, INFO, WARNING, ERROR).
    """
    root = logging.getLogger()  # root logger: every module logger propagates to it
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)


class StageTimer:
    """
    Accumulates wall time per pipeline stage: `with timer("inference"): ...`.

    Attributes:
        totals: seconds spent in each stage so far (missing stages read as 0.0).
    """

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, stage: str):
        """Time the body of the `with` block and add it to `totals[stage]`, even if it raises."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[stage] += time.perf_counter() - t0


def seed_everything(seed: int) -> None:
    """Seed every random number generator the pipeline may use (Python, numpy, OpenCV, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    """
    Short commit hash of the code, with "-dirty" if there are uncommitted changes.

    Returns "no-commit" in a repository without commits, and "unknown" if git is unavailable.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or "no-commit"
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        return commit + ("-dirty" if dirty.stdout.strip() else "")  # any output = uncommitted changes
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info() -> dict:
    """Versions and platform, logged and saved in metrics.json so a run can be reproduced."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "git_commit": git_commit(),
    }
