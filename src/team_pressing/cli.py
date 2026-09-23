"""
Command-line entry point.
"""

import argparse
from pathlib import Path

from team_pressing.config import load_config
from team_pressing.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    """
    Parse the arguments, apply CLI overrides to the config and run the pipeline.

    Args:
        argv: arguments without the program name; None means sys.argv (tests pass a list).

    Returns:
        Process exit code (0 on success; errors raise and end with a traceback).
    """
    # Step 1: arguments.
    parser = argparse.ArgumentParser(description="Detect players and ball at 10 FPS and analyse pressing.")
    parser.add_argument("video", type=Path, help="input video, e.g. cut.mp4")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="pipeline configuration")
    parser.add_argument("--max-seconds", type=float, help="only process the first N seconds (video.max_duration_s)")
    parser.add_argument("--output-root", type=Path, help="parent folder of the run folders (output.root_dir)")
    args = parser.parse_args(argv)

    # Step 2: configuration. CLI flags override the file; the resolved config is saved with the run.
    cfg = load_config(args.config)
    if args.max_seconds is not None:
        cfg.video.max_duration_s = args.max_seconds
    if args.output_root is not None:
        cfg.output.root_dir = str(args.output_root)

    # Step 3: run.
    run_pipeline(args.video, cfg)
    return 0
