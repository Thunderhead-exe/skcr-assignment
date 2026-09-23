"""
Program entry point.

Usage: uv run main.py cut.mp4 [--config config.yaml] [--max-seconds 20] [--output-root outputs]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.team_pressing.cli import main

if __name__ == "__main__":
    # Exit with main()'s return code (0 on success).
    raise SystemExit(main())
