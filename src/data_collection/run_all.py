"""Master data collection script - runs all downloaders and scrapers."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src" / "data_collection"

SCRIPTS = [
    ("download_all.py", "Downloading Priority 1 datasets (wrkout, Kaggle)"),
]


def main():
    print("=" * 60)
    print("FITNESS RAG - Data Collection Pipeline")
    print("=" * 60)

    for script, description in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"  Running: {script}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / script)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")
        else:
            print(f"  Done: {script}")

    print(f"\n{'='*60}")
    print("Data collection pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()