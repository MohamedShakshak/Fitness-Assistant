"""Download wrkout/exercises.json dataset from GitHub via git clone."""

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "wrkout_exercises"
REPO_DIR = DATA_DIR / "repo"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not (REPO_DIR / ".git").exists():
        print("Cloning wrkout/exercises.json repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/wrkout/exercises.json.git", str(REPO_DIR)],
            check=True,
        )
    else:
        print("Repo already cloned, pulling latest...")
        subprocess.run(["git", "pull"], cwd=str(REPO_DIR), check=True)

    exercises_dir = REPO_DIR / "exercises"
    all_exercises = []
    for exercise_folder in sorted(exercises_dir.iterdir()):
        if not exercise_folder.is_dir():
            continue
        exercise_file = exercise_folder / "exercise.json"
        if exercise_file.exists():
            with open(exercise_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_exercises.append(data)

    output_path = DATA_DIR / "exercises.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_exercises, f, indent=2)

    print(f"Saved {len(all_exercises)} exercises to {output_path}")


if __name__ == "__main__":
    main()