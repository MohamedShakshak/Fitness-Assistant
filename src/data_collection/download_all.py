"""Download all Priority 1 data sources that can be fetched programmatically."""

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_wrkout():
    dest = DATA_DIR / "wrkout_exercises"
    repo_dir = dest / "repo"

    if not (repo_dir / ".git").exists():
        print("Cloning wrkout/exercises.json repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/wrkout/exercises.json.git", str(repo_dir)],
            check=True,
        )
    else:
        print("wrkout repo already exists, skipping...")

    exercises_dir = repo_dir / "exercises"
    all_exercises = []
    for exercise_folder in sorted(exercises_dir.iterdir()):
        if not exercise_folder.is_dir():
            continue
        exercise_file = exercise_folder / "exercise.json"
        if exercise_file.exists():
            with open(exercise_file, "r", encoding="utf-8") as f:
                all_exercises.append(json.load(f))

    output_path = dest / "exercises.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_exercises, f, indent=2)
    print(f"  Saved {len(all_exercises)} exercises to {output_path}")


def download_kaggle_datasets():
    datasets = [
        ("niharika41298/gym-exercise-data", "kaggle_gym_dataset"),
        ("omarxadel/fitness-exercises-dataset", "kaggle_fitness_exercises"),
    ]
    for slug, folder in datasets:
        dest = DATA_DIR / folder
        os.makedirs(dest, exist_ok=True)
        if list(dest.glob("*.csv")):
            print(f"  {folder}: CSV files already exist, skipping...")
            continue
        print(f"  Downloading {slug}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True,
        )
        print(f"  Downloaded to {dest}")


def main():
    print("=" * 60)
    print("Phase 1: Data Collection")
    print("=" * 60)

    print("\n[1/2] Downloading wrkout/exercises.json...")
    download_wrkout()

    print("\n[2/2] Downloading Kaggle datasets...")
    download_kaggle_datasets()

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)

    print("\nNOTE: The following need manual download:")
    print("  - MET Compendium: Request Excel from pacompendium.com or email compendiumpa@gmail.com")
    print("  - WHO Guidelines: https://www.who.int/publications/i/item/9789240015128")
    print("  - HHS Guidelines: https://health.gov/our-work/nutrition-physical-activity/physical-activity-guidelines/current-guidelines")
    print("  Save PDFs to: data/raw/guidelines/")


if __name__ == "__main__":
    main()