"""Unify, clean, and deduplicate exercise datasets into a single schema."""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UNIFIED_SCHEMA = {
    "name": "",
    "description": "",
    "instructions": [],
    "category": "",
    "body_part": "",
    "primary_muscles": [],
    "secondary_muscles": [],
    "equipment": "",
    "level": "",
    "force": "",
    "mechanic": "",
    "source": [],
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_list(val):
    if isinstance(val, list):
        return [normalize_text(item) for item in val if item]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if "," in val:
            return [normalize_text(item) for item in val.split(",") if item.strip()]
        return [val]
    return []


def process_wrkout():
    filepath = RAW_DIR / "wrkout_exercises" / "exercises.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ex in data:
        record = {
            "name": normalize_text(ex.get("name", "")),
            "description": "",
            "instructions": ex.get("instructions", []),
            "category": normalize_text(ex.get("category", "")),
            "body_part": "",
            "primary_muscles": ex.get("primaryMuscles", []),
            "secondary_muscles": ex.get("secondaryMuscles", []),
            "equipment": normalize_text(ex.get("equipment", "")),
            "level": normalize_text(ex.get("level", "")),
            "force": normalize_text(ex.get("force", "")),
            "mechanic": normalize_text(ex.get("mechanic", "")),
            "source": ["wrkout"],
        }
        if record["name"]:
            records.append(record)
    return records


def process_kaggle_gym():
    filepath = RAW_DIR / "kaggle_gym_dataset" / "megaGymDataset.csv"
    df = pd.read_csv(filepath)

    records = []
    for _, row in df.iterrows():
        record = {
            "name": normalize_text(str(row.get("Title", ""))),
            "description": normalize_text(str(row.get("Desc", "") if pd.notna(row.get("Desc")) else "")),
            "instructions": [],
            "category": normalize_text(str(row.get("Type", ""))).lower(),
            "body_part": normalize_text(str(row.get("BodyPart", ""))).lower(),
            "primary_muscles": [],
            "secondary_muscles": [],
            "equipment": normalize_text(str(row.get("Equipment", ""))).lower(),
            "level": normalize_text(str(row.get("Level", ""))).lower(),
            "force": "",
            "mechanic": "",
            "source": ["kaggle_gym"],
        }
        if record["name"]:
            records.append(record)
    return records


def process_kaggle_fitness():
    filepath = RAW_DIR / "kaggle_fitness_exercises" / "exercises.csv"
    df = pd.read_csv(filepath)

    instruction_cols = [c for c in df.columns if c.startswith("instructions/")]
    secondary_cols = [c for c in df.columns if c.startswith("secondaryMuscles/")]

    records = []
    for _, row in df.iterrows():
        instructions = []
        for col in instruction_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                instructions.append(str(val).strip())

        secondary = []
        for col in secondary_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                secondary.append(str(val).strip())

        record = {
            "name": normalize_text(str(row.get("name", ""))),
            "description": "",
            "instructions": instructions,
            "category": "",
            "body_part": normalize_text(str(row.get("bodyPart", ""))).lower(),
            "primary_muscles": [normalize_text(str(row.get("target", "")))] if pd.notna(row.get("target")) else [],
            "secondary_muscles": secondary,
            "equipment": normalize_text(str(row.get("equipment", ""))).lower(),
            "level": "",
            "force": "",
            "mechanic": "",
            "source": ["kaggle_fitness"],
        }
        if record["name"]:
            records.append(record)
    return records


def merge_and_deduplicate(all_records):
    name_map = {}
    for record in all_records:
        key = record["name"].lower().strip()
        if key in name_map:
            existing = name_map[key]
            for field in ["description", "instructions", "primary_muscles", "secondary_muscles", "force", "mechanic", "body_part", "level", "equipment", "category"]:
                if not existing.get(field) and record.get(field):
                    existing[field] = record[field]
            if record["source"][0] not in existing["source"]:
                existing["source"].extend(record["source"])
        else:
            name_map[key] = record

    return sorted(name_map.values(), key=lambda x: x["name"])


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing data sources...")
    print("  [1/3] wrkout/exercises.json...")
    wrkout = process_wrkout()
    print(f"    {len(wrkout)} exercises")

    print("  [2/3] Kaggle Gym Exercise Dataset...")
    kaggle_gym = process_kaggle_gym()
    print(f"    {len(kaggle_gym)} exercises")

    print("  [3/3] Kaggle Fitness Exercises Dataset...")
    kaggle_fitness = process_kaggle_fitness()
    print(f"    {len(kaggle_fitness)} exercises")

    print("\nMerging and deduplicating...")
    all_records = wrkout + kaggle_gym + kaggle_fitness
    print(f"  Total before dedup: {len(all_records)}")

    unified = merge_and_deduplicate(all_records)
    print(f"  Total after dedup: {len(unified)}")

    exercises_path = PROCESSED_DIR / "exercises.json"
    with open(exercises_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"\nSaved exercises to {exercises_path}")

    df = pd.DataFrame(unified)
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Total exercises: {len(df)}")
    print(f"\nBy source:")
    sources = df["source"].explode().value_counts()
    for src, count in sources.items():
        print(f"  {src}: {count}")
    print(f"\nBy category:")
    print(df["category"].value_counts().to_string())
    print(f"\nBy level:")
    print(df["level"].value_counts().to_string())
    print(f"\nBy equipment (top 15):")
    print(df["equipment"].value_counts().head(15).to_string())
    print(f"\nWith instructions: {df['instructions'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()}")
    print(f"With description: {(df['description'] != '').sum()}")
    print(f"With primary muscles: {df['primary_muscles'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()}")


if __name__ == "__main__":
    main()