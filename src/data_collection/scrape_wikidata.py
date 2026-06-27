"""Scrape Wikidata for structured exercise-muscle-equipment relationships."""

import json
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
WIKIDATA_DIR = DATA_DIR / "wikidata"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

QUERY_EXERCISES = """
SELECT ?exercise ?exerciseLabel ?muscle ?muscleLabel ?equipment ?equipmentLabel ?description WHERE {
  ?exercise wdt:P31 wd:Q3268436.
  OPTIONAL { ?exercise wdt:P405 ?muscle. }
  OPTIONAL { ?exercise wdt:P527 ?equipment. }
  OPTIONAL { ?exercise schema:description ?description. FILTER(LANG(?description) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
"""

QUERY_MUSCLES = """
SELECT ?muscle ?muscleLabel ?musclePartOf ?musclePartOfLabel WHERE {
  ?muscle wdt:P279* wd:Q505864.
  OPTIONAL { ?muscle wdt:P361 ?musclePartOf. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
"""


def run_sparql(query):
    headers = {"Accept": "application/sparql-results+json", "User-Agent": "FitnessRAG-Bot/1.0"}
    resp = requests.get(WIKIDATA_ENDPOINT, params={"query": query}, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


def main():
    WIKIDATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Wikidata SPARQL queries...")

    print("  Fetching exercise-muscle-equipment relationships...")
    exercises = run_sparql(QUERY_EXERCISES)
    exercise_data = []
    for row in exercises:
        exercise_data.append({
            "exercise": row.get("exerciseLabel", {}).get("value", ""),
            "exercise_uri": row.get("exercise", {}).get("value", ""),
            "target_muscle": row.get("muscleLabel", {}).get("value", ""),
            "target_muscle_uri": row.get("muscle", {}).get("value", ""),
            "equipment": row.get("equipmentLabel", {}).get("value", ""),
            "equipment_uri": row.get("equipment", {}).get("value", ""),
            "description": row.get("description", {}).get("value", ""),
        })

    print("  Fetching muscle anatomy data...")
    muscles = run_sparql(QUERY_MUSCLES)
    muscle_data = []
    for row in muscles:
        muscle_data.append({
            "muscle": row.get("muscleLabel", {}).get("value", ""),
            "muscle_uri": row.get("muscle", {}).get("value", ""),
            "part_of": row.get("musclePartOfLabel", {}).get("value", ""),
            "part_of_uri": row.get("musclePartOf", {}).get("value", ""),
        })

    exercises_path = WIKIDATA_DIR / "wikidata_exercises.json"
    with open(exercises_path, "w", encoding="utf-8") as f:
        json.dump(exercise_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(exercise_data)} exercise relations to {exercises_path}")

    muscles_path = WIKIDATA_DIR / "wikidata_muscles.json"
    with open(muscles_path, "w", encoding="utf-8") as f:
        json.dump(muscle_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(muscle_data)} muscle relations to {muscles_path}")


if __name__ == "__main__":
    main()