"""Scrape Wikipedia exercise-related articles for RAG content."""

import json
import re
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
WIKI_DIR = DATA_DIR / "wikipedia_exercises"

WIKI_API = "https://en.wikipedia.org/w/api.php"

ARTICLES = [
    "List of weight training exercises",
    "Strength training",
    "Weight training",
    "Bodyweight exercise",
    "Plyometrics",
    "Muscle hypertrophy",
    "Resistance training",
    "Exercise",
    "Physical fitness",
    "Aerobic exercise",
    "Anaerobic exercise",
    "High-intensity interval training",
    "CrossFit",
    "Bench press",
    "Squat (exercise)",
    "Deadlift",
    "Pull-up",
    "Push-up",
    "Shoulder press",
    "Biceps curl",
    "Lunge (exercise)",
    "Plank (exercise)",
    "Burpee (exercise)",
    "Kettlebell",
    "Barbell",
    "Dumbbell",
    "Muscle",
    "Pectoralis major",
    "Latissimus dorsi",
    "Quadriceps",
    "Hamstring",
    "Gluteus maximus",
    "Rectus abdominis",
    "Deltoid muscle",
    "Biceps",
    "Triceps",
    "Calves",
    "Rotator cuff",
    "VO2 max",
    "One-repetition maximum",
    "Progressive overload",
    "Periodization (fitness)",
    "Delayed onset muscle soreness",
]


WIKI_HEADERS = {
    "User-Agent": "FitnessRAG/1.0 (https://github.com/fitness-rag; fitness-rag@example.com)",
    "Accept": "application/json",
}


def fetch_article(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "redirects": True,
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        page = next(iter(data["query"]["pages"].values()))
        if "missing" in page:
            return None
        content = page.get("extract", "").strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        return {"title": page.get("title", title), "pageid": page.get("pageid", 0), "content": content}
    except Exception as e:
        print(f"  Error fetching '{title}': {e}")
        return None


def main():
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(ARTICLES)} Wikipedia articles...")
    all_articles = []

    for i, title in enumerate(ARTICLES):
        print(f"  [{i+1}/{len(ARTICLES)}] {title}")
        article = fetch_article(title)
        if article and len(article["content"]) > 100:
            all_articles.append(article)
        time.sleep(0.5)

    output_path = WIKI_DIR / "wikipedia_exercises.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_articles)} articles to {output_path}")
    total_chars = sum(len(a["content"]) for a in all_articles)
    print(f"Total content: {total_chars:,} characters")


if __name__ == "__main__":
    main()