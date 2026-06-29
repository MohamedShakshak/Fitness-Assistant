"""Extract metadata filters from natural language query using keyword matching."""

import re

INFORMATIONAL_TRIGGERS = [
    "what", "which", "best", "good", "list",
    "exercises", "exercise", "workout",
    "work", "works", "target", "targets", "targeting",
    "build", "builds", "building",
    "can i", "for",
]

BODY_PART_KEYWORDS = {
    "chest": "chest", "pectoral": "chest", "pecs": "chest", "pectoralis": "chest",
    "shoulder": "shoulders", "shoulders": "shoulders", "deltoid": "shoulders",
    "bicep": "biceps", "biceps": "biceps",
    "tricep": "triceps", "triceps": "triceps",
    "quadricep": "quadriceps", "quadriceps": "quadriceps", "quads": "quadriceps",
    "hamstring": "hamstrings", "hamstrings": "hamstrings",
    "glute": "glutes", "glutes": "glutes",
    "back": "back",
    "lat": "lats", "lats": "lats",
    "trap": "traps", "traps": "traps",
    "abdominal": "waist", "abdominals": "waist", "abs": "waist", "core": "waist",
    "calves": "calves", "calf": "calves",
    "forearm": "forearms", "forearms": "forearms",
    "neck": "neck",
}

EQUIPMENT_KEYWORDS = {
    "dumbbell": "dumbbell", "dumbbells": "dumbbell",
    "kettlebell": "kettlebell", "kettlebells": "kettlebell",
    "barbell": "barbell", "barbells": "barbell",
    "cable": "cable", "cables": "cable",
    "bodyweight": "body only", "body weight": "body only",
    "no equipment": "body only",
    "band": "band", "bands": "band",
    "machine": "machine",
    "smith machine": "smith machine",
}

LEVEL_KEYWORDS = {
    "beginner": "beginner", "beginners": "beginner",
    "intermediate": "intermediate",
    "advanced": "expert", "expert": "expert",
}


def _is_informational(query: str) -> bool:
    q = query.lower().strip()
    for trigger in INFORMATIONAL_TRIGGERS:
        if trigger in q:
            return True
    return False


def extract_filters(query: str) -> dict:
    if not _is_informational(query):
        return {}

    q = query.lower().strip()
    words = re.findall(r"[a-z]+", q)
    filters = {}

    for word in words:
        if len(word) < 3:
            continue
        if word in BODY_PART_KEYWORDS:
            val = BODY_PART_KEYWORDS[word]
            if "body_part" not in filters:
                filters["body_part"] = val
            elif isinstance(filters["body_part"], list):
                if val not in filters["body_part"]:
                    filters["body_part"].append(val)
            elif filters["body_part"] != val:
                filters["body_part"] = [filters["body_part"], val]

        if word in EQUIPMENT_KEYWORDS:
            val = EQUIPMENT_KEYWORDS[word]
            if "equipment" not in filters:
                filters["equipment"] = val
            elif isinstance(filters["equipment"], list):
                if val not in filters["equipment"]:
                    filters["equipment"].append(val)
            elif filters["equipment"] != val:
                filters["equipment"] = [filters["equipment"], val]

        if word in LEVEL_KEYWORDS:
            filters["level"] = LEVEL_KEYWORDS[word]

    return filters
