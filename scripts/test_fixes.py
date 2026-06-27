"""Quick verification of bug fixes."""
import sys
sys.path.insert(0, ".")

from src.vectorstore.qdrant_store import _serialize_metadata, _deserialize_payload, _build_filter

# Test 1: Serialization
metadata = {
    "name": "Bench Press",
    "source": ["wrkout", "kaggle_gym"],
    "primary_muscles": ["chest", "triceps"],
    "level": "beginner",
}
serialized = _serialize_metadata(metadata)
print("=== Serialization ===")
print(f"  source: {serialized['source']}")
print(f"  primary_muscles: {serialized['primary_muscles']}")
print(f"  name: {serialized['name']}")
assert serialized["source"] == '["wrkout", "kaggle_gym"]'
assert serialized["primary_muscles"] == '["chest", "triceps"]'
assert serialized["name"] == "Bench Press"
print("  PASSED")

# Test 2: Deserialization
payload = {
    "text": "Exercise: Bench Press\nCategory: Strength",
    "name": "Bench Press",
    "source": '["wrkout", "kaggle_gym"]',
    "primary_muscles": '["chest", "triceps"]',
    "level": "beginner",
    "has_description": True,
}
deserialized = _deserialize_payload(payload)
print("\n=== Deserialization ===")
print(f"  source: {deserialized['source']}")
print(f"  primary_muscles: {deserialized['primary_muscles']}")
print(f"  source type: {type(deserialized['source'])}")
assert deserialized["source"] == ["wrkout", "kaggle_gym"]
assert deserialized["primary_muscles"] == ["chest", "triceps"]
assert isinstance(deserialized["source"], list)
print("  PASSED")

# Test 3: Filter building - single value
f = _build_filter({"body_part": "chest"})
print("\n=== Single-value filter ===")
print(f"  {f}")
assert f is not None
assert len(f.must) == 1
print("  PASSED")

# Test 4: Filter building - multi value
f2 = _build_filter({"body_part": ["chest", "back"]})
print("\n=== Multi-value filter ===")
print(f"  {f2}")
assert f2 is not None
assert len(f2.must) == 1
print("  PASSED")

# Test 5: Empty filter
f3 = _build_filter(None)
print("\n=== None filter ===")
print(f"  {f3}")
assert f3 is None
print("  PASSED")

# Test 6: format_context with list sources
from src.generation.prompts import format_context
results = [
    {
        "text": "Exercise: Bench Press",
        "score": 0.9,
        "metadata": {
            "name": "Bench Press",
            "source": ["wrkout", "kaggle_gym"],
            "category": "strength",
        },
    }
]
context = format_context(results)
print("\n=== format_context with list source ===")
print(f"  {context}")
assert "[Source: wrkout, kaggle_gym]" in context
print("  PASSED")

print("\n=== ALL TESTS PASSED ===")