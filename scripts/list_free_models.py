"""List free models on OpenRouter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from src.config import settings

resp = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
    timeout=15.0,
)
data = resp.json()
free_models = [m["id"] for m in data["data"] if ":free" in m["id"]]
print(f"Free models ({len(free_models)}):")
for m in sorted(free_models):
    print(f"  {m}")