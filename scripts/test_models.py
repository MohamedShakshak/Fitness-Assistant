"""Test different free models on OpenRouter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import time
from src.config import settings

models = [
    "liquid/lfm-2.5-1.2b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

for model in models:
    print(f"Trying {model}...")
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "FitAssist",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello, say hi in one word"}],
                "max_tokens": 20,
            },
            timeout=30.0,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  Response: {content[:200]}")
            print(f"  SUCCESS with {model}")
            break
        else:
            print(f"  Error: {resp.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")
    time.sleep(3)