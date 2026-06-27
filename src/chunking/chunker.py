"""Chunk exercise records into LangChain Documents with labeled-section format."""

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import settings


class ExerciseChunker:
    def format_exercise(self, record: dict) -> str:
        parts = []

        name = record.get("name", "Unknown Exercise")
        parts.append(f"Exercise: {name}")

        category = record.get("category", "")
        mechanic = record.get("mechanic", "")
        if category and mechanic:
            parts.append(f"Category: {category.capitalize()} ({mechanic.capitalize()})")
        elif category:
            parts.append(f"Category: {category.capitalize()}")

        primary = record.get("primary_muscles", [])
        if primary:
            primary_str = ", ".join(primary) if isinstance(primary, list) else primary
            body_part = record.get("body_part", "")
            if body_part:
                parts.append(f"Target: {body_part.capitalize()} ({primary_str})")
            else:
                parts.append(f"Target: {primary_str}")

        secondary = record.get("secondary_muscles", [])
        if secondary:
            sec_str = ", ".join(secondary) if isinstance(secondary, list) else secondary
            parts.append(f"Secondary: {sec_str}")

        equipment = record.get("equipment", "")
        if equipment:
            parts.append(f"Equipment: {equipment}")

        level = record.get("level", "")
        if level:
            parts.append(f"Level: {level.capitalize()}")

        force = record.get("force", "")
        if force:
            parts.append(f"Force: {force.capitalize()}")

        header = "\n".join(parts)

        description = record.get("description", "")
        instructions = record.get("instructions", [])

        body_parts = []
        if description:
            body_parts.append(description)

        if instructions:
            if isinstance(instructions, list):
                if len(instructions) == 1:
                    body_parts.append(f"How to perform: {instructions[0]}")
                else:
                    numbered = [f"{j+1}. {step}" for j, step in enumerate(instructions)]
                    body_parts.append("How to perform:\n" + "\n".join(numbered))
            else:
                body_parts.append(f"How to perform: {instructions}")

        if body_parts:
            return header + "\n\n" + "\n\n".join(body_parts)
        return header

    def chunk(self, exercises: list[dict]) -> list[Document]:
        documents = []
        for i, record in enumerate(exercises):
            has_description = bool(record.get("description", ""))
            has_instructions = bool(record.get("instructions", []))

            if not has_description and not has_instructions:
                continue

            name = record.get("name", f"exercise_{i}")
            text = self.format_exercise(record)
            if not text.strip():
                continue

            metadata = {
                "name": name,
                "category": record.get("category", ""),
                "body_part": record.get("body_part", ""),
                "primary_muscles": record.get("primary_muscles", []),
                "secondary_muscles": record.get("secondary_muscles", []),
                "equipment": record.get("equipment", ""),
                "level": record.get("level", ""),
                "force": record.get("force", ""),
                "mechanic": record.get("mechanic", ""),
                "source": record.get("source", []),
                "has_instructions": has_instructions,
                "has_description": has_description,
                "chunk_type": "exercise",
            }

            chunk_id = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())[:80]
            chunk_id = f"ex_{chunk_id}_{i}"

            documents.append(Document(
                id=chunk_id,
                page_content=text,
                metadata=metadata,
            ))

        return documents


def load_and_chunk() -> list[Document]:
    exercises_path = settings.PROCESSED_DIR / "exercises.json"

    with open(exercises_path, "r", encoding="utf-8") as f:
        exercises = json.load(f)

    total = len(exercises)
    chunker = ExerciseChunker()
    documents = chunker.chunk(exercises)
    skipped = total - len(documents)
    print(f"  Loaded {total} exercises, chunked {len(documents)}, skipped {skipped} (no content)")
    return documents


def save_chunks(documents: list[Document], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    data = []
    for doc in documents:
        data.append({
            "id": doc.id,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    documents = load_and_chunk()
    docs_path = save_chunks(documents, settings.CHUNKS_DIR / "exercises", "exercise_chunks.json")

    print(f"Documents: {len(documents)} -> {docs_path}")

    if documents:
        print(f"\nSample document:")
        sample = documents[0]
        print(f"  ID: {sample.id}")
        print(f"  Page content (first 300 chars): {sample.page_content[:300]}...")
        print(f"  Metadata keys: {list(sample.metadata.keys())}")