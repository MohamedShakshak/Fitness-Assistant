"""Streamlit frontend for fitness RAG assistant."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def main():
    st.set_page_config(page_title="FitAssist - Fitness Exercise Assistant", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        llm_provider = st.selectbox("LLM Provider", ["openrouter", "ollama"], index=0)

        st.header("Filters")
        try:
            resp = httpx.get(f"{API_URL}/filters", timeout=5.0)
            if resp.status_code == 200:
                filters_data = resp.json()
                body_part = st.multiselect("Body Part", filters_data.get("body_part", []))
                equipment = st.multiselect("Equipment", filters_data.get("equipment", []))
                level = st.multiselect("Level", filters_data.get("level", []))
                category = st.multiselect("Category", filters_data.get("category", []))
            else:
                body_part, equipment, level, category = [], [], [], []
                st.warning("Could not load filters. Is the API running?")
        except (httpx.ConnectError, httpx.TimeoutException):
            body_part, equipment, level, category = [], [], [], []
            st.warning("Could not connect to API. Start it with: uv run python -m src.api.main")

        filters = {}
        if body_part:
            filters["body_part"] = body_part[0] if len(body_part) == 1 else body_part
        if equipment:
            filters["equipment"] = equipment[0] if len(equipment) == 1 else equipment
        if level:
            filters["level"] = level[0] if len(level) == 1 else level
        if category:
            filters["category"] = category[0] if len(category) == 1 else category

        if st.button("Clear Chat"):
            st.session_state.messages = []

    st.title("FitAssist - Fitness Exercise Assistant")
    st.markdown("Ask me about exercises, workouts, muscle groups, or form tips!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about exercises, muscles, workouts..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history = st.session_state.messages[-10:]

        with st.chat_message("assistant"):
            with st.spinner("Searching exercises..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/query",
                        json={
                            "query": prompt,
                            "chat_history": chat_history,
                            "filters": filters if filters else None,
                            "llm_provider": llm_provider,
                        },
                        timeout=120.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("Sources"):
                                for src in sources:
                                    st.markdown(
                                        f"- **{src['name']}** | {src.get('category', '')} | "
                                        f"{src.get('body_part', '')} | {src.get('equipment', '')} | "
                                        f"{src.get('level', '')} [Source: {src.get('source_db', '')}]"
                                    )
                    else:
                        st.error(f"API Error: {resp.status_code} - {resp.text}")
                        answer = "Sorry, I encountered an error. Please try again."
                except httpx.ConnectError:
                    answer = "Could not connect to the API. Is it running?"
                    st.error(answer)
                except httpx.TimeoutException:
                    answer = "Request timed out. Please try again."
                    st.error(answer)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()