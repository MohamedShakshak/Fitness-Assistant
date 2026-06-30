"""Multi-query expansion — generates alternative search queries using the LLM."""

from langchain_core.prompts import ChatPromptTemplate

EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a search query expansion assistant for a fitness exercise database. "
               "Generate 3 alternative search queries for the given question. "
               "Use different terminology, synonyms, and phrasings. "
               "Return only the queries, one per line, no numbering or extra text."),
    ("human", "Original: {query}"),
])


def expand_query(query: str, llm=None, num_variants: int = 3) -> list[str]:
    """Generate alternative search queries using the LLM."""
    if llm is None:
        return [query]

    try:
        chain = EXPANSION_PROMPT | llm
        result = chain.invoke({"query": query})
        content = result.content if hasattr(result, "content") else str(result)
        variants = [q.strip() for q in content.strip().split("\n") if q.strip()]
        variants = [q for q in variants if len(q) > 5]
        return [query] + variants[:num_variants]
    except Exception:
        return [query]
