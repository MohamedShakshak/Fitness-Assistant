"""FastAPI backend for fitness RAG assistant."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    FiltersResponse,
)
from src.pipeline import query, search_only, get_filters

logger = logging.getLogger(__name__)

app = FastAPI(title="FitAssist RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        result = query(
            question=request.query,
            chat_history=[m.model_dump() for m in request.chat_history] if request.chat_history else None,
            filters=request.filters,
            llm_provider=request.llm_provider,
            top_k=request.top_k,
        )
        return QueryResponse(**result)
    except ValueError as e:
        logger.warning("Query validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Query endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    try:
        results = search_only(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
        )
        return SearchResponse(results=results)
    except Exception as e:
        logger.error("Search endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/filters", response_model=FiltersResponse)
def filters_endpoint():
    try:
        filters = get_filters()
        return FiltersResponse(**filters)
    except Exception as e:
        logger.error("Filters endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def health_endpoint():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)