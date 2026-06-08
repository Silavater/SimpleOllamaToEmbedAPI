from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://host.docker.internal:11434"
    allowed_embedding_models: str = "bge-m3:latest,mxbai-embed-large:latest"
    default_embedding_model: str = "bge-m3:latest"
    request_timeout_seconds: float = 120.0

    @property
    def models(self) -> list[str]:
        return [model.strip() for model in self.allowed_embedding_models.split(",") if model.strip()]


settings = Settings()
app = FastAPI(
    title="LAN Ollama Embedding API",
    description="FastAPI proxy exposing local Ollama embedding models to LAN/Tailscale clients.",
    version="1.0.0",
)


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(..., description="Text or list of texts to embed.")
    model: str | None = Field(None, description="Allowed Ollama embedding model name.")


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: dict[str, int]


def resolve_model(model: str | None) -> str:
    selected_model = model or settings.default_embedding_model
    if selected_model not in settings.models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "model_not_allowed",
                "allowed_models": settings.models,
                "requested_model": selected_model,
            },
        )
    return selected_model


def normalize_inputs(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not value:
        raise HTTPException(status_code=400, detail="input must contain at least one text")
    return value


def model_names(payload: dict[str, object]) -> set[str]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return set()

    names: set[str] = set()
    for raw_model in cast(list[object], raw_models):
        if isinstance(raw_model, Mapping):
            model = cast(Mapping[str, object], raw_model)
            name = model.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def float_list(values: object) -> list[float] | None:
    if not isinstance(values, Sequence):
        return None
    return [float(value) for value in values if isinstance(value, int | float)]


async def ollama_get(path: str) -> dict[str, object]:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=settings.request_timeout_seconds) as client:
            response = await client.get(path)
            _ = response.raise_for_status()
            return cast(dict[str, object], response.json())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc


async def embed_text(model: str, text: str) -> list[float]:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=settings.request_timeout_seconds) as client:
            response = await client.post("/api/embed", json={"model": model, "input": text})
            _ = response.raise_for_status()
            payload = cast(dict[str, object], response.json())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama embedding failed: {exc}") from exc

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        first_embedding = cast(object, embeddings[0])
        embedding = float_list(first_embedding)
        if embedding:
            return embedding


    embedding = payload.get("embedding")
    embedding_values = float_list(embedding)
    if embedding_values:
        return embedding_values


    raise HTTPException(status_code=502, detail="Ollama returned an unexpected embedding response")


@app.get("/health")
async def health() -> dict[str, object]:
    payload = await ollama_get("/api/tags")
    installed_models = model_names(payload)
    available_models = [model for model in settings.models if model in installed_models]
    return {
        "status": "ok" if available_models else "degraded",
        "ollama_base_url": settings.ollama_base_url,
        "allowed_models": settings.models,
        "available_models": available_models,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    payload = await ollama_get("/api/tags")
    installed_models = model_names(payload)
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "ollama",
            }
            for model in settings.models
            if model in installed_models
        ],
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    model = resolve_model(request.model)
    texts = normalize_inputs(request.input)
    started_at = time.perf_counter()
    items = [EmbeddingItem(embedding=await embed_text(model, text), index=index) for index, text in enumerate(texts)]

    return EmbeddingResponse(
        data=items,
        model=model,
        usage={
            "prompt_tokens": sum(len(text) for text in texts),
            "total_tokens": sum(len(text) for text in texts),
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
        },
    )


@app.post("/api/embed")
async def ollama_compatible_embed(request: EmbeddingRequest) -> dict[str, object]:
    model = resolve_model(request.model)
    texts = normalize_inputs(request.input)
    embeddings = [await embed_text(model, text) for text in texts]
    return {"model": model, "embeddings": embeddings}
