# LAN Ollama Embedding API

FastAPI server for exposing local Ollama embedding models to LAN/Tailscale clients through one public endpoint:

```text
http://your_ip:3210
```

Confirmed local embedding models:

- `bge-m3:latest`
- `mxbai-embed-large:latest`

## Run

Ollama must already be running on the host. Then start the API:

```powershell
docker compose up --build -d
```

The Compose file publishes only port `3210`, so clients should call `your_ip:3210` over Tailscale DNS/LAN.

## Endpoints

### Health

```bash
curl http://your_ip:3210/health
```

### List models

```bash
curl http://your_ip:3210/v1/models
```

### OpenAI-compatible embeddings

```bash
curl http://your_ip:3210/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3:latest","input":"hello from LAN"}'
```

### Ollama-style embeddings

```bash
curl http://your_ip:3210/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"mxbai-embed-large:latest","input":"hello from LAN"}'
```

## Configuration

Copy `.env.example` to `.env` if you need to override defaults.

```text
OLLAMA_BASE_URL=http://host.docker.internal:11434
ALLOWED_EMBEDDING_MODELS=bge-m3:latest,mxbai-embed-large:latest
DEFAULT_EMBEDDING_MODEL=bge-m3:latest
REQUEST_TIMEOUT_SECONDS=120
```

If Ollama is not reachable from Docker, make sure Ollama is listening on the host and not only inside another isolated container.

## Qdrant usage example

This API acts as the embedding provider. Qdrant acts as the vector store.

The normal retrieval flow is:

1. Send document text to `http://your_ip:3210/v1/embeddings`.
2. Receive the embedding vector.
3. Store `{text, metadata, vector}` in a Qdrant collection.
4. Send user query text to the same embedding API.
5. Search Qdrant with the query vector.
6. Use Qdrant results as context for RAG or semantic search.

Both confirmed local embedding models currently return `1024` dimensions, so the Qdrant collection must use vector size `1024` when using either of these models.

### Install Python clients

```bash
pip install requests qdrant-client
```

### Create a Qdrant collection

This example assumes Qdrant is running at `http://localhost:6333`.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

qdrant = QdrantClient(url="http://localhost:6333")

qdrant.recreate_collection(
    collection_name="docs_bge_m3",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE,
    ),
)
```

### Call the embedding API

```python
import requests

EMBEDDING_API = "http://your_ip:3210/v1/embeddings"
MODEL = "bge-m3:latest"


def embed(text: str) -> list[float]:
    response = requests.post(
        EMBEDDING_API,
        json={
            "model": MODEL,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
```

### Insert documents into Qdrant

```python
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

qdrant = QdrantClient(url="http://localhost:6333")

documents = [
    {
        "title": "Ollama embedding server",
        "text": "FastAPI server exposes local Ollama embedding models to LAN users.",
    },
    {
        "title": "Qdrant vector database",
        "text": "Qdrant stores embedding vectors and supports cosine similarity search.",
    },
]

points = []

for document in documents:
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(document["text"]),
            payload={
                "title": document["title"],
                "text": document["text"],
                "model": MODEL,
            },
        )
    )

qdrant.upsert(
    collection_name="docs_bge_m3",
    points=points,
)
```

### Search documents

```python
query = "How do I expose Ollama embeddings to LAN users?"
query_vector = embed(query)

results = qdrant.search(
    collection_name="docs_bge_m3",
    query_vector=query_vector,
    limit=3,
)

for result in results:
    print("score:", result.score)
    print("title:", result.payload["title"])
    print("text:", result.payload["text"])
    print()
```

### Complete minimal example

```python
import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_API = "http://your_ip:3210/v1/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "docs_bge_m3"
MODEL = "bge-m3:latest"


def embed(text: str) -> list[float]:
    response = requests.post(
        EMBEDDING_API,
        json={"model": MODEL, "input": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


qdrant = QdrantClient(url=QDRANT_URL)

qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

documents = [
    "FastAPI can expose local Ollama embedding models through a LAN API.",
    "Qdrant stores embedding vectors for semantic search.",
    "bge-m3:latest and mxbai-embed-large:latest both return 1024-dimensional embeddings here.",
]

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embed(text),
        payload={"text": text, "model": MODEL},
    )
    for text in documents
]

qdrant.upsert(collection_name=COLLECTION, points=points)

query_vector = embed("I need LAN semantic search using Ollama embeddings")

results = qdrant.search(
    collection_name=COLLECTION,
    query_vector=query_vector,
    limit=2,
)

for result in results:
    print(result.score, result.payload["text"])
```

### Model and collection rules

Use one embedding model per Qdrant collection. Do not mix `bge-m3:latest` and `mxbai-embed-large:latest` in the same collection, even though both currently return `1024` dimensions. Matching dimensions does not mean matching vector spaces.

Recommended collection names:

- `docs_bge_m3`
- `docs_mxbai_embed_large`

Recommended payload fields:

```json
{
  "text": "original chunk text",
  "title": "source title",
  "source": "source file or URL",
  "model": "bge-m3:latest"
}
```

For Chinese or multilingual documents, start with `bge-m3:latest`. For general English-heavy usage, test both models and compare retrieval quality with your real documents.
