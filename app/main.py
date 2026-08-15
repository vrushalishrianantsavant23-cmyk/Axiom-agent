from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, verify, adversarial, ingest, trajectory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load heavy models once at startup, so requests don't reload them
    from app.uncertainty.semantic_entropy import get_client, get_embedder
    from app.rag.ingestion import get_chroma_client, get_embedding_fn

    get_client()
    get_embedder()
    get_chroma_client()
    get_embedding_fn()

    yield
    # (nothing needed on shutdown)


app = FastAPI(
    title="Axiom-Agent",
    description="Self-verifying agentic AI system for claim and document fact-checking.",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(verify.router)
app.include_router(adversarial.router)
app.include_router(ingest.router)
app.include_router(trajectory.router)