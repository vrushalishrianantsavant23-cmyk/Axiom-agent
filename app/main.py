from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, verify, adversarial, ingest, trajectory


app = FastAPI(
    title="Axiom-Agent",
    description="Self-verifying agentic AI system for claim and document fact-checking.",
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
