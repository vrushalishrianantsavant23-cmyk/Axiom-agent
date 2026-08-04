# Axiom-Agent

**A self-verifying agentic AI system for claim and document fact-checking**

Axiom-Agent goes beyond typical RAG chatbots by combining multi-agent verification, research-grounded uncertainty quantification, and safety guardrails to produce grounded, confidence-scored, and transparent answers — rather than confidently hallucinating.

## Why this project

Most GenAI assistants report confidence as a simple self-rated number from the LLM itself, which is often overconfident and poorly calibrated. Axiom-Agent instead measures uncertainty statistically using **semantic entropy** (based on Farquhar et al., *Nature*, 2024) — sampling multiple responses and measuring how consistent the model actually is with itself, rather than trusting what it says about itself.

This directly addresses several open challenges discussed in current agentic AI / LLM research:
- Hallucination detection happening after generation rather than being addressed during it
- Lack of transparency in how agents arrive at conclusions
- Overconfidence and poor uncertainty calibration
- Vulnerability to adversarial / prompt-injection inputs

## Architecture

```
User Query
   │
   ▼
[Moderation Node] — blocks harmful/illegal/unethical requests
   │
   ▼
[Semantic Entropy Node] — samples 5 responses, clusters by meaning, computes uncertainty
   │
   ▼
[Retriever Node] — fetches supporting evidence from ChromaDB
   │
   ▼
[CrewAI Verification Crew]
   ├── Fact-Checker Agent
   ├── Skeptic Agent
   └── Judge Agent (neutral, multi-perspective on contested topics)
   │
   ▼
[Output Safety Node] — validates against schema, falls back to "insufficient evidence" if needed
   │
   ▼
Final structured response (answer + confidence + sources + full reasoning trace)
```

## Tech Stack

- **Backend:** FastAPI
- **LLM:** Google Gemini (gemini-2.0-flash)
- **Agent orchestration:** LangGraph
- **Multi-agent verification:** CrewAI
- **Vector store:** ChromaDB + sentence-transformers
- **Frontend:** Streamlit
- **Validation:** Pydantic

## Features

- 🧠 Semantic entropy-based uncertainty quantification (not simple LLM self-rating)
- 🔍 Multi-agent cross-verification (fact-checker, skeptic, judge)
- 🛡️ Input moderation + output safety guardrails
- ⚖️ Neutrality enforcement on contested/subjective topics
- 📜 Full reasoning trace logging for transparency and reproducibility
- ⚔️ Live adversarial testing endpoint to demonstrate guardrail robustness

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your GEMINI_API_KEY to .env

# run both servers
fuser -k 8000/tcp 2>/dev/null; fuser -k 8501/tcp 2>/dev/null
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app.py --server.port 8501
```

Or simply:
```bash
chmod +x run.sh
./run.sh
```

Open the Streamlit UI (port 8501) or the FastAPI docs at `/docs` (port 8000).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and index a document |
| POST | `/verify` | Verify a claim, returns full structured response |
| GET | `/trajectory/{query_id}` | Retrieve past reasoning trace |
| POST | `/adversarial-test` | Test guardrails against adversarial input |

## Honest scope

This project **reduces** hallucination and improves transparency — it does not fully **solve** hallucination, which remains an open research problem. Confidence scores are a statistically-grounded signal, not a rigorously calibrated probability. Guardrails handle common adversarial patterns; they are not enterprise-grade adversarial defense.

## Author

Built as a final-year academic project with a placement-portfolio focus.
