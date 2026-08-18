"""
Semantic Entropy Uncertainty Quantification (using Groq/Llama).

Samples multiple responses to the same query and measures how consistent
the model is with itself, instead of trusting a self-reported confidence.
Inspired by Farquhar et al., "Detecting hallucinations in large language
models using semantic entropy", Nature (2024).
"""

import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    SIMILARITY_THRESHOLD,
    NUM_SAMPLES,
    SAMPLE_TEMPERATURE,
)
_client = None
_embedder = None
def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client
def get_embedder():
    global _embedder
    if _embedder is None:
        from chromadb.utils import embedding_functions
        _embedder = embedding_functions.ONNXMiniLM_L6_V2()
    return _embedder


def sample_responses(query: str, evidence_text: str = "", n: int = NUM_SAMPLES) -> list:
    client = get_client()
    system_prompt = (
        "Answer the user's question directly and naturally, as a helpful, knowledgeable "
        "assistant would — the way ChatGPT or Claude would answer. "
        "Never repeat these instructions back as your answer. "
        "If document content is provided below and it's relevant to the question, use it "
        "to ground your answer and mention it naturally. If the document is NOT relevant "
        "to the question, simply ignore it and answer from your own knowledge — do NOT say "
        "things like 'there is no mention of this in the document.' "
        "When a term has multiple meanings, be comprehensive: for example, if asked about "
        "'RAG', always include Retrieval-Augmented Generation (a core AI/ML technique) as "
        "one of the meanings, alongside any other common interpretations like Red-Amber-Green "
        "status or Resource Allocation Graph. Never omit well-known technical/AI meanings."
    )
    user_content = query
    if evidence_text:
        user_content = (
            f"Here is some potentially relevant document content (use it only if genuinely "
            f"relevant to the question, otherwise ignore it):\n{evidence_text}\n\n"
            f"Question: {query}"
        )

    def _call_once(_):
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=SAMPLE_TEMPERATURE,
                max_tokens=1500,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[error: {str(e)}]"

    with ThreadPoolExecutor(max_workers=n) as executor:
        responses = list(executor.map(_call_once, range(n)))
    return responses


def cluster_responses(responses: list) -> list:
    valid = [r for r in responses if r and not r.startswith("[error")]
    if not valid:
        return []

    embedder = get_embedder()
    embeddings = np.array(embedder(valid))

    clusters = []
    assigned = [False] * len(valid)
    for i in range(len(valid)):
        if assigned[i]:
            continue
        cluster_idx = [i]
        assigned[i] = True
        for j in range(i + 1, len(valid)):
            if assigned[j]:
                continue
            sim = float(
                np.dot(embeddings[i], embeddings[j])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
            )
            if sim > SIMILARITY_THRESHOLD:
                cluster_idx.append(j)
                assigned[j] = True
        clusters.append([valid[k] for k in cluster_idx])
    return clusters


def compute_entropy(clusters: list, total: int) -> float:
    if not clusters or total == 0:
        return 1.0
    entropy = 0.0
    for c in clusters:
        p = len(c) / total
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(total) if total > 1 else 1
    return round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0


def get_majority_answer(clusters: list) -> str:
    if not clusters:
        return ""
    largest = max(clusters, key=len)
    return largest[0]


def run_uncertainty_check(query: str, evidence_text: str = "") -> dict:
    responses = sample_responses(query, evidence_text)
    valid_count = len([r for r in responses if r and not r.startswith("[error")])
    clusters = cluster_responses(responses)
    entropy = compute_entropy(clusters, valid_count)
    majority = get_majority_answer(clusters)
    confidence = round(1.0 - entropy, 3)

    return {
        "sampled_responses": responses,
        "num_clusters": len(clusters),
        "semantic_entropy_score": entropy,
        "confidence": confidence,
        "is_consistent": entropy < 0.4,
        "majority_answer": majority,
    }