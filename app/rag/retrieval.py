from app.rag.ingestion import get_chroma_client, get_embedder


def retrieve_evidence(query: str, top_k: int = 10) -> list:
    """Fetch top-k supporting evidence chunks for a query. Returns [] gracefully
    if no documents have been ingested yet."""
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection("documents")
        if collection.count() == 0:
            return []
        embedder = get_embedder()
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        evidence = [{"text": doc, "source": meta.get("source", "unknown")} for doc, meta in zip(docs, metas)]

        seen = set()
        unique_evidence = []
        for e in evidence:
            if e["source"] not in seen:
                seen.add(e["source"])
                unique_evidence.append(e)
        return unique_evidence
    except Exception:
        return []