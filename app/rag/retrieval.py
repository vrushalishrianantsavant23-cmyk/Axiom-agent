from app.rag.ingestion import get_collection


def retrieve_evidence(query: str, top_k: int = 10) -> list:
    try:
        collection = get_collection()
        if collection.count() == 0:
            return []
        results = collection.query(
            query_texts=[query],
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