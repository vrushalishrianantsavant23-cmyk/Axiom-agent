import uuid
import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_PATH

_client = None
_embedding_fn = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()
    return _embedding_fn


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection("documents", embedding_function=get_embedding_fn())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(chunk_size - overlap, 1)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text
    with open(file_path, "r", errors="ignore") as f:
        return f.read()

def ingest_document(file_path: str, doc_name: str) -> dict:
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return {"status": "error", "message": "No text could be extracted from this document."}

    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return {"status": "success", "chunks_added": len(chunks), "document": doc_name}