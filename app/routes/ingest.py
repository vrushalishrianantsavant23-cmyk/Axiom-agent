import os
import tempfile

from fastapi import APIRouter, UploadFile, File

from app.rag.ingestion import ingest_document

router = APIRouter()


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)

    return result
