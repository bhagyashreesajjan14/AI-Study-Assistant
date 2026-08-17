import json
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import faiss
try:
    import pymupdf as fitz
except ImportError:
    import fitz

import numpy as np
import ollama

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    VECTOR_DIR,
    get_user_vector_dir
)
from database import create_document_job, update_job_status


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text and page numbers from a PDF using PyMuPDF."""
    document = fitz.open(str(pdf_path))
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": page_number,
                "source": Path(pdf_path).name
            })

    document.close()
    return pages


def chunk_text(text: str) -> List[str]:
    """Splits raw text into overlapping token/character chunks."""
    text = text.replace("\n", " ").strip()
    chunks = []
    start = 0
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def create_chunks(
    pdf_path: str,
    subject: str,
    user_id: int = 1,
    document_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extracts text and generates metadata-rich chunks for RAG."""
    pages = extract_pdf_pages(pdf_path)
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "subject": subject,
                "page": page["page"],
                "source": page["source"],
                "user_id": user_id,
                "document_id": document_id
            })

    return chunks


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generates normalized vector embeddings via Ollama embedding model."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = np.array(
        response["embeddings"],
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)
    return embeddings


def build_index(chunks: List[Dict[str, Any]]) -> Optional[faiss.IndexFlatIP]:
    """Builds a normalized FAISS cosine-similarity index from chunk embeddings."""
    if not chunks:
        return None

    texts = [item["text"] for item in chunks]
    embeddings = get_embeddings(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search(
    question: str,
    chunks: List[Dict[str, Any]],
    index: Optional[faiss.IndexFlatIP],
    subject: Optional[str] = None,
    top_k: int = TOP_K,
    user_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Performs isolated vector search for the user and subject."""
    if index is None or not chunks:
        return []

    query_embedding = get_embeddings([question])
    k_val = min(top_k * 3, len(chunks))
    scores, indices = index.search(query_embedding, k_val)

    results = []
    for score, index_number in zip(scores[0], indices[0]):
        if index_number < 0 or index_number >= len(chunks):
            continue

        item = chunks[index_number]

        # Enforce subject filter if specified
        if subject and item.get("subject") != subject:
            continue

        # Enforce user isolation if user_id specified
        if user_id is not None and item.get("user_id") is not None and item.get("user_id") != user_id:
            continue

        results.append({
            **item,
            "score": float(score)
        })

        if len(results) >= top_k:
            break

    return results


def save_index(index, chunks: List[Dict[str, Any]], index_path: Path, metadata_path: Path):
    """Saves FAISS index and JSON metadata persistently."""
    if index is None:
        return

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_index(index_path: Path, metadata_path: Path) -> Tuple[Optional[faiss.IndexFlatIP], List[Dict[str, Any]]]:
    """Loads a FAISS index and metadata chunks from disk."""
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists() or not metadata_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return index, chunks


def get_user_subject_index_paths(user_id: int, subject: str) -> Tuple[Path, Path]:
    """Returns the user-isolated index and metadata file paths."""
    user_vec_dir = get_user_vector_dir(user_id)
    index_path = user_vec_dir / f"{subject}.index"
    metadata_path = user_vec_dir / f"{subject}.json"
    return index_path, metadata_path


def load_user_subject_index(user_id: int, subject: str) -> Tuple[Optional[faiss.IndexFlatIP], List[Dict[str, Any]]]:
    """Loads vector index and metadata for a specific user and subject."""
    if not subject or subject == "General":
        return None, []
    index_path, metadata_path = get_user_subject_index_paths(user_id, subject)
    return load_index(index_path, metadata_path)


def search_user_notes(
    question: str,
    user_id: int,
    document_name: Optional[str] = None,
    subject: Optional[str] = None,
    top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    """
    Directly and persistently searches vector stores isolated for a specific user.
    Can search across all completed notes, or filter to a specific subject or document.
    """
    user_vec_dir = get_user_vector_dir(user_id)
    if not user_vec_dir.exists():
        return []

    # Find index files to query
    if subject and subject not in ("General", "All my completed notes"):
        index_files = [user_vec_dir / f"{subject}.index"]
    else:
        index_files = list(user_vec_dir.glob("*.index"))

    all_results = []
    for idx_file in index_files:
        meta_file = idx_file.with_suffix(".json")
        if not idx_file.exists() or not meta_file.exists():
            continue
        try:
            index, chunks = load_index(idx_file, meta_file)
            if index is None or not chunks:
                continue

            res = search(
                question=question,
                chunks=chunks,
                index=index,
                subject=None,
                top_k=top_k * 3,
                user_id=user_id
            )
            for item in res:
                if document_name and document_name != "All my completed notes":
                    if item.get("source") != document_name and item.get("filename") != document_name:
                        continue
                all_results.append(item)
        except Exception:
            continue

    all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return all_results[:top_k]


# --------------------------------------------------
# BACKGROUND ASYNCHRONOUS PDF WORKER
# --------------------------------------------------

def process_pdf_background(
    job_id: int,
    user_id: int,
    document_id: int,
    file_path: str,
    filename: str,
    subject: str
):
    """
    Background worker executed in a separate daemon thread.
    Extracts PDF text, chunks content, creates Ollama embeddings, builds FAISS index,
    and saves user-isolated persistent vector store while updating persistent job status.
    """
    try:
        # Step 1: Mark processing started (10%)
        update_job_status(job_id, status="processing", progress=10)

        # Step 2: Extract text from PDF (35%)
        pages = extract_pdf_pages(file_path)
        if not pages:
            raise ValueError(f"No extractable text found in PDF: {filename}")
        update_job_status(job_id, status="processing", progress=35)

        # Step 3: Chunk text (50%)
        new_chunks = []
        for page in pages:
            page_chunks = chunk_text(page["text"])
            for chunk in page_chunks:
                new_chunks.append({
                    "text": chunk,
                    "subject": subject,
                    "page": page["page"],
                    "source": filename,
                    "user_id": user_id,
                    "document_id": document_id
                })

        if not new_chunks:
            raise ValueError("Failed to create text chunks from PDF.")
        update_job_status(job_id, status="processing", progress=55)

        # Step 4: Load any existing chunks for this user & subject to merge (65%)
        index_path, metadata_path = get_user_subject_index_paths(user_id, subject)
        _, existing_chunks = load_index(index_path, metadata_path)

        # Remove previous chunks of the same document if re-uploading
        combined_chunks = [c for c in existing_chunks if c.get("source") != filename]
        combined_chunks.extend(new_chunks)

        # Step 5: Compute Ollama embeddings (80%)
        update_job_status(job_id, status="processing", progress=75)
        texts = [item["text"] for item in combined_chunks]
        embeddings = get_embeddings(texts)

        # Step 6: Build FAISS index and save persistently (95%)
        update_job_status(job_id, status="processing", progress=90)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        save_index(index, combined_chunks, index_path, metadata_path)

        # Step 7: Completed (100%)
        update_job_status(job_id, status="completed", progress=100)

    except Exception as e:
        error_msg = str(e)
        update_job_status(job_id, status="failed", progress=0, error_message=error_msg)


def start_background_indexing(
    user_id: int,
    filename: str,
    file_path: str,
    subject: str,
    file_size: int = 0
) -> Tuple[int, int]:
    """
    Creates persistent job & document records and spawns a background thread.
    Returns (document_id, job_id).
    """
    doc_id, job_id = create_document_job(
        user_id=user_id,
        filename=filename,
        file_path=str(file_path),
        subject=subject,
        file_size=file_size
    )

    worker_thread = threading.Thread(
        target=process_pdf_background,
        args=(job_id, user_id, doc_id, str(file_path), filename, subject),
        daemon=True,
        name=f"PDFWorker-Job-{job_id}"
    )
    worker_thread.start()

    return doc_id, job_id