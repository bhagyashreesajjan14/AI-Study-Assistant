import json
from pathlib import Path

import faiss
import fitz
import numpy as np
import ollama

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)


def extract_pdf_pages(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text().strip()

        if text:

            pages.append({
                "text": text,
                "page": page_number,
                "source": Path(pdf_path).name
            })

    document.close()

    return pages


def chunk_text(text):

    text = text.replace(
        "\n",
        " "
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def create_chunks(pdf_path, subject):

    pages = extract_pdf_pages(pdf_path)

    chunks = []

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "subject": subject,
                "page": page["page"],
                "source": page["source"]
            })

    return chunks


def get_embeddings(texts):

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


def build_index(chunks):

    if not chunks:
        return None

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = get_embeddings(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def search(
    question,
    chunks,
    index,
    subject=None,
    top_k=TOP_K
):

    if index is None or not chunks:
        return []

    query_embedding = get_embeddings(
        [question]
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        item = chunks[index_number]

        if (
            subject
            and item["subject"] != subject
        ):
            continue

        results.append({
            **item,
            "score": float(score)
        })

    return results


def save_index(index, chunks, index_path, metadata_path):

    if index is None:
        return

    faiss.write_index(
        index,
        str(index_path)
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_index(index_path, metadata_path):

    if (
        not index_path.exists()
        or not metadata_path.exists()
    ):
        return None, []

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks