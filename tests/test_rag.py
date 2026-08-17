import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import rag


def test_chunk_text():
    text = "Sentence one. " * 50
    chunks = rag.chunk_text(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= rag.CHUNK_SIZE + 10


@patch("rag.fitz.open")
def test_extract_pdf_pages(mock_fitz_open):
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 content about DBMS."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 content about SQL."
    
    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz_open.return_value = mock_doc
    
    pages = rag.extract_pdf_pages("dummy.pdf")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[0]["text"] == "Page 1 content about DBMS."
    assert pages[0]["source"] == "dummy.pdf"
    assert pages[1]["page"] == 2


@patch("rag.extract_pdf_pages")
def test_create_chunks(mock_extract):
    mock_extract.return_value = [
        {"text": "Short page text", "page": 1, "source": "test.pdf"}
    ]
    chunks = rag.create_chunks("test.pdf", "dbms", user_id=1, document_id=5)
    assert len(chunks) == 1
    assert chunks[0]["subject"] == "dbms"
    assert chunks[0]["text"] == "Short page text"
    assert chunks[0]["source"] == "test.pdf"
    assert chunks[0]["user_id"] == 1
    assert chunks[0]["document_id"] == 5


@patch("rag.ollama.embed")
def test_get_embeddings(mock_embed):
    mock_embed.return_value = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    emb = rag.get_embeddings(["Sample text"])
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 4)


@patch("rag.get_embeddings")
def test_build_index_and_search(mock_get_embeddings):
    # Mock embeddings of dimension 4
    mock_get_embeddings.side_effect = lambda texts: np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype="float32")
    
    chunks = [
        {"text": "DBMS transactions and ACID", "subject": "dbms", "page": 1, "source": "notes.pdf", "user_id": 1},
        {"text": "Computer networks TCP/IP", "subject": "computer_networks", "page": 2, "source": "cn.pdf", "user_id": 1},
        {"text": "DBMS User 2 notes", "subject": "dbms", "page": 1, "source": "notes2.pdf", "user_id": 2}
    ]
    
    index = rag.build_index(chunks)
    assert index is not None
    assert index.ntotal == 3
    
    # Search isolated to User 1
    results_user1 = rag.search("What is ACID?", chunks, index, subject="dbms", top_k=2, user_id=1)
    assert len(results_user1) == 1
    assert results_user1[0]["user_id"] == 1
    assert "ACID" in results_user1[0]["text"]

    # Search isolated to User 2
    results_user2 = rag.search("What is ACID?", chunks, index, subject="dbms", top_k=2, user_id=2)
    assert len(results_user2) == 1
    assert results_user2[0]["user_id"] == 2


def test_save_and_load_index(tmp_path):
    import faiss
    
    dim = 4
    index = faiss.IndexFlatIP(dim)
    data = np.array([[0.5, 0.5, 0.5, 0.5]], dtype="float32")
    index.add(data)
    
    chunks = [{"text": "Sample text", "subject": "dbms", "page": 1, "source": "dummy.pdf"}]
    
    idx_path = tmp_path / "test.index"
    meta_path = tmp_path / "test.json"
    
    rag.save_index(index, chunks, idx_path, meta_path)
    assert idx_path.exists()
    assert meta_path.exists()
    
    loaded_index, loaded_chunks = rag.load_index(idx_path, meta_path)
    assert loaded_index is not None
    assert loaded_index.ntotal == 1
    assert len(loaded_chunks) == 1
    assert loaded_chunks[0]["text"] == "Sample text"


@patch("rag.update_job_status")
@patch("rag.extract_pdf_pages")
@patch("rag.get_embeddings")
def test_process_pdf_background(mock_get_emb, mock_extract, mock_update_status, tmp_path):
    mock_extract.return_value = [{"text": "Sample DBMS content", "page": 1, "source": "dbms.pdf"}]
    mock_get_emb.side_effect = lambda texts: np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype="float32")

    with patch("rag.get_user_vector_dir", return_value=tmp_path):
        rag.process_pdf_background(
            job_id=101,
            user_id=1,
            document_id=1,
            file_path="dummy.pdf",
            filename="dbms.pdf",
            subject="dbms"
        )

    # Check status updates
    calls = [c[1]["status"] for c in mock_update_status.call_args_list]
    assert "processing" in calls
    assert "completed" in calls


@patch("rag.get_embeddings")
def test_search_user_notes_multi_document(mock_get_emb, tmp_path):
    def fake_embeddings(texts):
        embs = []
        for t in texts:
            if "ACID" in t:
                embs.append([1.0, 0.0, 0.0, 0.0])
            else:
                embs.append([0.0, 1.0, 0.0, 0.0])
        return np.array(embs, dtype="float32")

    mock_get_emb.side_effect = fake_embeddings

    # Build user index
    chunks = [
        {"text": "DBMS Transaction ACID", "subject": "dbms", "page": 5, "source": "DBMS 4th Sem m1.pdf", "user_id": 1},
        {"text": "OS Virtual Memory", "subject": "operating_systems", "page": 10, "source": "OS Notes.pdf", "user_id": 1}
    ]
    index = rag.build_index(chunks)
    rag.save_index(index, chunks, tmp_path / "dbms.index", tmp_path / "dbms.json")

    with patch("rag.get_user_vector_dir", return_value=tmp_path):
        # Search all documents for user 1 with query ACID
        res = rag.search_user_notes("ACID", user_id=1)
        assert len(res) >= 1
        assert res[0]["source"] == "DBMS 4th Sem m1.pdf"

        # Search filtered by document name
        res_doc = rag.search_user_notes("Memory", user_id=1, document_name="OS Notes.pdf")
        assert len(res_doc) == 1
        assert res_doc[0]["source"] == "OS Notes.pdf"


