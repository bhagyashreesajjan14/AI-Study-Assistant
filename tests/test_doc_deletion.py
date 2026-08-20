import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import database
import rag


def test_delete_user_document_in_db():
    # Setup test database records
    user_id = 999
    doc_id, job_id = database.create_document_job(
        user_id=user_id,
        filename="test_file_delete.pdf",
        file_path="dummy_path.pdf",
        subject="dbms",
        file_size=1024,
        file_type="pdf"
    )

    # Verify created
    doc = database.get_document_by_id(doc_id, user_id=user_id)
    assert doc is not None
    assert doc["filename"] == "test_file_delete.pdf"

    # User isolation: User 888 cannot delete User 999's doc
    res_wrong_user = database.delete_user_document(doc_id, user_id=888)
    assert res_wrong_user is False
    assert database.get_document_by_id(doc_id, user_id=user_id) is not None

    # Delete with correct user
    res_correct = database.delete_user_document(doc_id, user_id=user_id)
    assert res_correct is True

    # Verify removed from database and jobs
    assert database.get_document_by_id(doc_id, user_id=user_id) is None
    jobs = database.get_user_document_jobs(user_id)
    assert not any(j["id"] == job_id or j["document_id"] == doc_id for j in jobs)


@patch("rag.get_embeddings")
def test_delete_document_data_rag_cleanup(mock_get_emb, tmp_path):
    # Mock embeddings matching input length
    mock_get_emb.side_effect = lambda texts, **kwargs: np.array([[0.1 * (i + 1), 0.2, 0.3, 0.4] for i in range(len(texts))], dtype="float32")

    user_id = 777
    user_vec_dir = tmp_path / "vec"
    user_notes_dir = tmp_path / "notes"
    user_vec_dir.mkdir(parents=True, exist_ok=True)
    user_notes_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy physical file
    test_pdf = user_notes_dir / "sample_doc1.pdf"
    test_pdf.write_text("sample content")

    # Create dummy index and chunks with 2 documents
    chunks = [
        {
            "text": "Doc 1 content about ACID properties",
            "subject": "dbms",
            "page": 1,
            "source": "sample_doc1.pdf",
            "user_id": user_id,
            "document_id": 101
        },
        {
            "text": "Doc 2 content about Normalization",
            "subject": "dbms",
            "page": 1,
            "source": "sample_doc2.pdf",
            "user_id": user_id,
            "document_id": 102
        }
    ]

    index = rag.build_index(chunks)
    idx_path = user_vec_dir / "dbms.index"
    meta_path = user_vec_dir / "dbms.json"
    rag.save_index(index, chunks, idx_path, meta_path)

    # Insert DB record for doc 101
    d_id, _ = database.create_document_job(
        user_id=user_id,
        filename="sample_doc1.pdf",
        file_path=str(test_pdf),
        subject="dbms",
        file_size=500
    )

    with patch("rag.get_user_vector_dir", return_value=user_vec_dir), \
         patch("rag.get_user_notes_dir", return_value=user_notes_dir), \
         patch("rag.get_user_subject_index_paths", return_value=(idx_path, meta_path)):

        # Delete doc 101
        success, msg = rag.delete_document_data(
            user_id=user_id,
            document_id=101,
            filename="sample_doc1.pdf",
            subject="dbms",
            file_path=str(test_pdf)
        )

        assert success is True
        # Verify physical file deleted
        assert not test_pdf.exists()

        # Verify index still exists but only contains doc 102
        loaded_idx, loaded_chunks = rag.load_index(idx_path, meta_path)
        assert len(loaded_chunks) == 1
        assert loaded_chunks[0]["source"] == "sample_doc2.pdf"
        assert loaded_chunks[0]["document_id"] == 102

        # Delete doc 102 (last document in subject)
        success2, msg2 = rag.delete_document_data(
            user_id=user_id,
            document_id=102,
            filename="sample_doc2.pdf",
            subject="dbms"
        )
        assert success2 is True
        # Both index and meta files should be cleaned up
        assert not idx_path.exists()
        assert not meta_path.exists()
