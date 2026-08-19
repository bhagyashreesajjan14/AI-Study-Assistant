import sqlite3
import pytest
from unittest.mock import patch, MagicMock
import database
import ai


@pytest.fixture(autouse=True)
def mock_db():
    keep_alive = sqlite3.connect("file::memory:?cache=shared", uri=True)
    def get_test_conn():
        return sqlite3.connect("file::memory:?cache=shared", uri=True)
    with patch("database.get_connection", side_effect=get_test_conn):
        database.init_database()
        yield
    keep_alive.close()


def test_create_and_update_quiz_job():
    job_id = database.create_quiz_job(
        user_id=1,
        subject="dbms",
        topic="Indexing and B-Trees",
        difficulty="Advanced",
        number_of_questions=5,
        source_type="topic"
    )
    assert job_id > 0

    # Check initial pending state
    job = database.get_quiz_job_by_id(job_id, user_id=1)
    assert job is not None
    assert job["status"] == "pending"
    assert job["progress"] == 0

    # Update to processing (50%)
    database.update_quiz_job_status(job_id, status="processing", progress=50)
    job = database.get_quiz_job_by_id(job_id, user_id=1)
    assert job["status"] == "processing"
    assert job["progress"] == 50

    # Complete job with questions
    mock_questions = [
        {
            "question": "What is the primary benefit of a B+ Tree over a B Tree?",
            "options": ["Faster sequential access", "Less memory", "No primary key", "None"],
            "answer": 0,
            "explanation": "Leaves are linked for efficient range scans."
        }
    ]
    database.update_quiz_job_status(
        job_id,
        status="completed",
        progress=100,
        quiz_data={"questions": mock_questions}
    )

    job_done = database.get_quiz_job_by_id(job_id, user_id=1)
    assert job_done["status"] == "completed"
    assert job_done["progress"] == 100
    assert len(job_done["quiz_data"]["questions"]) == 1
    assert job_done["quiz_data"]["questions"][0]["answer"] == 0


def test_failed_quiz_job():
    job_id = database.create_quiz_job(
        user_id=1,
        subject="machine_learning",
        topic="Neural Networks",
        difficulty="Intermediate",
        number_of_questions=5,
        source_type="topic"
    )
    database.update_quiz_job_status(
        job_id,
        status="failed",
        progress=0,
        error_message="Ollama connection timeout"
    )
    job = database.get_quiz_job_by_id(job_id, user_id=1)
    assert job["status"] == "failed"
    assert "timeout" in job["error_message"]


@patch("ai.generate_quiz")
def test_background_quiz_generation_worker(mock_gen_quiz):
    mock_gen_quiz.return_value = {
        "questions": [
            {
                "question": "What is Normalization?",
                "options": ["Data redundancy reduction", "Encryption", "Sorting", "Compilation"],
                "answer": 0,
                "explanation": "Normalization organizes fields and tables to minimize redundancy."
            }
        ]
    }

    job_id = ai.start_background_quiz_generation(
        user_id=1,
        subject="dbms",
        topic="Normalization",
        difficulty="Intermediate",
        number_of_questions=5,
        source_type="topic"
    )
    assert job_id > 0
