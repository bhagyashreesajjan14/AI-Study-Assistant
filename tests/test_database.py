import sqlite3
import pytest
from unittest.mock import patch
import database


@pytest.fixture(autouse=True)
def mock_db():
    # Use a shared in-memory SQLite database connection for tests
    keep_alive = sqlite3.connect("file::memory:?cache=shared", uri=True)
    
    def get_test_conn():
        return sqlite3.connect("file::memory:?cache=shared", uri=True)
        
    with patch("database.get_connection", side_effect=get_test_conn):
        database.init_database()
        yield
    keep_alive.close()


def test_student_profile_isolation():
    database.save_profile("John Doe", "CS", 3, user_id=1)
    database.save_profile("Jane Doe", "IT", 4, user_id=2)
    
    profile1 = database.get_profile(user_id=1)
    assert profile1 == ("John Doe", "CS", 3)
    
    profile2 = database.get_profile(user_id=2)
    assert profile2 == ("Jane Doe", "IT", 4)


def test_auth_registration_and_login():
    # Test successful registration
    ok, msg, user = database.register_user("student1", "secret123", "Alice Smith", "Computer Science", 4)
    assert ok is True
    assert user["username"] == "student1"
    assert user["name"] == "Alice Smith"

    # Test duplicate registration
    ok_dup, msg_dup, _ = database.register_user("student1", "secret123", "Alice 2")
    assert ok_dup is False
    assert "already exists" in msg_dup

    # Test correct login
    auth_ok, auth_msg, auth_user = database.authenticate_user("student1", "secret123")
    assert auth_ok is True
    assert auth_user["name"] == "Alice Smith"

    # Test wrong password
    bad_ok, bad_msg, _ = database.authenticate_user("student1", "wrongpassword")
    assert bad_ok is False


def test_dynamic_subjects():
    # Verify default subjects
    subjects = database.get_all_subjects()
    assert "dbms" in subjects
    assert "operating_systems" in subjects

    # Add custom manual subject
    ok, msg = database.add_subject("Cloud Computing")
    assert ok is True
    
    updated_subjects = database.get_all_subjects()
    assert "cloud_computing" in updated_subjects


def test_chat_sessions_user_isolation():
    # User 1 session
    session_id_1 = database.create_chat_session("dbms", "User 1 Chat", user_id=1)
    assert session_id_1 > 0
    
    # User 2 session
    session_id_2 = database.create_chat_session("dbms", "User 2 Chat", user_id=2)
    assert session_id_2 > 0
    
    # User 1 should only see User 1's chat
    sessions_user1 = database.get_chat_sessions("dbms", user_id=1)
    assert len(sessions_user1) == 1
    assert sessions_user1[0]["title"] == "User 1 Chat"
    
    # User 2 should only see User 2's chat
    sessions_user2 = database.get_chat_sessions("dbms", user_id=2)
    assert len(sessions_user2) == 1
    assert sessions_user2[0]["title"] == "User 2 Chat"
    
    # Test renaming and deleting by user
    database.rename_chat_session(session_id_1, "User 1 Renamed", user_id=1)
    assert database.get_chat_sessions("dbms", user_id=1)[0]["title"] == "User 1 Renamed"
    
    database.delete_chat_session(session_id_1, user_id=1)
    assert len(database.get_chat_sessions("dbms", user_id=1)) == 0
    assert len(database.get_chat_sessions("dbms", user_id=2)) == 1


def test_chat_messages():
    session_id = database.create_chat_session("operating_systems", "OS Chat", user_id=1)
    
    # Test saving message
    database.save_chat_message(session_id, "user", "What is virtual memory?", user_id=1)
    sources = [{"source": "os_notes.pdf", "page": 4, "text": "Virtual memory is...", "score": 0.95}]
    database.save_chat_message(session_id, "assistant", "Virtual memory is a memory management technique.", sources=sources, user_id=1)
    
    # Test retrieving
    messages = database.get_chat_messages(session_id, user_id=1)
    assert len(messages) == 2
    
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is virtual memory?"
    assert messages[0]["sources"] == []
    
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Virtual memory is a memory management technique."
    assert messages[1]["sources"] == sources


def test_document_jobs_lifecycle():
    doc_id, job_id = database.create_document_job(
        user_id=1,
        filename="lecture_dbms.pdf",
        file_path="/tmp/lecture_dbms.pdf",
        subject="dbms",
        file_size=1024
    )
    assert doc_id > 0
    assert job_id > 0
    
    job = database.get_job_by_id(job_id)
    assert job["status"] == "queued"
    assert job["progress"] == 0
    assert job["user_id"] == 1
    
    # Update progress
    database.update_job_status(job_id, status="processing", progress=45)
    job = database.get_job_by_id(job_id)
    assert job["status"] == "processing"
    assert job["progress"] == 45
    assert job["started_at"] is not None
    
    # Complete job
    database.update_job_status(job_id, status="completed", progress=100)
    job = database.get_job_by_id(job_id)
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["completed_at"] is not None
    
    # Check user documents
    docs = database.get_user_documents(user_id=1, subject="dbms")
    assert len(docs) == 1
    assert docs[0]["filename"] == "lecture_dbms.pdf"
    assert docs[0]["status"] == "completed"


def test_quiz_user_isolation():
    answers = [{"question": "Q1", "selected_answer": "A", "correct_answer": "A", "is_correct": 1}]
    database.save_quiz_result("dbms", "Transactions", "Easy", 100.0, 1, 1, answers, user_id=1)
    database.save_quiz_result("dbms", "Normalization", "Hard", 50.0, 2, 1, answers, user_id=2)
    
    attempts_u1 = database.get_quiz_attempts(user_id=1)
    attempts_u2 = database.get_quiz_attempts(user_id=2)
    
    assert len(attempts_u1) == 1
    assert attempts_u1[0][1] == "Transactions"
    
    assert len(attempts_u2) == 1
    assert attempts_u2[0][1] == "Normalization"


def test_study_plans_user_isolation():
    plan_id_1 = database.save_study_plan(
        user_id=1,
        topic="Transactions",
        target_level="Intermediate",
        schedule_summary="Monday: 2 hours 30 minutes",
        plan_content="# Day 1 Plan"
    )
    assert plan_id_1 > 0
    
    database.save_study_plan(
        user_id=2,
        topic="Operating Systems",
        target_level="Advanced",
        schedule_summary="Tuesday: 1 hour",
        plan_content="# OS Plan"
    )
    
    plans_u1 = database.get_user_study_plans(user_id=1)
    plans_u2 = database.get_user_study_plans(user_id=2)
    
    assert len(plans_u1) == 1
    assert plans_u1[0]["topic"] == "Transactions"
    
    assert len(plans_u2) == 1
    assert plans_u2[0]["topic"] == "Operating Systems"
    
    latest_u1 = database.get_latest_study_plan(user_id=1)
    assert latest_u1["target_level"] == "Intermediate"


def test_document_extracted_text_caching_and_retrieval():
    doc_id, job_id = database.create_document_job(
        user_id=1,
        filename="lecture_slide.png",
        file_path="/path/to/lecture_slide.png",
        subject="machine_learning",
        file_size=10240,
        file_type="png"
    )
    assert doc_id > 0
    assert job_id > 0

    # Save extracted text
    sample_text = "Supervised learning uses labeled training datasets."
    database.save_document_extracted_text(doc_id, sample_text)

    # Mark as completed
    database.update_job_status(job_id, status="completed", progress=100)

    # Fetch by ID with user isolation
    doc = database.get_document_by_id(doc_id, user_id=1)
    assert doc is not None
    assert doc["filename"] == "lecture_slide.png"
    assert doc["file_type"] == "png"
    assert doc["extracted_text"] == sample_text

    # Another user should not be able to fetch user 1's document
    doc_user2 = database.get_document_by_id(doc_id, user_id=2)
    assert doc_user2 is None

    # Test get_latest_user_document
    latest = database.get_latest_user_document(user_id=1)
    assert latest is not None
    assert latest["id"] == doc_id


def test_full_user_flow_and_relogin(tmp_path):
    import rag
    import ai

    # Step 1: Register user with exact registered name
    reg_name = "Dr. Evelyn Reed"
    ok, msg, user_data = database.register_user("evelyn", "securePass123", reg_name, "Data Science", 3)
    assert ok is True
    assert user_data["name"] == reg_name

    # Step 2: Authenticate and retrieve profile
    auth_ok, _, auth_data = database.authenticate_user("evelyn", "securePass123")
    assert auth_ok is True
    assert auth_data["name"] == reg_name

    profile = database.get_profile(user_id=auth_data["id"])
    assert profile[0] == reg_name
    assert profile[1] == "Data Science"
    assert profile[2] == 3

    # Step 3: Multi-format file creation (TXT, MD)
    notes_file = tmp_path / "Decision_Trees.txt"
    notes_file.write_text("Decision trees recursively split dataset based on Gini impurity and information gain.", encoding="utf-8")

    # Step 4: Create document job and extract text
    doc_id, job_id = database.create_document_job(
        user_id=auth_data["id"],
        filename="Decision_Trees.txt",
        file_path=str(notes_file),
        subject="machine_learning",
        file_size=len(notes_file.read_bytes()),
        file_type="txt"
    )
    assert doc_id > 0
    assert job_id > 0

    extracted_content = rag.get_full_document_text(str(notes_file))
    assert "Decision trees" in extracted_content
    database.save_document_extracted_text(doc_id, extracted_content)
    database.update_job_status(job_id, status="completed", progress=100)

    # Step 5: Verify document is completed and available for Evelyn
    docs = database.get_user_completed_documents(user_id=auth_data["id"])
    assert len(docs) == 1
    assert docs[0]["filename"] == "Decision_Trees.txt"
    assert docs[0]["extracted_text"] == extracted_content

    # Step 6: Verify User 2 cannot see Evelyn's document
    docs_user2 = database.get_user_completed_documents(user_id=999)
    assert len(docs_user2) == 0

    # Step 7: Reuse the same extracted content across features
    # Feature 7a: Quiz from material
    with patch("ai.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {
                "content": '{"questions": [{"question": "What metric is used for tree splitting?", "options": ["Gini impurity", "Accuracy", "Recall", "F1"], "answer": 0, "explanation": "Gini is used."}]}'
            }
        }
        quiz = ai.generate_quiz_from_material(docs[0]["extracted_text"], "machine_learning", "Intermediate", 1)
        assert len(quiz["questions"]) == 1
        assert quiz["questions"][0]["options"][0] == "Gini impurity"

    # Feature 7b: Summary from the same material
    with patch("ai.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {"content": "**Core Summary**\nDecision trees split based on information gain."}
        }
        summary = ai.generate_summary(docs[0]["extracted_text"])
        assert "Decision trees" in summary

    # Feature 7c: Flashcards from the same material
    with patch("ai.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {
                "content": '{"flashcards": [{"front": "Gini Impurity", "back": "A measure of purity for splits", "tag": "ML"}]}'
            }
        }
        fcs = ai.generate_flashcards(docs[0]["extracted_text"], 1)
        assert len(fcs) == 1
        assert fcs[0]["front"] == "Gini Impurity"

    # Step 8: Save quiz results and verify performance
    database.save_quiz_result(
        subject="machine_learning",
        topic="Material: Decision_Trees.txt",
        difficulty="Intermediate",
        score=100.0,
        total_questions=1,
        correct_answers=1,
        answers=[{"question": "Q1", "selected_answer": "Gini", "correct_answer": "Gini", "is_correct": 1}],
        user_id=auth_data["id"]
    )

    perf = database.get_topic_performance(user_id=auth_data["id"])
    assert len(perf) == 1
    assert perf[0][0] == "machine_learning"
    assert perf[0][1] == "Material: Decision_Trees.txt"

    # Step 9: Re-login simulation - retrieve profile and documents
    re_auth_ok, _, re_user = database.authenticate_user("evelyn", "securePass123")
    assert re_auth_ok is True
    re_profile = database.get_profile(user_id=re_user["id"])
    assert re_profile[0] == reg_name
    re_docs = database.get_user_completed_documents(user_id=re_user["id"])
    assert len(re_docs) == 1
    assert re_docs[0]["filename"] == "Decision_Trees.txt"

