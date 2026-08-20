import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from config import DB_PATH, DEFAULT_SUBJECTS


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = "ai_study_assistant_salt_"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _migrate_table_add_column(cursor, table_name: str, column_name: str, column_def: str):
    """Safely adds a column to an existing table if it does not exist already."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Users Table for Login & Signup
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            course TEXT,
            semester INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    # Dynamic Subjects Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Seed default subjects if empty
    for subj in DEFAULT_SUBJECTS:
        cursor.execute("""
            INSERT OR IGNORE INTO subjects (name, created_at)
            VALUES (?, ?)
        """, (subj, datetime.now().isoformat()))

    # Student Profile Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            name TEXT NOT NULL,
            course TEXT,
            semester INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Documents Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Document Processing Jobs Table (Persistent Job Status)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            progress INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)

    # Quiz Attempts Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT,
            score REAL NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Quiz Answers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            attempt_id INTEGER,
            question TEXT,
            selected_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER,
            topic TEXT,
            created_at TEXT,
            FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE
        )
    """)

    # Chat Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            title TEXT NOT NULL,
            subject TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Chat Messages Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
    """)

    # Personalized Study Plans Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            target_level TEXT NOT NULL,
            schedule_summary TEXT NOT NULL,
            plan_content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Multiple Notes Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT 'General',
            associated_files TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Persistent Background Quiz Jobs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            number_of_questions INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            source_id INTEGER,
            source_name TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            progress INTEGER NOT NULL DEFAULT 0,
            quiz_data TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------
    # SAFE SCHEMA MIGRATIONS FOR EXISTING DATABASES
    # --------------------------------------------------
    _migrate_table_add_column(cursor, "student_profile", "user_id", "INTEGER DEFAULT 1")
    _migrate_table_add_column(cursor, "quiz_attempts", "user_id", "INTEGER DEFAULT 1")
    _migrate_table_add_column(cursor, "quiz_answers", "user_id", "INTEGER DEFAULT 1")
    _migrate_table_add_column(cursor, "chat_sessions", "user_id", "INTEGER DEFAULT 1")
    _migrate_table_add_column(cursor, "chat_messages", "user_id", "INTEGER DEFAULT 1")
    _migrate_table_add_column(cursor, "documents", "file_type", "TEXT DEFAULT 'pdf'")
    _migrate_table_add_column(cursor, "documents", "extracted_text", "TEXT DEFAULT ''")

    # Set default user_id = 1 for any legacy records where user_id is null
    for table in ["student_profile", "quiz_attempts", "quiz_answers", "chat_sessions", "chat_messages"]:
        try:
            cursor.execute(f"UPDATE {table} SET user_id = 1 WHERE user_id IS NULL")
        except Exception:
            pass

    conn.commit()
    conn.close()


# --------------------------------------------------
# AUTHENTICATION & USER MANAGEMENT
# --------------------------------------------------

def register_user(username: str, password: str, name: str, course: str = "", semester: int = 1) -> Tuple[bool, str, Optional[Dict]]:
    """Registers a new student user with isolated profile."""
    username = username.strip().lower()
    if not username:
        return False, "Username cannot be empty.", None
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters.", None
    if not name.strip():
        return False, "Name cannot be empty.", None

    conn = get_connection()
    cursor = conn.cursor()

    try:
        pw_hash = hash_password(password)
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO users (username, password_hash, name, course, semester, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, pw_hash, name.strip(), course.strip(), int(semester), timestamp))
        user_id = cursor.lastrowid

        # Create isolated student profile for this user
        cursor.execute("""
            INSERT OR REPLACE INTO student_profile (user_id, name, course, semester)
            VALUES (?, ?, ?, ?)
        """, (user_id, name.strip(), course.strip(), int(semester)))

        conn.commit()

        user_data = {
            "id": user_id,
            "username": username,
            "name": name.strip(),
            "course": course.strip(),
            "semester": int(semester)
        }
        return True, "Account created successfully!", user_data
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another or log in.", None
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    """Authenticates student user credentials."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Please enter both username and password.", None

    conn = get_connection()
    cursor = conn.cursor()

    pw_hash = hash_password(password)
    cursor.execute("""
        SELECT id, username, name, course, semester
        FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, pw_hash))

    row = cursor.fetchone()
    conn.close()

    if row:
        user_data = {
            "id": row[0],
            "username": row[1],
            "name": row[2],
            "course": row[3],
            "semester": row[4]
        }
        return True, "Login successful!", user_data
    else:
        return False, "Invalid username or password.", None


# --------------------------------------------------
# SUBJECT MANAGEMENT
# --------------------------------------------------

def get_all_subjects() -> List[str]:
    """Returns all available subjects including defaults and custom added ones."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM subjects ORDER BY name ASC")
    rows = cursor.fetchall()
    conn.close()

    subjects = [r[0] for r in rows] if rows else list(DEFAULT_SUBJECTS)
    for s in DEFAULT_SUBJECTS:
        if s not in subjects:
            subjects.append(s)
    return subjects


def add_subject(subject_name: str) -> Tuple[bool, str]:
    """Adds a new subject manually."""
    clean_name = subject_name.strip().lower().replace(" ", "_")
    if not clean_name:
        return False, "Subject name cannot be empty."

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO subjects (name, created_at)
            VALUES (?, ?)
        """, (clean_name, datetime.now().isoformat()))
        conn.commit()
        return True, f"Subject '{clean_name}' added successfully."
    except sqlite3.IntegrityError:
        return True, f"Subject '{clean_name}' already exists."
    finally:
        conn.close()


# --------------------------------------------------
# USER-ISOLATED PROFILE
# --------------------------------------------------

def save_profile(name: str, course: str, semester: int, user_id: int = 1):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM student_profile WHERE user_id = ?", (user_id,))
    cursor.execute("""
        INSERT INTO student_profile (user_id, name, course, semester)
        VALUES (?, ?, ?, ?)
    """, (user_id, name, course, semester))

    # Also update in users table
    cursor.execute("""
        UPDATE users
        SET name = ?, course = ?, semester = ?
        WHERE id = ?
    """, (name, course, semester, user_id))

    conn.commit()
    conn.close()


def get_profile(user_id: int = 1) -> Optional[Tuple[str, str, int]]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, course, semester
        FROM student_profile
        WHERE user_id = ?
        LIMIT 1
    """, (user_id,))

    result = cursor.fetchone()
    if not result:
        cursor.execute("""
            SELECT name, course, semester
            FROM users
            WHERE id = ?
            LIMIT 1
        """, (user_id,))
        result = cursor.fetchone()

    conn.close()
    return result


# --------------------------------------------------
# DOCUMENTS & BACKGROUND JOBS
# --------------------------------------------------

def create_document_job(
    user_id: int,
    filename: str,
    file_path: str,
    subject: str,
    file_size: int = 0,
    file_type: str = "pdf"
) -> Tuple[int, int]:
    """Creates persistent document and background job records."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO documents (user_id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at)
        VALUES (?, ?, ?, ?, 'processing', ?, ?, '', ?)
    """, (user_id, filename, file_path, subject, file_size, file_type, now))
    doc_id = cursor.lastrowid

    cursor.execute("""
        INSERT INTO document_jobs (user_id, document_id, filename, file_path, subject, status, progress, error_message, created_at)
        VALUES (?, ?, ?, ?, ?, 'queued', 0, NULL, ?)
    """, (user_id, doc_id, filename, file_path, subject, now))
    job_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return doc_id, job_id


def save_document_extracted_text(doc_id: int, extracted_text: str):
    """Saves cached extracted text for a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE documents
        SET extracted_text = ?
        WHERE id = ?
    """, (extracted_text, doc_id))
    conn.commit()
    conn.close()


def update_job_status(job_id: int, status: str, progress: Optional[int] = None, error_message: Optional[str] = None):
    """Updates background job progress and status persistently."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    updates = ["status = ?"]
    params = [status]

    if progress is not None:
        updates.append("progress = ?")
        params.append(progress)

    if status == "processing":
        updates.append("started_at = COALESCE(started_at, ?)")
        params.append(now)
    elif status in ("completed", "failed"):
        updates.append("completed_at = ?")
        params.append(now)

    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    params.append(job_id)
    query = f"UPDATE document_jobs SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)

    # Sync document status as well
    cursor.execute("SELECT document_id FROM document_jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    if row and row[0]:
        cursor.execute("UPDATE documents SET status = ? WHERE id = ?", (status, row[0]))

    conn.commit()
    conn.close()


def get_user_document_jobs(user_id: int) -> List[Dict[str, Any]]:
    """Returns all document processing jobs for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, document_id, filename, file_path, subject, status, progress, error_message, created_at, started_at, completed_at
        FROM document_jobs
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for r in rows:
        jobs.append({
            "id": r[0],
            "document_id": r[1],
            "filename": r[2],
            "file_path": r[3],
            "subject": r[4],
            "status": r[5],
            "progress": r[6],
            "error_message": r[7],
            "created_at": r[8],
            "started_at": r[9],
            "completed_at": r[10]
        })
    return jobs


def get_user_documents(user_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns all uploaded documents isolated to a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    if subject:
        cursor.execute("""
            SELECT id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ? AND subject = ?
            ORDER BY created_at DESC
        """, (user_id, subject))
    else:
        cursor.execute("""
            SELECT id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for r in rows:
        docs.append({
            "id": r[0],
            "filename": r[1],
            "file_path": r[2],
            "subject": r[3],
            "status": r[4],
            "file_size": r[5],
            "file_type": r[6] if len(r) > 6 else "pdf",
            "extracted_text": r[7] if len(r) > 7 else "",
            "created_at": r[8] if len(r) > 8 else r[6]
        })
    return docs


def get_user_completed_documents(user_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns only successfully completed documents for the user."""
    conn = get_connection()
    cursor = conn.cursor()
    if subject and subject not in ("General", "All my completed notes"):
        cursor.execute("""
            SELECT id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ? AND subject = ? AND status = 'completed'
            ORDER BY created_at DESC
        """, (user_id, subject))
    else:
        cursor.execute("""
            SELECT id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ? AND status = 'completed'
            ORDER BY created_at DESC
        """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    return [{
        "id": r[0],
        "filename": r[1],
        "file_path": r[2],
        "subject": r[3],
        "status": r[4],
        "file_size": r[5],
        "file_type": r[6] if len(r) > 6 else "pdf",
        "extracted_text": r[7] if len(r) > 7 else "",
        "created_at": r[8] if len(r) > 8 else r[6]
    } for r in rows]


def get_document_by_id(doc_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Fetches a specific document record by ID, optionally isolated by user_id."""
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("""
            SELECT id, user_id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE id = ? AND user_id = ?
        """, (doc_id, user_id))
    else:
        cursor.execute("""
            SELECT id, user_id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE id = ?
        """, (doc_id,))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "user_id": r[1],
        "filename": r[2],
        "file_path": r[3],
        "subject": r[4],
        "status": r[5],
        "file_size": r[6],
        "file_type": r[7] if len(r) > 7 else "pdf",
        "extracted_text": r[8] if len(r) > 8 else "",
        "created_at": r[9] if len(r) > 9 else ""
    }


def get_latest_user_document(user_id: int, subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches the latest document uploaded by the user."""
    conn = get_connection()
    cursor = conn.cursor()
    if subject and subject not in ("General", "All my completed notes"):
        cursor.execute("""
            SELECT id, user_id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ? AND subject = ?
            ORDER BY id DESC
            LIMIT 1
        """, (user_id, subject))
    else:
        cursor.execute("""
            SELECT id, user_id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "user_id": r[1],
        "filename": r[2],
        "file_path": r[3],
        "subject": r[4],
        "status": r[5],
        "file_size": r[6],
        "file_type": r[7] if len(r) > 7 else "pdf",
        "extracted_text": r[8] if len(r) > 8 else "",
        "created_at": r[9] if len(r) > 9 else ""
    }


def check_duplicate_document(user_id: int, filename: str, subject: str) -> Optional[Dict[str, Any]]:
    """Checks if a document with the same filename already exists for that user and subject."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, file_path, subject, status, file_size, file_type, extracted_text, created_at
        FROM documents
        WHERE user_id = ? AND filename = ? AND subject = ?
        ORDER BY id DESC
        LIMIT 1
    """, (user_id, filename, subject))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "filename": r[1],
        "file_path": r[2],
        "subject": r[3],
        "status": r[4],
        "file_size": r[5],
        "file_type": r[6] if len(r) > 6 else "pdf",
        "extracted_text": r[7] if len(r) > 7 else "",
        "created_at": r[8] if len(r) > 8 else ""
    }


def get_job_by_id(job_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a specific job by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, document_id, filename, file_path, subject, status, progress, error_message, created_at, started_at, completed_at
        FROM document_jobs
        WHERE id = ?
    """, (job_id,))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "user_id": r[1],
        "document_id": r[2],
        "filename": r[3],
        "file_path": r[4],
        "subject": r[5],
        "status": r[6],
        "progress": r[7],
        "error_message": r[8],
        "created_at": r[9],
        "started_at": r[10],
        "completed_at": r[11]
    }


def delete_user_document(document_id: int, user_id: int) -> bool:
    """Safely deletes a document and associated job records for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM document_jobs WHERE (document_id = ? OR filename IN (SELECT filename FROM documents WHERE id = ?)) AND user_id = ?", (document_id, document_id, user_id))
        cursor.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count > 0
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


# --------------------------------------------------
# USER-ISOLATED QUIZ RESULTS & ANALYTICS
# --------------------------------------------------

def save_quiz_result(
    subject: str,
    topic: str,
    difficulty: str,
    score: float,
    total_questions: int,
    correct_answers: int,
    answers: List[Dict],
    user_id: int = 1
) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO quiz_attempts
        (
            user_id,
            subject,
            topic,
            difficulty,
            score,
            total_questions,
            correct_answers,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        subject,
        topic,
        difficulty,
        score,
        total_questions,
        correct_answers,
        timestamp
    ))

    attempt_id = cursor.lastrowid

    for answer in answers:
        cursor.execute("""
            INSERT INTO quiz_answers
            (
                user_id,
                attempt_id,
                question,
                selected_answer,
                correct_answer,
                is_correct,
                topic,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            attempt_id,
            answer["question"],
            answer["selected_answer"],
            answer["correct_answer"],
            int(answer["is_correct"]),
            topic,
            timestamp
        ))

    conn.commit()
    conn.close()
    return attempt_id


def get_quiz_attempts(user_id: int = 1) -> List[Tuple]:
    """Retrieves quiz attempts isolated to a specific user."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            subject,
            topic,
            difficulty,
            score,
            total_questions,
            correct_answers,
            created_at
        FROM quiz_attempts
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))

    results = cursor.fetchall()
    conn.close()
    return results


def get_topic_performance(user_id: int = 1) -> List[Tuple]:
    """Calculates topic performance metrics isolated to a specific user."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            subject,
            topic,
            AVG(score) AS average_score,
            COUNT(*) AS attempts
        FROM quiz_attempts
        WHERE user_id = ?
        GROUP BY subject, topic
        ORDER BY average_score ASC
    """, (user_id,))

    results = cursor.fetchall()
    conn.close()
    return results


def get_recent_topic_score(subject: str, topic: str, user_id: int = 1) -> Optional[float]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT score
        FROM quiz_attempts
        WHERE user_id = ?
        AND subject = ?
        AND topic = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id, subject, topic))

    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# --------------------------------------------------
# USER-ISOLATED CHAT SESSIONS & MESSAGES
# --------------------------------------------------

def create_chat_session(subject: str, title: str = "New Chat", user_id: int = 1) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO chat_sessions (user_id, title, subject, created_at)
        VALUES (?, ?, ?, ?)
    """, (user_id, title, subject, timestamp))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_chat_sessions(subject: Optional[str] = None, user_id: int = 1) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if subject:
        cursor.execute("""
            SELECT id, title, subject, created_at
            FROM chat_sessions
            WHERE user_id = ? AND subject = ?
            ORDER BY created_at DESC
        """, (user_id, subject))
    else:
        cursor.execute("""
            SELECT id, title, subject, created_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
    results = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "subject": r[2], "created_at": r[3]} for r in results]


def get_chat_messages(session_id: int, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("""
            SELECT m.role, m.content, m.sources, m.created_at
            FROM chat_messages m
            JOIN chat_sessions s ON m.session_id = s.id
            WHERE m.session_id = ? AND s.user_id = ?
            ORDER BY m.id ASC
        """, (session_id, user_id))
    else:
        cursor.execute("""
            SELECT role, content, sources, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
    results = cursor.fetchall()
    conn.close()

    messages = []
    for r in results:
        sources_list = []
        if r[2]:
            try:
                sources_list = json.loads(r[2])
            except Exception:
                sources_list = []
        messages.append({
            "role": r[0],
            "content": r[1],
            "sources": sources_list,
            "created_at": r[3]
        })
    return messages


def save_chat_message(session_id: int, role: str, content: str, sources: Optional[List] = None, user_id: int = 1):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    sources_str = json.dumps(sources) if sources else None
    cursor.execute("""
        INSERT INTO chat_messages (user_id, session_id, role, content, sources, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, session_id, role, content, sources_str, timestamp))
    conn.commit()
    conn.close()


def delete_chat_session(session_id: int, user_id: Optional[int] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ? AND session_id IN (SELECT id FROM chat_sessions WHERE user_id = ?)", (session_id, user_id))
        cursor.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    else:
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def rename_chat_session(session_id: int, title: str, user_id: Optional[int] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("""
            UPDATE chat_sessions
            SET title = ?
            WHERE id = ? AND user_id = ?
        """, (title, session_id, user_id))
    else:
        cursor.execute("""
            UPDATE chat_sessions
            SET title = ?
            WHERE id = ?
        """, (title, session_id))
    conn.commit()
    conn.close()


# --------------------------------------------------
# USER-ISOLATED PERSONALIZED STUDY PLANS
# --------------------------------------------------

def save_study_plan(user_id: int, topic: str, target_level: str, schedule_summary: str, plan_content: str) -> int:
    """Saves a personalized study plan for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO study_plans (user_id, topic, target_level, schedule_summary, plan_content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, topic, target_level, schedule_summary, plan_content, timestamp))
    plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return plan_id


def get_latest_study_plan(user_id: int) -> Optional[Dict[str, Any]]:
    """Returns the most recent study plan for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, target_level, schedule_summary, plan_content, created_at
        FROM study_plans
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (user_id,))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "topic": r[1],
        "target_level": r[2],
        "schedule_summary": r[3],
        "plan_content": r[4],
        "created_at": r[5]
    }


def get_user_study_plans(user_id: int) -> List[Dict[str, Any]]:
    """Returns all study plans created for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, target_level, schedule_summary, plan_content, created_at
        FROM study_plans
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{
        "id": r[0],
        "topic": r[1],
        "target_level": r[2],
        "schedule_summary": r[3],
        "plan_content": r[4],
        "created_at": r[5]
    } for r in rows]


# --------------------------------------------------
# MULTIPLE PERSISTENT NOTES MANAGEMENT
# --------------------------------------------------

def create_note(
    user_id: int,
    title: str,
    content: str,
    subject: str = "General",
    associated_files: Optional[List[str]] = None
) -> int:
    """Creates a new note for a user without overwriting existing notes."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    files_json = json.dumps(associated_files or [])
    cursor.execute("""
        INSERT INTO notes (user_id, title, content, subject, associated_files, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, title.strip() or "Untitled Note", content, subject, files_json, now_str, now_str))
    note_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return note_id


def update_note(
    note_id: int,
    user_id: int,
    title: str,
    content: str,
    subject: str = "General",
    associated_files: Optional[List[str]] = None
) -> bool:
    """Updates an existing note for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    files_json = json.dumps(associated_files or [])
    cursor.execute("""
        UPDATE notes
        SET title = ?, content = ?, subject = ?, associated_files = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
    """, (title.strip() or "Untitled Note", content, subject, files_json, now_str, note_id, user_id))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def delete_note(note_id: int, user_id: int) -> bool:
    """Deletes a note for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def get_user_notes(user_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all notes for a specific user, optionally filtered by subject."""
    conn = get_connection()
    cursor = conn.cursor()
    if subject and subject != "All Subjects" and subject != "General":
        cursor.execute("""
            SELECT id, user_id, title, content, subject, associated_files, created_at, updated_at
            FROM notes
            WHERE user_id = ? AND subject = ?
            ORDER BY updated_at DESC, id DESC
        """, (user_id, subject))
    else:
        cursor.execute("""
            SELECT id, user_id, title, content, subject, associated_files, created_at, updated_at
            FROM notes
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
        """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    notes = []
    for r in rows:
        files = []
        if r[5]:
            try:
                files = json.loads(r[5])
            except Exception:
                files = []
        notes.append({
            "id": r[0],
            "user_id": r[1],
            "title": r[2],
            "content": r[3],
            "subject": r[4],
            "associated_files": files,
            "created_at": r[6],
            "updated_at": r[7]
        })
    return notes


def get_note_by_id(note_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single note by ID for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, title, content, subject, associated_files, created_at, updated_at
        FROM notes
        WHERE id = ? AND user_id = ?
    """, (note_id, user_id))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    files = []
    if r[5]:
        try:
            files = json.loads(r[5])
        except Exception:
            files = []
    return {
        "id": r[0],
        "user_id": r[1],
        "title": r[2],
        "content": r[3],
        "subject": r[4],
        "associated_files": files,
        "created_at": r[6],
        "updated_at": r[7]
    }


# --------------------------------------------------
# PERSISTENT BACKGROUND QUIZ JOBS
# --------------------------------------------------

def create_quiz_job(
    user_id: int,
    subject: str,
    topic: str,
    difficulty: str,
    number_of_questions: int,
    source_type: str = "topic",
    source_id: Optional[int] = None,
    source_name: Optional[str] = None
) -> int:
    """Creates a persistent background quiz generation job record."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO quiz_jobs (
            user_id, subject, topic, difficulty, number_of_questions,
            source_type, source_id, source_name, status, progress,
            error_message, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?)
    """, (
        user_id, subject, topic, difficulty, number_of_questions,
        source_type, source_id, source_name or topic, now_str
    ))
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return job_id


def update_quiz_job_status(
    job_id: int,
    status: str,
    progress: int = 0,
    quiz_data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
):
    """Updates the status, progress, questions, or error of a quiz generation job."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    quiz_json = json.dumps(quiz_data) if quiz_data is not None else None

    if status == "completed":
        cursor.execute("""
            UPDATE quiz_jobs
            SET status = ?, progress = 100, quiz_data = COALESCE(?, quiz_data),
                error_message = NULL, completed_at = ?
            WHERE id = ?
        """, (status, quiz_json, now_str, job_id))
    elif status == "failed":
        cursor.execute("""
            UPDATE quiz_jobs
            SET status = ?, progress = ?, error_message = ?, completed_at = ?
            WHERE id = ?
        """, (status, progress, error_message, now_str, job_id))
    else:
        cursor.execute("""
            UPDATE quiz_jobs
            SET status = ?, progress = ?, quiz_data = COALESCE(?, quiz_data)
            WHERE id = ?
        """, (status, progress, quiz_json, job_id))

    conn.commit()
    conn.close()


def get_user_quiz_jobs(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves all quiz generation jobs for a specific user ordered by latest."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, subject, topic, difficulty, number_of_questions,
               source_type, source_id, source_name, status, progress,
               quiz_data, error_message, created_at, completed_at
        FROM quiz_jobs
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for r in rows:
        data = None
        if r[11]:
            try:
                data = json.loads(r[11])
            except Exception:
                data = None
        jobs.append({
            "id": r[0],
            "user_id": r[1],
            "subject": r[2],
            "topic": r[3],
            "difficulty": r[4],
            "number_of_questions": r[5],
            "source_type": r[6],
            "source_id": r[7],
            "source_name": r[8],
            "status": r[9],
            "progress": r[10],
            "quiz_data": data,
            "error_message": r[12],
            "created_at": r[13],
            "completed_at": r[14]
        })
    return jobs


def get_quiz_job_by_id(job_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single quiz job by ID for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, subject, topic, difficulty, number_of_questions,
               source_type, source_id, source_name, status, progress,
               quiz_data, error_message, created_at, completed_at
        FROM quiz_jobs
        WHERE id = ? AND user_id = ?
    """, (job_id, user_id))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return None
    data = None
    if r[11]:
        try:
            data = json.loads(r[11])
        except Exception:
            data = None
    return {
        "id": r[0],
        "user_id": r[1],
        "subject": r[2],
        "topic": r[3],
        "difficulty": r[4],
        "number_of_questions": r[5],
        "source_type": r[6],
        "source_id": r[7],
        "source_name": r[8],
        "status": r[9],
        "progress": r[10],
        "quiz_data": data,
        "error_message": r[12],
        "created_at": r[13],
        "completed_at": r[14]
    }


def get_latest_quiz_job(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent quiz generation job for a user."""
    jobs = get_user_quiz_jobs(user_id, limit=1)
    return jobs[0] if jobs else None


def delete_quiz_job(job_id: int, user_id: int) -> bool:
    """Deletes a quiz job record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM quiz_jobs WHERE id = ? AND user_id = ?", (job_id, user_id))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected