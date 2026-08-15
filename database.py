import sqlite3
from pathlib import Path
from datetime import datetime

from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            course TEXT,
            semester INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT,
            score REAL NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER,
            question TEXT,
            selected_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER,
            topic TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_profile(name, course, semester):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM student_profile")

    cursor.execute("""
        INSERT INTO student_profile
        (name, course, semester)
        VALUES (?, ?, ?)
    """, (name, course, semester))

    conn.commit()
    conn.close()


def get_profile():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, course, semester
        FROM student_profile
        LIMIT 1
    """)

    result = cursor.fetchone()

    conn.close()

    return result


def save_quiz_result(
    subject,
    topic,
    difficulty,
    score,
    total_questions,
    correct_answers,
    answers
):

    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO quiz_attempts
        (
            subject,
            topic,
            difficulty,
            score,
            total_questions,
            correct_answers,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
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
                attempt_id,
                question,
                selected_answer,
                correct_answer,
                is_correct,
                topic,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
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


def get_quiz_attempts():

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
        ORDER BY created_at DESC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


def get_topic_performance():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            subject,
            topic,
            AVG(score) AS average_score,
            COUNT(*) AS attempts
        FROM quiz_attempts
        GROUP BY subject, topic
        ORDER BY average_score ASC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


def get_recent_topic_score(subject, topic):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT score
        FROM quiz_attempts
        WHERE subject = ?
        AND topic = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (subject, topic))

    result = cursor.fetchone()

    conn.close()

    return result[0] if result else None