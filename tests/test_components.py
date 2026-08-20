import pandas as pd
import pytest
from unittest.mock import patch

from quiz import calculate_score, choose_difficulty, validate_all_answered
from recommendations import generate_recommendations, get_weak_topics
from ml_model import load_performance_data, classify_score, analyze_performance, train_model, predict_status
from utils import clean_text, format_subject
from study_planner import format_duration, generate_study_plan, generate_study_plan_pdf


def test_quiz_calculate_score():
    quiz = [
        {"question": "Q1", "options": ["A", "B", "C"], "answer": 1, "explanation": "Exp 1"},
        {"question": "Q2", "options": ["X", "Y", "Z"], "answer": 0, "explanation": "Exp 2"}
    ]
    answers = {0: 1, 1: 2}  # Q1 correct, Q2 wrong
    correct, total, score, details = calculate_score(quiz, answers)
    assert total == 2
    assert correct == 1
    assert score == 50.0
    assert details[0]["is_correct"] is True
    assert details[1]["is_correct"] is False


def test_validate_all_answered():
    quiz = [
        {"question": "Q1", "options": ["A", "B"], "answer": 0},
        {"question": "Q2", "options": ["C", "D"], "answer": 1},
        {"question": "Q3", "options": ["E", "F"], "answer": 0}
    ]
    # Partially answered
    incomplete_answers = {0: 0, 1: None}
    complete, missing = validate_all_answered(quiz, incomplete_answers)
    assert complete is False
    assert missing == [2, 3]  # Question 2 and Question 3 unanswered

    # Fully answered
    complete_answers = {0: 0, 1: 1, 2: 0}
    complete, missing = validate_all_answered(quiz, complete_answers)
    assert complete is True
    assert missing == []


def test_choose_difficulty():
    assert choose_difficulty(None) == "Easy"
    assert choose_difficulty(40) == "Easy"
    assert choose_difficulty(60) == "Medium"
    assert choose_difficulty(85) == "Hard"


def test_recommendations():
    df = pd.DataFrame([
        {"topic": "Normalization", "average_score": 40.0, "attempts": 2},
        {"topic": "Transactions", "average_score": 70.0, "attempts": 3},
        {"topic": "Indexing", "average_score": 90.0, "attempts": 4}
    ])
    recs = generate_recommendations(df)
    assert len(recs) == 3
    assert recs[0]["priority"] == "HIGH"
    assert recs[0]["topic"] == "Normalization"
    
    weak = get_weak_topics(df, threshold=60)
    assert weak == ["Normalization"]


def test_ml_model_and_utils():
    assert clean_text("  hello    world \n \t ") == "hello world"
    assert format_subject("operating_systems") == "Operating Systems"
    
    assert classify_score(45) == "Weak"
    assert classify_score(65) == "Average"
    assert classify_score(80) == "Strong"
    
    df = load_performance_data([])
    assert df.empty
    assert analyze_performance(df).empty


def test_study_planner_format_duration():
    assert format_duration(2, 30) == "2 hours 30 minutes"
    assert format_duration(1, 0) == "1 hour"
    assert format_duration(0, 45) == "45 minutes"
    assert format_duration(0, 0) == "0 minutes"


from unittest.mock import patch, MagicMock

@patch("study_planner.get_groq_client")
def test_study_planner_generation_and_pdf(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="## Day 1 — Monday\n- Review Transaction concepts\n- Practice ACID questions"))]
    mock_client.chat.completions.create.return_value = mock_response
    schedules = [
        {"day": "Monday", "hours": 2, "minutes": 30, "total_minutes": 150},
        {"day": "Wednesday", "hours": 1, "minutes": 30, "total_minutes": 90}
    ]
    plan = generate_study_plan("Transactions", "Intermediate", schedules)
    assert "Monday" in plan
    assert "ACID" in plan

    pdf_bytes = generate_study_plan_pdf("Alex", "Transactions", "Intermediate", schedules, plan)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")
