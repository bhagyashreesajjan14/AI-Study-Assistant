from unittest.mock import patch
from ai import ask_ai, generate_quiz, explain_mistake, ask_ai_chat


@patch("ai.ollama.chat")
def test_ask_ai(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "A database is an organized collection of data."}
    }
    answer = ask_ai("What is a database?")
    assert answer == "A database is an organized collection of data."
    assert isinstance(answer, str)


@patch("ai.ollama.chat")
def test_ask_ai_with_context(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "ACID properties ensure reliable transactions."}
    }
    answer = ask_ai("Explain ACID", context="Transactions are atomic, consistent, isolated, durable.")
    assert "ACID" in answer


@patch("ai.ollama.chat")
def test_generate_quiz(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": '{"questions": [{"question": "What is SQL?", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "SQL is structured query language"}]}'
        }
    }
    quiz = generate_quiz("dbms", "SQL", "Easy", number_of_questions=1)
    assert "questions" in quiz
    assert len(quiz["questions"]) == 1
    assert quiz["questions"][0]["answer"] == 0


@patch("ai.ollama.chat")
def test_explain_mistake(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "You selected option B, but option A is correct because..."}
    }
    explanation = explain_mistake("What is SQL?", "Option B", "Option A", "SQL")
    assert "Option B" in explanation or "correct" in explanation


@patch("ai.ollama.chat")
def test_ask_ai_chat(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "Normalization helps reduce data redundancy."}
    }
    messages = [
        {"role": "user", "content": "What is normalization?"}
    ]
    response = ask_ai_chat(messages, context="1NF, 2NF, 3NF", mode="Simple Explanation")
    assert "Normalization" in response