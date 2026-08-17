from typing import List, Dict, Tuple, Optional, Any


def validate_all_answered(quiz: List[Dict[str, Any]], answers: Dict[int, Optional[int]]) -> Tuple[bool, List[int]]:
    """
    Validates whether all questions in the quiz have been answered by the student.
    Returns (is_complete, list_of_unanswered_question_1based_numbers).
    """
    missing = []
    for i in range(len(quiz)):
        ans = answers.get(i)
        if ans is None:
            missing.append(i + 1)
    return len(missing) == 0, missing


def calculate_score(
    quiz: List[Dict[str, Any]],
    answers: Dict[int, Optional[int]]
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """
    Calculates the final score and generates detailed solution analysis for each question.
    """
    correct = 0
    details = []

    for i, question in enumerate(quiz):
        selected_index = answers.get(i)
        correct_index = question["answer"]

        is_correct = (
            selected_index is not None
            and selected_index == correct_index
        )

        if is_correct:
            correct += 1

        selected_text = (
            question["options"][selected_index]
            if selected_index is not None and 0 <= selected_index < len(question["options"])
            else "Not answered"
        )

        correct_text = (
            question["options"][correct_index]
            if 0 <= correct_index < len(question["options"])
            else ""
        )

        details.append({
            "question": question["question"],
            "selected_answer": selected_text,
            "correct_answer": correct_text,
            "is_correct": is_correct,
            "explanation": question.get("explanation", "")
        })

    total = len(quiz)
    score = (correct / total * 100) if total else 0.0

    return correct, total, score, details


def choose_difficulty(score: Optional[float]) -> str:
    """Adapts difficulty level based on previous performance score."""
    if score is None:
        return "Easy"

    if score < 50:
        return "Easy"

    if score < 75:
        return "Medium"

    return "Hard"