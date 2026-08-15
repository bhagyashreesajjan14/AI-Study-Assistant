def calculate_score(
    quiz,
    answers
):

    correct = 0

    details = []

    for i, question in enumerate(quiz):

        selected_index = answers.get(i)

        correct_index = question["answer"]

        is_correct = (
            selected_index == correct_index
        )

        if is_correct:
            correct += 1

        selected_text = (
            question["options"][selected_index]
            if selected_index is not None
            else "Not answered"
        )

        correct_text = (
            question["options"][correct_index]
        )

        details.append({
            "question":
                question["question"],
            "selected_answer":
                selected_text,
            "correct_answer":
                correct_text,
            "is_correct":
                is_correct,
            "explanation":
                question.get(
                    "explanation",
                    ""
                )
        })

    total = len(quiz)

    score = (
        correct / total * 100
        if total
        else 0
    )

    return correct, total, score, details


def choose_difficulty(score):

    if score is None:
        return "Easy"

    if score < 50:
        return "Easy"

    if score < 75:
        return "Medium"

    return "Hard"