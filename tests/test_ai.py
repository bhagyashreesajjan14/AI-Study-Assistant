from ai import ask_ai


def test_ai_connection():

    answer = ask_ai(
        "What is a database?"
    )

    assert answer
    assert isinstance(answer, str)