import json
import ollama

from config import LLM_MODEL


def ask_ai(question, context=""):

    if context.strip():

        prompt = f"""
You are a college AI tutor.

Answer the student's question using the provided study
material as the primary source.

STUDY MATERIAL:
{context}

QUESTION:
{question}

Rules:
1. Explain clearly.
2. Use simple language.
3. Give examples when useful.
4. Do not invent facts from the study material.
5. If the answer is not present in the study material,
   say that clearly.
"""

    else:

        prompt = f"""
You are a helpful college AI tutor.

Answer this question clearly:

{question}

Use simple language and examples where appropriate.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def generate_quiz(
    subject,
    topic,
    difficulty,
    number_of_questions=5
):

    prompt = f"""
Create a college-level multiple-choice quiz.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}
Number of questions: {number_of_questions}

Return ONLY JSON.

Format:

{{
    "questions": [
        {{
            "question": "Question",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0,
            "explanation": "Explanation"
        }}
    ]
}}

Rules:

- answer must be 0, 1, 2 or 3.
- Exactly {number_of_questions} questions.
- Questions must be relevant to the topic.
- Avoid ambiguous questions.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    raw = response["message"]["content"]

    return json.loads(raw)


def explain_mistake(
    question,
    student_answer,
    correct_answer,
    topic
):

    prompt = f"""
You are an AI tutor helping a college student understand
a mistake.

Topic:
{topic}

Question:
{question}

Student answer:
{student_answer}

Correct answer:
{correct_answer}

Explain:
1. Why the student's answer is incorrect.
2. Why the correct answer is correct.
3. The concept the student should revise.

Keep the explanation educational and concise.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]