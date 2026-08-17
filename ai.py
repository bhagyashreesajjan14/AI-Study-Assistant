import json
from typing import List, Dict, Any, Optional
import ollama

from config import LLM_MODEL


def ask_ai(question: str, context: str = "") -> str:
    """Answers a single academic question using optional study context."""
    if context.strip():
        prompt = f"""
You are a college AI tutor.

Answer the student's question using the provided study material as the primary source.

STUDY MATERIAL:
{context}

QUESTION:
{question}

Rules:
1. Explain clearly in simple language.
2. Use examples when useful.
3. Do not invent facts outside the study material.
4. If the answer is not present in the study material, say clearly: "I couldn't find this information in your uploaded notes."
"""
    else:
        prompt = f"""
You are a helpful college AI tutor.

Answer this question clearly:
{question}

Use simple language and practical examples where appropriate.
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
    subject: str,
    topic: str,
    difficulty: str = "Intermediate",
    number_of_questions: int = 5
) -> Dict[str, Any]:
    """
    Generates a college-level multiple-choice quiz tailored specifically to the user's manual topic and target level.
    """
    level_guidance = ""
    if difficulty == "Beginner":
        level_guidance = "Focus on foundational concepts, fundamental definitions, and basic principles suitable for a beginner."
    elif difficulty == "Advanced":
        level_guidance = "Focus on complex edge-cases, deep architectural mechanisms, advanced problem-solving, and in-depth analysis."
    else:
        level_guidance = "Focus on standard conceptual understanding, practical scenarios, and core college-level curriculum questions."

    prompt = f"""
Create a college-level multiple-choice quiz.

Subject: {subject}
Topic: {topic}
Target Level / Difficulty: {difficulty}
Number of questions: {number_of_questions}

Level Guidance:
{level_guidance}

Return ONLY JSON.

Format:
{{
    "questions": [
        {{
            "question": "Question text specifically about {topic}",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0,
            "explanation": "Clear explanation of why Option A is correct and why other options are incorrect."
        }}
    ]
}}

Rules:
- answer must be an integer index: 0, 1, 2, or 3.
- Exactly {number_of_questions} questions.
- Every question MUST be directly relevant to the topic: "{topic}".
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
    question: str,
    student_answer: str,
    correct_answer: str,
    topic: str
) -> str:
    """Generates a detailed educational explanation of a mistake made in a quiz."""
    prompt = f"""
You are an AI tutor helping a college student understand a mistake in a quiz.

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
3. The specific core concept the student should revise.

Keep the explanation educational, structured, and concise.
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


def ask_ai_chat(
    messages: List[Dict[str, str]],
    context: str = "",
    mode: str = "Explain",
    notes_only: bool = False
) -> str:
    """
    Conversational AI Tutor with strict notes-only mode support.
    """
    if notes_only:
        system_prompt = f"""
You are an academic AI tutor answering questions strictly and exclusively based on the student's uploaded notes.

STUDY MATERIAL:
---
{context}
---

CRITICAL RULES:
1. Base your answer ONLY and EXCLUSIVELY on the provided STUDY MATERIAL above.
2. Do NOT use any pre-trained or outside general knowledge.
3. Do NOT speculate, extrapolate, or hallucinate.
4. If the answer is NOT present or cannot be directly proven from the study material above, you MUST respond EXACTLY with:
"I couldn't find this information in your uploaded notes."
5. If the study material contains the answer, explain it clearly and cite the relevant concepts.
"""
    else:
        system_prompt = "You are a helpful college AI tutor.\n\n"

        if mode == "Simple Explanation":
            system_prompt += "Focus on explaining concepts in a very simple, beginner-friendly way with clear analogies.\n"
        elif mode == "Example":
            system_prompt += "Explain concepts clearly and provide practical, real-world examples.\n"
        elif mode == "Exam Preparation":
            system_prompt += "Explain concepts in an exam-oriented format, highlighting key bullet points, formulas, and potential exam question angles.\n"
        else:
            system_prompt += "Explain concepts clearly and educationally in simple, structured language.\n"

        if context.strip():
            system_prompt += f"""
Use the following provided study material as your primary source of truth:
---
{context}
---
Answer the user's questions clearly, prioritizing the study material.
"""
        else:
            system_prompt += "\nAnswer the user's questions clearly, using simple language and examples where appropriate."

    ollama_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        ollama_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = ollama.chat(
        model=LLM_MODEL,
        messages=ollama_messages
    )

    return response["message"]["content"]