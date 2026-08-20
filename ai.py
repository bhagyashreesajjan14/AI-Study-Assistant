import json
import re
import threading
from typing import List, Dict, Any, Optional
from groq import Groq

from config import LLM_MODEL, get_groq_api_key
from database import update_quiz_job_status, create_quiz_job


def fix_mermaid_syntax(markdown_text: str) -> str:
    """
    Fixes common Mermaid diagram parse errors caused by unquoted parentheses, brackets, or colons
    inside node labels like D[Explore (EDA)] -> D["Explore (EDA)"].
    """
    if not markdown_text or "```mermaid" not in markdown_text:
        return markdown_text

    def _repair_block(match: re.Match) -> str:
        code = match.group(1)
        lines = []
        for line in code.splitlines():
            # Fix unquoted square bracket node labels with parentheses: D[Explore (EDA)] -> D["Explore (EDA)"]
            line = re.sub(r'(\b[A-Za-z0-9_]+)\[([^"\]\n]*\([^"\]\n]*\)[^"\]\n]*)\]', r'\1["\2"]', line)
            # Fix unquoted round bracket node labels with parentheses: D(Explore (EDA)) -> D("Explore (EDA)")
            line = re.sub(r'(\b[A-Za-z0-9_]+)\(([^"\)\n]*\([^"\)\n]*\)[^"\)\n]*)\)', r'\1("\2")', line)
            # Fix unquoted curly bracket node labels with parentheses: D{Explore (EDA)} -> D{"Explore (EDA)"}
            line = re.sub(r'(\b[A-Za-z0-9_]+)\{([^"\}\n]*\([^"\}\n]*\)[^"\}\n]*)\}', r'\1{"\2"}', line)
            lines.append(line)
        return "```mermaid\n" + "\n".join(lines) + "\n```"

    return re.sub(r'```mermaid\s*\n?(.*?)\n?```', _repair_block, markdown_text, flags=re.DOTALL)


def get_groq_client() -> Groq:
    """Returns a Groq API client initialized with the configured Groq API key."""
    api_key = get_groq_api_key()
    return Groq(api_key=api_key or "PASTE_YOUR_GROQ_API_KEY_HERE")


def _chat_completion(messages: List[Dict[str, str]], json_mode: bool = False) -> str:
    """Helper to call Groq chat completion synchronously."""
    client = get_groq_client()
    kwargs: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return fix_mermaid_syntax(content) if not json_mode else content


def _chat_completion_stream(messages: List[Dict[str, str]]):
    """Helper to stream Groq chat completion tokens."""
    client = get_groq_client()
    stream_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    )
    for chunk in stream_response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            content_piece = getattr(delta, "content", None) or ""
            if content_piece:
                yield content_piece


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

    return _chat_completion([{"role": "user", "content": prompt}])


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

    raw = _chat_completion([{"role": "user", "content": prompt}], json_mode=True)
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

    return _chat_completion([{"role": "user", "content": prompt}])


def _build_tutor_messages(
    messages: List[Dict[str, str]],
    context: str = "",
    mode: str = "Explain",
    notes_only: bool = False
) -> List[Dict[str, str]]:
    """Builds structured system and conversational prompt messages for AI Tutor."""
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
5. If the study material contains the answer, explain it clearly, structure with headings/bullet points, and cite the relevant concepts.
"""
    else:
        system_prompt = "You are a helpful and knowledgeable college AI study tutor.\n\n"

        if mode == "Simple Explanation":
            system_prompt += "Focus on explaining concepts in a very simple, beginner-friendly way with clear analogies.\n"
        elif mode == "Example":
            system_prompt += "Explain concepts clearly and provide practical, real-world examples and code/diagram descriptions where appropriate.\n"
        elif mode == "Exam Preparation":
            system_prompt += "Explain concepts in an exam-oriented format, highlighting key bullet points, formulas, common pitfalls, and potential exam question angles.\n"
        else:
            system_prompt += "Explain concepts clearly, educationally, and comprehensively with structured headings, paragraphs, bullet points, and code blocks where applicable.\n"

        system_prompt += "\nFormatting Guidelines:\n- Use Markdown headings (##, ###), bullet lists, bold text, and numbered steps for readability.\n- If generating Mermaid diagrams (```mermaid), ALWAYS enclose node labels in double quotes if they contain parentheses, colons, or special characters (e.g., write D[\"Explore (EDA)\"] instead of D[Explore (EDA)]).\n- When solving problems, present step-by-step reasoning.\n- Never claim you cannot provide study notes or materials; provide rich educational content directly in your response.\n"

        if context.strip():
            system_prompt += f"""
Use the following provided study material as your primary source of truth:
---
{context}
---
Answer the user's questions clearly, prioritizing the study material.
"""
        else:
            system_prompt += "\nAnswer the student's questions clearly, using simple language, clear structure, and examples where appropriate."

    groq_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        groq_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return groq_messages


def ask_ai_chat(
    messages: List[Dict[str, str]],
    context: str = "",
    mode: str = "Explain",
    notes_only: bool = False
) -> str:
    """
    Conversational AI Tutor with strict notes-only mode support (complete response).
    """
    groq_messages = _build_tutor_messages(messages, context=context, mode=mode, notes_only=notes_only)
    return _chat_completion(groq_messages)


def ask_ai_chat_stream(
    messages: List[Dict[str, str]],
    context: str = "",
    mode: str = "Explain",
    notes_only: bool = False
):
    """
    GPT-style progressive streaming AI Tutor generator that yields text chunks/tokens as they arrive.
    """
    groq_messages = _build_tutor_messages(messages, context=context, mode=mode, notes_only=notes_only)
    yield from _chat_completion_stream(groq_messages)


def generate_quiz_from_material(
    content: str,
    subject: str,
    target_level: str = "Intermediate",
    number_of_questions: int = 5
) -> Dict[str, Any]:
    """
    Generates a multiple-choice quiz directly from uploaded learning material (PDF/Image/Notes).
    """
    # Truncate material to avoid exceeding LLM context window if very large
    sample_content = content[:7000] if len(content) > 7000 else content

    prompt = f"""
You are an expert college examiner creating a multiple-choice practice quiz directly based on the provided study material.

STUDY MATERIAL:
---
{sample_content}
---

Subject: {subject}
Difficulty / Target Level: {target_level}
Number of Questions: {number_of_questions}

Return ONLY valid JSON in the following format:
{{
    "questions": [
        {{
            "question": "Question based directly on concepts in the study material",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0,
            "explanation": "Clear explanation referencing facts from the study material."
        }}
    ]
}}

Rules:
- Generate exactly {number_of_questions} questions.
- answer MUST be an integer (0, 1, 2, or 3) corresponding to the correct option.
- All questions MUST be strictly based on the provided study material.
- Make incorrect options realistic and educational.
"""

    raw = _chat_completion([{"role": "user", "content": prompt}], json_mode=True)
    return json.loads(raw)


def generate_summary(
    content: str,
    summary_type: str = "Detailed"
) -> str:
    """
    Generates a structured executive or detailed summary of the uploaded document/image.
    """
    sample_content = content[:8000] if len(content) > 8000 else content

    prompt = f"""
You are a helpful college AI study tutor.

Please summarize the following study material clearly and comprehensively.

STUDY MATERIAL:
---
{sample_content}
---

Summary Format:
- **Core Summary**: A concise 2-3 paragraph overview of the main topic.
- **Key Takeaways & Concepts**: Bulleted list of critical definitions, laws, equations, or concepts.
- **Important Insights**: Real-world application, exam focal points, or edge cases.

Style: {summary_type} (Clear, structured, easy for students to revise quickly).
"""

    return _chat_completion([{"role": "user", "content": prompt}])


def generate_flashcards(
    content: str,
    num_cards: int = 6
) -> List[Dict[str, str]]:
    """
    Generates question/concept front and definition/answer back flashcards from uploaded material.
    """
    sample_content = content[:7000] if len(content) > 7000 else content

    prompt = f"""
Generate {num_cards} study flashcards based on the provided material.

STUDY MATERIAL:
---
{sample_content}
---

Return ONLY valid JSON in the format:
{{
    "flashcards": [
        {{
            "front": "Key concept or question to test recall",
            "back": "Clear, concise definition or answer",
            "tag": "Category/Topic Tag"
        }}
    ]
}}

Rules:
- Exactly {num_cards} flashcards.
- Grounded directly in the provided material.
"""

    raw = _chat_completion([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)
    return data.get("flashcards", [])


def generate_notes_explanation(
    content: str,
    focus_area: str = ""
) -> str:
    """
    Generates detailed conceptual breakdown and study notes from the uploaded material.
    """
    sample_content = content[:8000] if len(content) > 8000 else content
    focus_clause = f"Focus particularly on: {focus_area}" if focus_area.strip() else "Cover all major sections."

    prompt = f"""
You are an expert college professor explaining concepts from the student's study material.

STUDY MATERIAL:
---
{sample_content}
---

{focus_clause}

Provide:
1. **Concept Breakdown**: Step-by-step educational explanation in simple terms.
2. **Real-World Analogies / Examples**: Intuitive mental models to understand the concept easily.
3. **Common Pitfalls & Exam Tips**: Frequent mistakes students make and how to avoid them.
"""

    return _chat_completion([{"role": "user", "content": prompt}])


# --------------------------------------------------
# BACKGROUND ASYNCHRONOUS QUIZ GENERATION WORKER
# --------------------------------------------------

def process_quiz_job_background(
    job_id: int,
    user_id: int,
    subject: str,
    topic: str,
    difficulty: str,
    number_of_questions: int,
    source_type: str,
    content: str = ""
):
    """
    Background worker executed in a separate daemon thread for independent quiz generation.
    Does not tie to any UI component or page lifecycle.
    """
    try:
        # Step 1: Mark job as processing (20%)
        update_quiz_job_status(job_id, status="processing", progress=20)

        # Step 2: Generate quiz based on source type (70%)
        if source_type == "material" and content.strip():
            update_quiz_job_status(job_id, status="processing", progress=40)
            quiz_result = generate_quiz_from_material(
                content=content,
                subject=subject,
                target_level=difficulty,
                number_of_questions=number_of_questions
            )
        else:
            update_quiz_job_status(job_id, status="processing", progress=40)
            quiz_result = generate_quiz(
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                number_of_questions=number_of_questions
            )

        questions = quiz_result.get("questions", [])
        if not questions:
            raise ValueError("The AI model failed to produce structured questions. Please retry.")

        # Step 3: Complete job & save questions to database (100%)
        update_quiz_job_status(
            job_id=job_id,
            status="completed",
            progress=100,
            quiz_data={"questions": questions}
        )

    except Exception as e:
        error_msg = str(e)
        update_quiz_job_status(
            job_id=job_id,
            status="failed",
            progress=0,
            error_message=error_msg
        )


def start_background_quiz_generation(
    user_id: int,
    subject: str,
    topic: str,
    difficulty: str = "Intermediate",
    number_of_questions: int = 5,
    source_type: str = "topic",
    source_id: Optional[int] = None,
    source_name: Optional[str] = None,
    content: str = ""
) -> int:
    """
    Creates persistent quiz job record in database and spawns an independent daemon worker thread.
    Returns job_id.
    """
    job_id = create_quiz_job(
        user_id=user_id,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        source_type=source_type,
        source_id=source_id,
        source_name=source_name
    )

    worker = threading.Thread(
        target=process_quiz_job_background,
        args=(job_id, user_id, subject, topic, difficulty, number_of_questions, source_type, content),
        daemon=True,
        name=f"QuizWorker-Job-{job_id}"
    )
    worker.start()

    return job_id