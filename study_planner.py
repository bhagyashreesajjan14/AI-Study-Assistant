import ollama

from config import LLM_MODEL


def generate_study_plan(
    weak_topics,
    days=7,
    hours_per_day=2
):

    topics = ", ".join(
        weak_topics
    )

    prompt = f"""
Create a personalized study plan.

Weak topics:
{topics}

Number of days:
{days}

Available study time:
{hours_per_day} hours per day

Create a realistic plan.

For each day include:
- Topic
- Study activity
- Practice activity
- Revision activity

Prioritize the weakest topics.
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