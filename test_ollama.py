from groq import Groq
from config import LLM_MODEL, get_groq_api_key


def main():
    api_key = get_groq_api_key()
    client = Groq(api_key=api_key or "PASTE_YOUR_GROQ_API_KEY_HERE")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Explain DBMS normalization in simple terms."
            }
        ]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()