import ollama


def main():
    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": "Explain DBMS normalization in simple terms."
            }
        ]
    )

    print(response["message"]["content"])


if __name__ == "__main__":
    main()