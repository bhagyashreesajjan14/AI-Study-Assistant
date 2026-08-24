import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Explain DBMS normalization in simple terms."
        }
    ]
)

print(response["message"]["content"])