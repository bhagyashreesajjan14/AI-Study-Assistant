# AI-Powered Study Assistant

An AI and Machine Learning based personalized learning assistant for college students.

## Features

- PDF study material upload
- Question answering using RAG
- Local LLM using Ollama
- Quiz generation
- Quiz performance tracking
- SQLite database
- ML-based performance analysis
- Weak topic detection
- Personalized study recommendations
- AI-generated study plans

## Technology Stack

- Python
- Streamlit
- Ollama
- Qwen3
- FAISS
- Sentence Transformers
- PyMuPDF
- Scikit-learn
- SQLite
- Plotly

## Architecture

Student
↓
Streamlit
↓
RAG / Quiz / Performance
↓
Ollama + Machine Learning
↓
Personalized Recommendations

## Running the project

Create a virtual environment:

```bash
python -m venv .venv