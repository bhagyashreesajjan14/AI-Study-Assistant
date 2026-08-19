import pytest
from pdf_generator import (
    generate_response_pdf,
    generate_chat_pdf,
    derive_pdf_filename,
    HAS_REPORTLAB
)


def test_derive_pdf_filename():
    fname1 = derive_pdf_filename("Explain Database Normalization with 1NF, 2NF, 3NF")
    assert fname1.endswith(".pdf")
    assert "Database_Normalization" in fname1 or "Explain_Database" in fname1

    fname2 = derive_pdf_filename("ACID Properties!", prefix="Notes")
    assert fname2.startswith("Notes_")
    assert fname2.endswith(".pdf")


@pytest.mark.skipif(not HAS_REPORTLAB, reason="ReportLab not available")
def test_generate_response_pdf():
    content = """
# Database Design Principles

## 1. Entity Integrity
- Every table should have a primary key.
- Primary key values must be unique and not null.

## 2. Referential Integrity
- Foreign keys maintain valid relationships between tables.

| Term | Definition |
| --- | --- |
| Primary Key | Unique row identifier |
| Foreign Key | References primary key in another table |

```sql
CREATE TABLE Students (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);
```
"""
    pdf_bytes = generate_response_pdf(
        title="Database Design Principles",
        content=content,
        student_name="Alex Johnson",
        subject="dbms"
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


@pytest.mark.skipif(not HAS_REPORTLAB, reason="ReportLab not available")
def test_generate_chat_pdf():
    messages = [
        {"role": "user", "content": "What is the CAP theorem in distributed databases?"},
        {"role": "assistant", "content": "The CAP theorem states that a distributed system can only provide two of **Consistency**, **Availability**, and **Partition Tolerance** simultaneously."}
    ]
    chat_pdf = generate_chat_pdf(
        session_title="Distributed Systems Chat",
        subject="computer_networks",
        student_name="Alex Johnson",
        messages=messages
    )
    assert isinstance(chat_pdf, bytes)
    assert len(chat_pdf) > 500
    assert chat_pdf.startswith(b"%PDF")
