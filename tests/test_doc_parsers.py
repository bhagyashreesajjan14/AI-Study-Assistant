import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import rag


def test_extract_plain_text(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Introduction to Operating Systems\nProcesses and Threads.", encoding="utf-8")
    extracted = rag.extract_plain_text(str(txt_file))
    assert "Operating Systems" in extracted
    assert "Processes and Threads" in extracted


@patch("rag.docx.Document")
def test_extract_docx_text_with_tables(mock_docx_doc):
    # Mock paragraphs
    mock_p1 = MagicMock()
    mock_p1.text = "Chapter 1: Database Management Systems"
    mock_p2 = MagicMock()
    mock_p2.text = "Relational Data Models"

    # Mock table with rows and cells
    mock_cell1 = MagicMock()
    mock_cell1.text = "1NF"
    mock_cell2 = MagicMock()
    mock_cell2.text = "Atomic values only"

    mock_row = MagicMock()
    mock_row.cells = [mock_cell1, mock_cell2]

    mock_table = MagicMock()
    mock_table.rows = [mock_row]

    mock_instance = MagicMock()
    mock_instance.paragraphs = [mock_p1, mock_p2]
    mock_instance.tables = [mock_table]
    mock_instance.sections = []
    mock_instance._element.xpath.return_value = []
    mock_docx_doc.return_value = mock_instance

    extracted = rag.extract_docx_text("dummy.docx")
    assert "Database Management Systems" in extracted
    assert "1NF | Atomic values only" in extracted


def test_extract_doc_rtf_fallback(tmp_path):
    # Test RTF format disguised as .doc
    rtf_file = tmp_path / "sample.doc"
    rtf_content = b'{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Courier;}}\\f0\\fs20 Core Computer Networks Concepts: TCP/IP and OSI Layering.}'
    rtf_file.write_bytes(rtf_content)

    extracted = rag.extract_doc_text(str(rtf_file))
    assert "Computer Networks Concepts" in extracted
    assert "TCP/IP" in extracted


@patch("rag._ocr_engine")
def test_extract_image_text_rapidocr(mock_ocr):
    mock_ocr.return_value = (
        [
            (None, "Diagram of CPU Architecture"),
            (None, "ALU, Control Unit, Registers")
        ],
        None
    )
    extracted = rag.extract_image_text("test_screenshot.png")
    assert "CPU Architecture" in extracted
    assert "Registers" in extracted
