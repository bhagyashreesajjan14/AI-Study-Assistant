import sqlite3
import pytest
from unittest.mock import patch
import database


@pytest.fixture(autouse=True)
def mock_db():
    keep_alive = sqlite3.connect("file::memory:?cache=shared", uri=True)
    def get_test_conn():
        return sqlite3.connect("file::memory:?cache=shared", uri=True)
    with patch("database.get_connection", side_effect=get_test_conn):
        database.init_database()
        yield
    keep_alive.close()


def test_create_and_get_multiple_notes():
    # User 1 creates note 1
    n1_id = database.create_note(
        user_id=1,
        title="DBMS Transactions",
        content="ACID properties explanation and examples.",
        subject="dbms",
        associated_files=["Lecture1.docx"]
    )
    assert n1_id > 0

    # User 1 creates note 2 (does not overwrite note 1)
    n2_id = database.create_note(
        user_id=1,
        title="OS Deadlock",
        content="Banker's algorithm and resource allocation graphs.",
        subject="operating_systems",
        associated_files=["Slide2.png"]
    )
    assert n2_id > 0
    assert n1_id != n2_id

    # Verify both notes exist for User 1
    notes = database.get_user_notes(user_id=1)
    assert len(notes) == 2
    titles = [n["title"] for n in notes]
    assert "DBMS Transactions" in titles
    assert "OS Deadlock" in titles


def test_update_and_delete_notes():
    note_id = database.create_note(
        user_id=1,
        title="Original Title",
        content="Original content",
        subject="General"
    )
    
    # Update note
    updated = database.update_note(
        note_id=note_id,
        user_id=1,
        title="Updated Title",
        content="Updated content with more details.",
        subject="computer_networks"
    )
    assert updated is True

    note = database.get_note_by_id(note_id, user_id=1)
    assert note is not None
    assert note["title"] == "Updated Title"
    assert note["content"] == "Updated content with more details."
    assert note["subject"] == "computer_networks"

    # Delete note
    deleted = database.delete_note(note_id, user_id=1)
    assert deleted is True
    assert database.get_note_by_id(note_id, user_id=1) is None


def test_notes_user_isolation():
    database.create_note(user_id=1, title="User 1 Private Note", content="Secret 1")
    database.create_note(user_id=2, title="User 2 Private Note", content="Secret 2")

    u1_notes = database.get_user_notes(user_id=1)
    u2_notes = database.get_user_notes(user_id=2)

    assert len(u1_notes) == 1
    assert u1_notes[0]["title"] == "User 1 Private Note"

    assert len(u2_notes) == 1
    assert u2_notes[0]["title"] == "User 2 Private Note"
