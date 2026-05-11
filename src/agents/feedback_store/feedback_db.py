"""SQLite helpers for feedback persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def init_db(db_path: str) -> None:
    """Create the feedback database if it does not exist."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.commit()


def insert_feedback(db_path: str, record: dict) -> int:
    """Insert one feedback record and return the row id."""

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback (event_id, state, action_taken, human_action, reward, timestamp)
            VALUES (:event_id, :state, :action_taken, :human_action, :reward, :timestamp)
            """,
            record,
        )
        connection.commit()
        return int(cursor.lastrowid)


def fetch_feedback(db_path: str, limit: int | None = None) -> list[dict]:
    """Fetch feedback rows as dictionaries."""

    query = "SELECT id, event_id, state, action_taken, human_action, reward, timestamp FROM feedback ORDER BY id DESC"
    params: Iterable[object] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]
