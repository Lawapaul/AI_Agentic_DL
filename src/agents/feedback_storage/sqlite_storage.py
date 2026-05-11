"""SQLite feedback storage backend."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List


class SQLiteFeedbackStorage:
    """Structured feedback storage with SQL queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_table()

    def _init_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features TEXT,
                predicted_attack TEXT,
                risk_score REAL,
                decision TEXT,
                human_feedback TEXT,
                final_label TEXT
            )
            """
        )
        self.conn.commit()

    def save(self, record: Dict[str, object]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (timestamp, features, predicted_attack, risk_score, decision, human_feedback, final_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("timestamp", "")),
                json.dumps(record.get("features", []), ensure_ascii=True),
                str(record.get("predicted_attack", "")),
                float(record.get("risk_score", 0.0)),
                str(record.get("decision", "")),
                str(record.get("human_feedback", "")),
                str(record.get("final_label", "")),
            ),
        )
        self.conn.commit()

    def query_recent(self, limit: int = 50) -> List[Dict[str, object]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT timestamp, features, predicted_attack, risk_score, decision, human_feedback, final_label
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        out = []
        for ts, features, pred, risk, decision, hf, label in rows:
            out.append(
                {
                    "timestamp": ts,
                    "features": json.loads(features),
                    "predicted_attack": pred,
                    "risk_score": risk,
                    "decision": decision,
                    "human_feedback": hf,
                    "final_label": label,
                }
            )
        return out
