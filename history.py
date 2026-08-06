"""
history.py — Persistent storage for completed DeepFER sessions.

Turns individual video/webcam sessions into a real history: every completed
session gets saved to a local SQLite database (deepfer_history.db, created
automatically alongside your app), so you can compare satisfaction trends
across days instead of only ever looking at one session in isolation.
"""

import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

import pandas as pd

DB_PATH = "deepfer_history.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the sessions table if it doesn't exist yet. Safe to call every
    app startup — CREATE TABLE IF NOT EXISTS is a no-op if already set up.
    """
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                label TEXT,
                total_predictions INTEGER,
                dominant_emotion TEXT,
                satisfaction_score REAL,
                emotion_distribution_json TEXT,
                satisfaction_distribution_json TEXT,
                records_csv TEXT
            )
        """)


def save_session(session, source: str, label: str = "") -> int:
    """Save a completed EmotionSession to the database.

    Args:
        session: an analytics.EmotionSession with data already recorded.
        source: "video" or "webcam" — where this session came from.
        label: optional human-readable name (e.g. the uploaded video's
            filename), shown in the history list to help identify sessions.

    Returns:
        The new row's id.
    """
    summary = session.summary()
    if summary["total_predictions"] == 0:
        raise ValueError("Can't save an empty session — no predictions were recorded.")

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                created_at, source, label, total_predictions, dominant_emotion,
                satisfaction_score, emotion_distribution_json,
                satisfaction_distribution_json, records_csv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                label,
                summary["total_predictions"],
                summary["dominant_emotion"],
                summary["satisfaction_score"],
                json.dumps(summary["emotion_distribution"]),
                json.dumps(summary["satisfaction_distribution"]),
                session.to_csv_bytes().decode("utf-8"),
            ),
        )
        return cursor.lastrowid


def list_sessions() -> pd.DataFrame:
    """Return every saved session as a DataFrame, most recent first —
    used to populate the History page's session list table.
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, created_at, source, label, total_predictions,
                   dominant_emotion, satisfaction_score
            FROM sessions
            ORDER BY created_at DESC
            """,
            conn,
        )
    return df


def load_session_records(session_id: int) -> pd.DataFrame:
    """Load the full per-prediction record log for one saved session
    (reconstructed from its stored CSV) — used to re-render that session's
    charts on the History page.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT records_csv FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if row is None or not row[0]:
        return pd.DataFrame()
    from io import StringIO
    return pd.read_csv(StringIO(row[0]))


def get_trend_data() -> pd.DataFrame:
    """Return (created_at, satisfaction_score, dominant_emotion, source, label)
    for every saved session, sorted chronologically — the data behind the
    "satisfaction over time, across sessions" trend chart.
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT created_at, satisfaction_score, dominant_emotion, source, label
            FROM sessions
            ORDER BY created_at ASC
            """,
            conn,
        )
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_session(session_id: int) -> dict:
    """Return one saved session's full summary as a dict (with the
    distribution JSON columns parsed back into Python dicts) — used by
    the session comparison view.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, source, label, total_predictions,
                   dominant_emotion, satisfaction_score,
                   emotion_distribution_json, satisfaction_distribution_json
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return {}
    return {
        "id": row[0],
        "created_at": row[1],
        "source": row[2],
        "label": row[3],
        "total_predictions": row[4],
        "dominant_emotion": row[5],
        "satisfaction_score": row[6],
        "emotion_distribution": json.loads(row[7]) if row[7] else {},
        "satisfaction_distribution": json.loads(row[8]) if row[8] else {},
    }


def delete_session(session_id: int):
    """Remove one saved session. Only called when the user explicitly asks
    (a delete button per row) — never automatically.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
