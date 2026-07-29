"""Records every /predict call and any later human-confirmed ground truth
for it — the raw material a future retraining run consumes. A prediction
alone is not a label: `true_label` stays NULL until someone calls
POST /feedback, and only verified rows are ever exported for training (see
export.py) — a model must not be allowed to retrain on its own guesses.

SQLite, not because this needs to scale, but because it's a single file
with zero setup, consistent with how mlflow.db already works here.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("flywheel.db")

# RETRAINING_POLICY.md, decision 1: a prediction only goes on the review
# queue if the model itself wasn't confident — a human re-confirming
# something the model already got right at 99% is wasted reviewer time.
# High-confidence predictions are still logged (for /flywheel/stats and in
# case someone corrects one they happen to notice via a direct /feedback
# call), they just never show up in GET /flywheel/pending.
CONFIDENCE_REVIEW_THRESHOLD = 0.70

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    image_path TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_verification INTEGER NOT NULL DEFAULT 0,
    true_label TEXT,
    verified_at TEXT,
    used_in_training INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrain_triggers (
    dataset_type TEXT PRIMARY KEY,
    triggered_at TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    dataset_type: str,
    model_name: str,
    model_version: str,
    image_path: str,
    predicted_class: str,
    confidence: float,
) -> int:
    needs_verification = int(confidence < CONFIDENCE_REVIEW_THRESHOLD)
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO predictions
               (created_at, dataset_type, model_name, model_version, image_path,
                predicted_class, confidence, needs_verification)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                dataset_type,
                model_name,
                model_version,
                str(image_path),
                predicted_class,
                confidence,
                needs_verification,
            ),
        )
        return cursor.lastrowid


def get_pending_verification(dataset_type: str | None = None) -> list[sqlite3.Row]:
    """What a reviewer should actually look at — low-confidence predictions
    that haven't been confirmed/corrected yet. High-confidence predictions
    never appear here even if unverified (see CONFIDENCE_REVIEW_THRESHOLD)."""
    with _connect() as conn:
        return conn.execute(
            """SELECT * FROM predictions
               WHERE needs_verification = 1 AND true_label IS NULL
                 AND (? IS NULL OR dataset_type = ?)
               ORDER BY created_at ASC""",
            (dataset_type, dataset_type),
        ).fetchall()


def submit_feedback(prediction_id: int, true_label: str) -> bool:
    """Returns False if prediction_id doesn't exist — caller should 404."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE predictions SET true_label = ?, verified_at = ? WHERE id = ?",
            (true_label, datetime.now(timezone.utc).isoformat(), prediction_id),
        )
        return cursor.rowcount > 0


def count_verified_unused(dataset_type: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM predictions
               WHERE dataset_type = ? AND true_label IS NOT NULL AND used_in_training = 0""",
            (dataset_type,),
        ).fetchone()
        return row["n"]


def get_verified_unused(dataset_type: str) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            """SELECT * FROM predictions
               WHERE dataset_type = ? AND true_label IS NOT NULL AND used_in_training = 0""",
            (dataset_type,),
        ).fetchall()


def mark_used_in_training(ids: list[int]) -> None:
    with _connect() as conn:
        conn.executemany(
            "UPDATE predictions SET used_in_training = 1 WHERE id = ?", [(i,) for i in ids]
        )


def record_retrain_trigger(dataset_type: str) -> None:
    """Called once check_flywheel_threshold.py actually submits a retrain
    job — resets the "days since we last checked" clock used by decision 2
    in RETRAINING_POLICY.md."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO retrain_triggers (dataset_type, triggered_at) VALUES (?, ?)
               ON CONFLICT(dataset_type) DO UPDATE SET triggered_at = excluded.triggered_at""",
            (dataset_type, datetime.now(timezone.utc).isoformat()),
        )


def days_since_last_trigger(dataset_type: str) -> float | None:
    """None means "never triggered" — treated as infinitely overdue by the
    time-based retraining condition, not as "skip the check"."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT triggered_at FROM retrain_triggers WHERE dataset_type = ?", (dataset_type,)
        ).fetchone()
    if row is None:
        return None
    last = datetime.fromisoformat(row["triggered_at"])
    return (datetime.now(timezone.utc) - last).total_seconds() / 86400


def get_stats(dataset_type: str | None = None) -> dict:
    with _connect() as conn:
        base = "FROM predictions WHERE (? IS NULL OR dataset_type = ?)"
        params = (dataset_type, dataset_type)
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", params).fetchone()["n"]
        verified = conn.execute(
            f"SELECT COUNT(*) AS n {base} AND true_label IS NOT NULL", params
        ).fetchone()["n"]
        avg_confidence = conn.execute(f"SELECT AVG(confidence) AS a {base}", params).fetchone()["a"]
        pending_verification = conn.execute(
            f"SELECT COUNT(*) AS n {base} AND needs_verification = 1 AND true_label IS NULL", params
        ).fetchone()["n"]

    return {
        "dataset_type": dataset_type or "all",
        "total_predictions": total,
        "verified": verified,
        "unverified": total - verified,
        "pending_verification": pending_verification,
        "avg_confidence": round(avg_confidence, 4) if avg_confidence is not None else None,
    }
