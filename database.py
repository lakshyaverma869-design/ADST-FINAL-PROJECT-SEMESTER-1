"""SQLite database setup, schema migration, and short-lived API response cache."""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from config import DATABASE_PATH


@contextmanager
def connect(database_path=None):
    path = database_path or DATABASE_PATH
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(database_path=None):
    path = Path(database_path or DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        old_columns = []
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone()
        if table:
            old_columns = [row["name"] for row in connection.execute("PRAGMA table_info(predictions)")]

        legacy_rows = []
        if old_columns and "prediction_mode" not in old_columns:
            connection.execute("ALTER TABLE predictions RENAME TO predictions_legacy_v1")
            legacy_rows = [dict(row) for row in connection.execute("SELECT * FROM predictions_legacy_v1")]

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                city TEXT NOT NULL,
                prediction_mode TEXT NOT NULL,
                forecast_for TEXT,
                input_data_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                predicted_demand REAL NOT NULL,
                traffic_status TEXT,
                traffic_volume REAL,
                average_speed REAL,
                temperature REAL,
                rainfall REAL,
                relative_humidity REAL,
                weather_condition TEXT,
                hour INTEGER,
                day_of_week INTEGER,
                model_version TEXT
            )
            """
        )
        if legacy_rows:
            for row in legacy_rows:
                input_values = {
                    key: row.get(key)
                    for key in (
                        "hour", "day_of_week", "traffic_volume", "average_speed",
                        "temperature", "rainfall", "weather_condition", "ev_count"
                    )
                }
                connection.execute(
                    """
                    INSERT INTO predictions (
                        id, timestamp, city, prediction_mode, input_data_json,
                        sources_json, predicted_demand, traffic_volume, average_speed,
                        temperature, rainfall, weather_condition, hour, day_of_week,
                        model_version
                    ) VALUES (?, ?, ?, 'legacy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'legacy-model')
                    """,
                    (
                        row["id"], row["timestamp"], row["city"],
                        json.dumps(input_values), json.dumps({"legacy": "original project database"}),
                        row["predicted_demand"], row["traffic_volume"], row["average_speed"],
                        row["temperature"], row["rainfall"], row["weather_condition"],
                        row["hour"], row["day_of_week"],
                    ),
                )
            connection.execute("DROP TABLE predictions_legacy_v1")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_cache (
                city TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                source TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (city, cache_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_cache (
                city TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                source TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (city, cache_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                rows_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                chart_prefix TEXT NOT NULL
            )
            """
        )


def get_cache(table, city, cache_key, database_path=None, allow_expired=False):
    if table not in {"weather_cache", "traffic_cache"}:
        raise ValueError("Unsupported cache table.")
    initialize_database(database_path)
    with connect(database_path) as connection:
        row = connection.execute(
            f"SELECT * FROM {table} WHERE city=? AND cache_key=?", (city, cache_key)
        ).fetchone()
    if row is None:
        return None
    expires = datetime.fromisoformat(row["expires_at"])
    now = datetime.now(timezone.utc)
    if not allow_expired and expires <= now:
        return None
    return {
        "city": row["city"],
        "cache_key": row["cache_key"],
        "source": row["source"],
        "retrieved_at": row["retrieved_at"],
        "expires_at": row["expires_at"],
        "payload": json.loads(row["payload_json"]),
        "stale": expires <= now,
    }


def set_cache(table, city, cache_key, source, payload, ttl_seconds, database_path=None):
    if table not in {"weather_cache", "traffic_cache"}:
        raise ValueError("Unsupported cache table.")
    initialize_database(database_path)
    now = datetime.now(timezone.utc)
    retrieved = now.isoformat(timespec="seconds")
    expires = (now + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
    with connect(database_path) as connection:
        connection.execute(
            f"""
            INSERT INTO {table} (city, cache_key, source, retrieved_at, expires_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(city, cache_key) DO UPDATE SET
                source=excluded.source,
                retrieved_at=excluded.retrieved_at,
                expires_at=excluded.expires_at,
                payload_json=excluded.payload_json
            """,
            (city, cache_key, source, retrieved, expires, json.dumps(payload)),
        )
    return {
        "city": city,
        "cache_key": cache_key,
        "source": source,
        "retrieved_at": retrieved,
        "expires_at": expires,
        "payload": payload,
        "stale": False,
    }
