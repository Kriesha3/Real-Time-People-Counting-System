
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_count_history.db"


class Database:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=10
        )
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        """Create tables and indexes without deleting existing data."""
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    tracking_id TEXT,
                    entries INTEGER NOT NULL DEFAULT 0,
                    exits INTEGER NOT NULL DEFAULT 0,
                    people_inside INTEGER NOT NULL DEFAULT 0,
                    note TEXT DEFAULT ''
                )
            """)

            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp)
            """)

            db.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                )
            """)

            db.execute("""
                INSERT OR IGNORE INTO settings(
                    setting_key, setting_value
                )
                VALUES ('crowd_limit', '10')
            """)

    # ---------------- SETTINGS ----------------

    def get_setting(self, key, default=None):
        with self.connect() as db:
            row = db.execute("""
                SELECT setting_value
                FROM settings
                WHERE setting_key = ?
            """, (key,)).fetchone()

        return row["setting_value"] if row else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute("""
                INSERT INTO settings(setting_key, setting_value)
                VALUES (?, ?)
                ON CONFLICT(setting_key)
                DO UPDATE SET setting_value = excluded.setting_value
            """, (key, str(value)))

    def get_crowd_limit(self):
        try:
            return max(1, int(self.get_setting("crowd_limit", "10")))
        except (TypeError, ValueError):
            return 10

    def set_crowd_limit(self, limit):
        limit = max(1, int(limit))
        self.set_setting("crowd_limit", limit)
        return limit

    # ---------------- EVENT HISTORY ----------------

    def add_event(
        self,
        event_type,
        tracking_id,
        entries,
        exits,
        people_inside,
        note=""
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.connect() as db:
            cursor = db.execute("""
                INSERT INTO events(
                    timestamp,
                    event_type,
                    tracking_id,
                    entries,
                    exits,
                    people_inside,
                    note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                event_type,
                "" if tracking_id is None else str(tracking_id),
                int(entries),
                int(exits),
                max(0, int(people_inside)),
                note
            ))

            return cursor.lastrowid

    def get_events(
        self,
        start_date=None,
        end_date=None,
        event_type=None,
        search="",
        limit=None
    ):
        sql = "SELECT * FROM events WHERE 1=1"
        params = []

        if start_date:
            sql += " AND timestamp >= ?"
            params.append(start_date + " 00:00:00")

        if end_date:
            next_day = (
                date.fromisoformat(end_date) + timedelta(days=1)
            ).isoformat()
            sql += " AND timestamp < ?"
            params.append(next_day + " 00:00:00")

        if event_type and event_type != "ALL":
            sql += " AND event_type = ?"
            params.append(event_type)

        if search:
            term = f"%{search}%"
            sql += """
                AND (
                    timestamp LIKE ?
                    OR event_type LIKE ?
                    OR tracking_id LIKE ?
                    OR note LIKE ?
                )
            """
            params.extend([term] * 4)

        sql += " ORDER BY id DESC"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))

        with self.connect() as db:
            return db.execute(sql, params).fetchall()

    # ---------------- DAILY REPORT ----------------

    def daily_summary(self, selected_date):
        day = date.fromisoformat(selected_date)
        next_day = (day + timedelta(days=1)).isoformat()

        with self.connect() as db:
            row = db.execute("""
                SELECT
                    SUM(
                        CASE WHEN event_type = 'ENTRY'
                        THEN 1 ELSE 0 END
                    ) AS entries,
                    SUM(
                        CASE WHEN event_type = 'EXIT'
                        THEN 1 ELSE 0 END
                    ) AS exits,
                    MAX(people_inside) AS peak
                FROM events
                WHERE timestamp >= ? AND timestamp < ?
            """, (
                selected_date + " 00:00:00",
                next_day + " 00:00:00"
            )).fetchone()

        return {
            "entries": row["entries"] or 0,
            "exits": row["exits"] or 0,
            "peak": row["peak"] or 0
        }

    # ---------------- HOURLY CHART ----------------

    def hourly_summary(self, selected_date):
        entries = [0] * 24
        exits = [0] * 24

        with self.connect() as db:
            rows = db.execute("""
                SELECT
                    CAST(substr(timestamp, 12, 2) AS INTEGER) AS hour,
                    event_type,
                    COUNT(*) AS total
                FROM events
                WHERE timestamp >= ? AND timestamp < ?
                  AND event_type IN ('ENTRY', 'EXIT')
                GROUP BY hour, event_type
            """, (
                selected_date + " 00:00:00",
                (
                    date.fromisoformat(selected_date)
                    + timedelta(days=1)
                ).isoformat() + " 00:00:00"
            )).fetchall()

        for row in rows:
            hour = row["hour"]

            if 0 <= hour < 24:
                if row["event_type"] == "ENTRY":
                    entries[hour] = row["total"]
                elif row["event_type"] == "EXIT":
                    exits[hour] = row["total"]

        return entries, exits

    # ---------------- OCCUPANCY ANALYTICS ----------------

    def peak_occupancy(self, selected_date=None):
        """Return the highest recorded occupancy."""
        sql = """
            SELECT MAX(people_inside) AS peak
            FROM events
            WHERE 1=1
        """
        params = []

        if selected_date:
            day = date.fromisoformat(selected_date)
            next_day = (day + timedelta(days=1)).isoformat()

            sql += " AND timestamp >= ? AND timestamp < ?"
            params.extend([
                selected_date + " 00:00:00",
                next_day + " 00:00:00"
            ])

        with self.connect() as db:
            row = db.execute(sql, params).fetchone()

        return row["peak"] or 0

    def busiest_hour(self, selected_date):
        """Return the hour with the most entries and its entry count."""
        entries, _ = self.hourly_summary(selected_date)

        if not any(entries):
            return None, 0

        hour = max(range(24), key=lambda h: entries[h])
        return hour, entries[hour]

    # ---------------- SAFE COUNTER RESET ----------------

    def record_reset(self, entries=0, exits=0, note="Counters reset"):
        """
        Save a reset event for the audit trail.
        Does not delete existing event history.
        """
        return self.add_event(
            event_type="RESET",
            tracking_id=None,
            entries=entries,
            exits=exits,
            people_inside=0,
            note=note
        )
