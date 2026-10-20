import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "kite.db", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, data TEXT, updated REAL, PRIMARY KEY(kind,id))"
        )
        self.db.commit()

    def put(self, kind, value):
        value = {
            **value,
            "id": value.get("id", uuid.uuid4().hex),
            "updated": time.time(),
        }
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                (
                    kind,
                    value["id"],
                    json.dumps(value, ensure_ascii=False),
                    value["updated"],
                ),
            )
            self.db.commit()
        return value

    def get(self, kind, identifier):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM records WHERE kind=? AND id=?", (kind, identifier)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind):
        with self.lock:
            rows = self.db.execute(
                "SELECT data FROM records WHERE kind=? ORDER BY updated DESC", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, kind, identifier):
        with self.lock:
            self.db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier))
            self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()
