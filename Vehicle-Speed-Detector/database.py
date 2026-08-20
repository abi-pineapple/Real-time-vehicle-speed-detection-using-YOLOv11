import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "traffic.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_id INTEGER,
        vehicle_type TEXT,
        max_speed REAL,
        average_speed REAL,
        timestamp TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id INTEGER,
        vehicle_type TEXT,
        speed REAL,
        speed_limit REAL,
        timestamp TEXT,
        snapshot TEXT
    )""")
    conn.commit()
    conn.close()


def upsert_vehicle(tracking_id, vehicle_type, speed):
    """Insert a vehicle or update its running max/average speed."""
    conn = get_conn()
    c = conn.cursor()
    row = c.execute("SELECT * FROM vehicles WHERE tracking_id=?", (tracking_id,)).fetchone()
    if row is None:
        c.execute(
            "INSERT INTO vehicles (tracking_id, vehicle_type, max_speed, average_speed, timestamp) "
            "VALUES (?,?,?,?,?)",
            (tracking_id, vehicle_type, speed, speed, datetime.now().isoformat(timespec="seconds")),
        )
    else:
        new_max = max(row["max_speed"], speed)
        new_avg = (row["average_speed"] + speed) / 2
        c.execute(
            "UPDATE vehicles SET max_speed=?, average_speed=? WHERE tracking_id=?",
            (new_max, new_avg, tracking_id),
        )
    conn.commit()
    conn.close()


def insert_violation(tracking_id, vehicle_type, speed, speed_limit, snapshot_path):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO violations (vehicle_id, vehicle_type, speed, speed_limit, timestamp, snapshot) "
        "VALUES (?,?,?,?,?,?)",
        (tracking_id, vehicle_type, speed, speed_limit, datetime.now().isoformat(timespec="seconds"), snapshot_path),
    )
    conn.commit()
    conn.close()


def has_violation(tracking_id):
    """One violation record per vehicle tracking session."""
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM violations WHERE vehicle_id=?", (tracking_id,)).fetchone()
    conn.close()
    return row is not None


def get_all_vehicles():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM vehicles ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_violations():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM violations ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()
    total_vehicles = conn.execute("SELECT COUNT(*) c FROM vehicles").fetchone()["c"]
    speeding = conn.execute("SELECT COUNT(DISTINCT vehicle_id) c FROM violations").fetchone()["c"]
    avg_speed = conn.execute("SELECT AVG(average_speed) a FROM vehicles").fetchone()["a"] or 0
    max_speed = conn.execute("SELECT MAX(max_speed) m FROM vehicles").fetchone()["m"] or 0
    by_type = conn.execute(
        "SELECT vehicle_type, COUNT(*) c FROM vehicles GROUP BY vehicle_type"
    ).fetchall()
    violations_by_type = conn.execute(
        "SELECT vehicle_type, COUNT(*) c FROM violations GROUP BY vehicle_type"
    ).fetchall()
    conn.close()
    return {
        "total_vehicles": total_vehicles,
        "speeding_vehicles": speeding,
        "average_speed": round(avg_speed, 1),
        "max_speed": round(max_speed, 1),
        "vehicles_by_type": {r["vehicle_type"]: r["c"] for r in by_type},
        "violations_by_type": {r["vehicle_type"]: r["c"] for r in violations_by_type},
    }


def clear_session():
    """Wipe data for a fresh detection session (keeps schema)."""
    conn = get_conn()
    conn.execute("DELETE FROM vehicles")
    conn.execute("DELETE FROM violations")
    conn.commit()
    conn.close()
