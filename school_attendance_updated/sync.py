"""Offline-first sync: local SQLite  ->  cloud PostgreSQL.

* The app always writes to the local SQLite file first, so attendance works with no internet.
* Every attendance row has `synced = 0` until it has been pushed to the cloud AND read back
  from the cloud to verify it arrived.
* Set the environment variable DATABASE_URL (e.g. the Render / Railway PostgreSQL URL) to turn
  cloud sync on. Without it the app runs "local only" and nothing breaks.
* Every failed attempt is written to the `sync_log` table and to data/sync.log.
"""
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime

log = logging.getLogger("attendance.sync")

LOCAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_deletes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,            -- 'student' or 'attendance'
    key1 TEXT NOT NULL,            -- student uid
    key2 TEXT                      -- date (attendance only)
);
CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,            -- ok | offline | error | conflict
    message TEXT NOT NULL,
    pushed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# Written so that the same SQL runs on PostgreSQL (and on SQLite for the automated tests).
CLOUD_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS teachers (
        uid TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        name TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS classes (
        uid TEXT PRIMARY KEY,
        teacher_uid TEXT NOT NULL REFERENCES teachers(uid) ON DELETE CASCADE,
        name TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS students (
        uid TEXT PRIMARY KEY,
        class_uid TEXT NOT NULL REFERENCES classes(uid) ON DELETE CASCADE,
        roll_no TEXT NOT NULL,
        name TEXT NOT NULL,
        UNIQUE (class_uid, roll_no))""",
    """CREATE TABLE IF NOT EXISTS sessions (
        class_uid TEXT NOT NULL REFERENCES classes(uid) ON DELETE CASCADE,
        date TEXT NOT NULL,
        PRIMARY KEY (class_uid, date))""",
    """CREATE TABLE IF NOT EXISTS attendance_log (
        student_uid TEXT NOT NULL REFERENCES students(uid) ON DELETE CASCADE,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        confidence REAL,
        source TEXT NOT NULL,
        marked_at TEXT,
        device_id TEXT,
        synced_at TEXT,
        PRIMARY KEY (student_uid, date))""",
]

_lock = threading.Lock()
_last_fail_logged = {"kind": None, "at": 0.0}


def cloud_url():
    return os.environ.get("DATABASE_URL", "").strip()


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ small helpers

def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT INTO meta(key,value) VALUES (?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def write_log(db, kind, message, pushed=0):
    db.execute("INSERT INTO sync_log(at,kind,message,pushed) VALUES (?,?,?,?)",
               (_now(), kind, message[:500], pushed))
    db.execute("DELETE FROM sync_log WHERE id <= (SELECT MAX(id) FROM sync_log) - 500")
    db.commit()


def _connect_cloud(url):
    try:
        import psycopg2
    except ImportError:
        psycopg2 = None
    if psycopg2 is not None:
        return psycopg2.connect(url, connect_timeout=8)
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("PostgreSQL driver missing. Run: pip install psycopg2-binary")
    return psycopg.connect(url, connect_timeout=8)


def _is_offline(exc):
    return isinstance(exc, OSError) or type(exc).__name__ in ("OperationalError", "InterfaceError")


def _is_integrity(exc):
    return type(exc).__name__ in ("IntegrityError", "UniqueViolation", "ForeignKeyViolation")


def _first_line(exc):
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:300]


def _fail(local, exc):
    kind = "offline" if _is_offline(exc) else "error"
    msg = _first_line(exc)
    set_meta(local, "last_state", kind)
    set_meta(local, "last_error", msg)
    local.commit()
    # Do not flood the log when the internet is simply down: log on change, then every 15 min.
    now = time.time()
    if _last_fail_logged["kind"] != kind or now - _last_fail_logged["at"] > 900:
        write_log(local, kind, msg)
        _last_fail_logged.update(kind=kind, at=now)
        log.warning("sync failed (%s): %s", kind, msg)
    return dict(ok=False, state=kind, pushed=0, message=msg)


# ------------------------------------------------------------------ public API

def pending_count(db_path):
    db = sqlite3.connect(db_path, timeout=30)
    try:
        n = db.execute("SELECT COUNT(*) FROM attendance WHERE synced=0").fetchone()[0]
        n += db.execute("SELECT COUNT(*) FROM pending_deletes").fetchone()[0]
        return n
    finally:
        db.close()


def status(db_path):
    db = sqlite3.connect(db_path, timeout=30)
    try:
        pending = db.execute("SELECT COUNT(*) FROM attendance WHERE synced=0").fetchone()[0]
        pending += db.execute("SELECT COUNT(*) FROM pending_deletes").fetchone()[0]
        last_state = get_meta(db, "last_state")
        configured = bool(cloud_url())
        if not configured:
            state = "not_configured"
        elif last_state in ("offline", "error"):
            state = last_state
        elif pending:
            state = "pending"
        else:
            state = "synced"
        return dict(state=state, configured=configured, pending=pending,
                    last_sync_at=get_meta(db, "last_sync_at"),
                    last_error=get_meta(db, "last_error") if state in ("offline", "error") else None)
    finally:
        db.close()


def recent_log(db_path, limit=30):
    db = sqlite3.connect(db_path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(
            "SELECT at, kind, message, pushed FROM sync_log ORDER BY id DESC LIMIT ?", (limit,))]
    finally:
        db.close()


def sync_now(db_path, device_id="device"):
    """Push everything pending to the cloud. Safe to call any time; never raises."""
    if not cloud_url():
        return dict(ok=False, state="not_configured", pushed=0,
                    message="Cloud database is not set up (DATABASE_URL is empty). Data is kept locally.")
    if not _lock.acquire(blocking=False):
        return dict(ok=False, state="busy", pushed=0, message="A sync is already running.")
    local = cloud = None
    try:
        local = sqlite3.connect(db_path, timeout=30)
        local.row_factory = sqlite3.Row
        try:
            cloud = _connect_cloud(cloud_url())
        except Exception as e:  # no internet / wrong URL / driver missing
            return _fail(local, e)
        try:
            return _run(local, cloud, device_id)
        except Exception as e:
            try:
                cloud.rollback()
            except Exception:
                pass
            return _fail(local, e)
    finally:
        for conn in (cloud, local):
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        _lock.release()


# ------------------------------------------------------------------ the sync itself

def _run(local, cloud, device_id):
    cur = cloud.cursor()
    for stmt in CLOUD_SCHEMA:
        cur.execute(stmt)
    cloud.commit()

    # 1. deletions made while offline
    dels = local.execute("SELECT * FROM pending_deletes ORDER BY id").fetchall()
    for d in dels:
        if d["kind"] == "student":
            cur.execute("DELETE FROM students WHERE uid=%s", (d["key1"],))
        elif d["kind"] == "attendance":
            cur.execute("DELETE FROM attendance_log WHERE student_uid=%s AND date=%s",
                        (d["key1"], d["key2"]))
    cloud.commit()
    if dels:
        local.executemany("DELETE FROM pending_deletes WHERE id=?", [(d["id"],) for d in dels])
        local.commit()

    # 2. profiles (teachers -> classes -> students -> sessions). Idempotent upserts.
    for t in local.execute("SELECT uid, username, name FROM teachers"):
        cur.execute("INSERT INTO teachers(uid,username,name) VALUES (%s,%s,%s) "
                    "ON CONFLICT (uid) DO UPDATE SET username=EXCLUDED.username, name=EXCLUDED.name",
                    (t["uid"], t["username"], t["name"]))
    for c in local.execute("SELECT c.uid, t.uid AS teacher_uid, c.name FROM classes c "
                           "JOIN teachers t ON t.id=c.teacher_id"):
        cur.execute("INSERT INTO classes(uid,teacher_uid,name) VALUES (%s,%s,%s) "
                    "ON CONFLICT (uid) DO UPDATE SET teacher_uid=EXCLUDED.teacher_uid, name=EXCLUDED.name",
                    (c["uid"], c["teacher_uid"], c["name"]))
    cloud.commit()

    bad_students, conflicts = set(), 0
    for s in local.execute("SELECT s.uid, c.uid AS class_uid, s.roll_no, s.name FROM students s "
                           "JOIN classes c ON c.id=s.class_id"):
        cur.execute("SAVEPOINT stu")
        try:
            cur.execute("INSERT INTO students(uid,class_uid,roll_no,name) VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT (uid) DO UPDATE SET class_uid=EXCLUDED.class_uid, "
                        "roll_no=EXCLUDED.roll_no, name=EXCLUDED.name",
                        (s["uid"], s["class_uid"], s["roll_no"], s["name"]))
            cur.execute("RELEASE SAVEPOINT stu")
        except Exception as e:
            if not _is_integrity(e):
                raise
            cur.execute("ROLLBACK TO SAVEPOINT stu")
            bad_students.add(s["uid"])
            conflicts += 1
            write_log(local, "conflict",
                      f"Roll {s['roll_no']} ({s['name']}) already exists in the cloud under a "
                      f"different student record; skipped.")
    for ss in local.execute("SELECT c.uid AS class_uid, s.date FROM sessions s "
                            "JOIN classes c ON c.id=s.class_id"):
        cur.execute("INSERT INTO sessions(class_uid,date) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                    (ss["class_uid"], ss["date"]))
    cloud.commit()

    # 3. attendance rows that are still pending. Duplicates are ignored (first record wins).
    rows = local.execute(
        "SELECT a.student_id, st.uid AS student_uid, a.date, a.time, a.confidence, a.source, "
        "a.created_at FROM attendance a JOIN students st ON st.id=a.student_id "
        "WHERE a.synced=0 ORDER BY a.created_at").fetchall()
    to_push = [r for r in rows if r["student_uid"] not in bad_students]
    stamp = _now()
    for r in to_push:
        cur.execute("INSERT INTO attendance_log(student_uid,date,time,confidence,source,marked_at,"
                    "device_id,synced_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (student_uid,date) DO NOTHING",
                    (r["student_uid"], r["date"], r["time"], r["confidence"], r["source"],
                     r["created_at"], device_id, stamp))
    cloud.commit()

    # 4. verify: read the rows back from the cloud before calling them "synced"
    present = set()
    uids = sorted({r["student_uid"] for r in to_push})
    for i in range(0, len(uids), 400):
        chunk = uids[i:i + 400]
        cur.execute("SELECT student_uid, date FROM attendance_log WHERE student_uid IN (%s)"
                    % ",".join(["%s"] * len(chunk)), tuple(chunk))
        present.update((a[0], a[1]) for a in cur.fetchall())
    verified = [r for r in to_push if (r["student_uid"], r["date"]) in present]
    if verified:
        local.executemany("UPDATE attendance SET synced=1 WHERE student_id=? AND date=?",
                          [(r["student_id"], r["date"]) for r in verified])
    missing = len(to_push) - len(verified)

    set_meta(local, "last_state", "synced" if not missing else "error")
    if not missing:
        set_meta(local, "last_sync_at", _now())
        set_meta(local, "last_error", "")
    local.commit()
    _last_fail_logged.update(kind=None, at=0.0)

    if missing:
        msg = f"{missing} record(s) could not be verified in the cloud and will be retried."
        set_meta(local, "last_error", msg)
        write_log(local, "error", msg, len(verified))
        return dict(ok=False, state="error", pushed=len(verified), message=msg)
    if verified or dels or conflicts:
        write_log(local, "ok", f"Synced {len(verified)} attendance record(s), "
                               f"{len(dels)} deletion(s), verified in cloud.", len(verified))
    log.info("sync ok: %d records", len(verified))
    return dict(ok=True, state="synced", pushed=len(verified),
                message=f"Synced {len(verified)} record(s). Cloud copy verified.")


# ------------------------------------------------------------------ background auto-sync

def start_background(db_path, device_id, interval=60, full_every=300):
    """Retry automatically: every `interval` s if something is pending, else every `full_every` s."""
    def loop():
        last_full = 0.0
        while True:
            time.sleep(interval)
            try:
                if not cloud_url():
                    continue
                if pending_count(db_path) or time.time() - last_full > full_every:
                    sync_now(db_path, device_id)
                    last_full = time.time()
            except Exception:
                log.exception("background sync crashed")
    threading.Thread(target=loop, daemon=True, name="auto-sync").start()
