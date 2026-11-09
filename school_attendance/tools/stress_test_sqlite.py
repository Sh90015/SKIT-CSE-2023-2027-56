"""Stress-test SQLite attendance writes under repeated recognition calls.

    python tools/stress_test_sqlite.py                       (defaults: 8 threads, 40 students, 50 calls each)
    python tools/stress_test_sqlite.py --threads 16 --students 60 --calls 100

It runs on a TEMPORARY database (your real attendance.db is never touched) and needs no face models:
the face detector is replaced by a fake that returns a seeded student's stored face vector.

Test A  raw writes   : many threads INSERT OR IGNORE into `attendance` (30 different days) at the same time
Test B  full API     : many threads call /api/recognize repeatedly (same code path as the camera page)

PASS means: no 'database is locked' errors, no HTTP 500, no duplicate rows, and exactly one
attendance row per student per day.
"""
import argparse
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
import uuid
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402


def person_vec(i):
    rng = np.random.RandomState(i * 7 + 1)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


def make_db(tmp, n_students):
    """Temp database with 1 teacher, 1 class and n seeded students (10 face vectors each)."""
    A.DB_PATH = os.path.join(tmp, "stress.db")
    A.MODELS_DIR = os.path.join(tmp, "models")
    os.makedirs(A.MODELS_DIR, exist_ok=True)
    A._galleries.clear()
    A._streaks.clear()
    A.init_db()
    db = sqlite3.connect(A.DB_PATH, timeout=30)
    db.execute("INSERT INTO teachers(uid,username,name,password_hash) VALUES (?,?,?,?)",
               (uuid.uuid4().hex, "stress", "Stress", "x"))
    db.execute("INSERT INTO classes(uid,teacher_id,name) VALUES (?,?,?)", (uuid.uuid4().hex, 1, "Stress class"))
    rng = np.random.RandomState(0)
    for i in range(1, n_students + 1):
        sid = db.execute("INSERT INTO students(uid,class_id,roll_no,name) VALUES (?,?,?,?)",
                         (uuid.uuid4().hex, 1, str(i), f"Student {chr(65 + i % 26)}{chr(65 + i // 26 % 26)}")).lastrowid
        for _ in range(10):
            v = person_vec(i) + rng.randn(128).astype(np.float32) * 0.01
            v /= np.linalg.norm(v)
            db.execute("INSERT INTO face_embeddings(student_id,vector,created_at) VALUES (?,?,?)",
                       (sid, v.tobytes(), A.now_iso()))
    db.commit()
    db.close()


def report(name, ok_calls, errors, times, elapsed):
    print(f"  calls {ok_calls + len(errors)}, ok {ok_calls}, errors {len(errors)}, "
          f"{(ok_calls + len(errors)) / elapsed:.0f} calls/s, "
          f"median {statistics.median(times) * 1000:.1f} ms, "
          f"p95 {sorted(times)[int(0.95 * (len(times) - 1))] * 1000:.1f} ms")
    for e in errors[:3]:
        print("   first errors:", e)


def test_raw_writes(threads, n_students, calls):
    """Test A: concurrent INSERT OR IGNORE / UPDATE on the attendance table."""
    print(f"\nTest A  raw SQLite writes: {threads} threads x {calls} writes")
    today = date.today().isoformat()
    errors, times, ok = [], [], [0]
    lock = threading.Lock()
    init = sqlite3.connect(A.DB_PATH, timeout=30)
    days = [(date.today() - timedelta(days=d)).isoformat() for d in range(30)]
    for d in days:
        init.execute("INSERT OR IGNORE INTO sessions(class_id,date) VALUES (1,?)", (d,))
    init.commit()
    init.close()

    def worker(seed):
        r = random.Random(seed)
        db = sqlite3.connect(A.DB_PATH, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        for _ in range(calls):
            sid = r.randint(1, n_students)
            t0 = time.perf_counter()
            try:
                db.execute("INSERT OR IGNORE INTO attendance(student_id,date,time,confidence,source,created_at,synced) "
                           "VALUES (?,?,?,?,'camera',?,0)",
                           (sid, r.choice(days), time.strftime("%H:%M:%S"), round(r.uniform(0.5, 0.95), 4), A.now_iso()))
                db.commit()
                dt = time.perf_counter() - t0
                with lock:
                    ok[0] += 1
                    times.append(dt)
            except sqlite3.Error as e:
                with lock:
                    errors.append(str(e))
                    times.append(time.perf_counter() - t0)
        db.close()

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    report("A", ok[0], errors, times, time.perf_counter() - t0)
    return not errors


def test_api(threads, n_students, calls):
    """Test B: repeated /api/recognize calls from many threads (face detector faked)."""
    print(f"\nTest B  /api/recognize: {threads} threads x {calls} calls")
    today = date.today().isoformat()
    db = sqlite3.connect(A.DB_PATH, timeout=30)
    db.execute("DELETE FROM attendance")
    db.commit()
    db.close()
    with A.app.app_context():
        A.train_class(1)

    current = threading.local()
    A.detect_and_embed = lambda img: [(np.zeros(15), person_vec(current.pid))]
    img = "data:image/jpeg;base64," + __import__("base64").b64encode(
        __import__("cv2").imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tobytes()).decode()

    errors, times, ok = [], [], [0]
    lock = threading.Lock()

    def worker(seed):
        r = random.Random(seed)
        client = A.app.test_client()
        with client.session_transaction() as s:
            s["teacher_id"], s["teacher_name"] = 1, "Stress"
        for _ in range(calls):
            current.pid = r.randint(1, n_students)
            t0 = time.perf_counter()
            resp = client.post("/api/recognize/1", json={"image": img})
            dt = time.perf_counter() - t0
            with lock:
                times.append(dt)
                if resp.status_code == 200:
                    ok[0] += 1
                else:
                    errors.append(f"HTTP {resp.status_code}: {resp.get_data(as_text=True)[:120]}")

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(100 + i,)) for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    report("B", ok[0], errors, times, time.perf_counter() - t0)
    return not errors, today


def verify(today):
    """Final data checks on the temp database."""
    db = sqlite3.connect(A.DB_PATH, timeout=30)
    rows = db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    distinct = db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT student_id,date FROM attendance)").fetchone()[0]
    no_conf = db.execute("SELECT COUNT(*) FROM attendance WHERE confidence IS NULL").fetchone()[0]
    bad_ts = db.execute("SELECT COUNT(*) FROM attendance WHERE time NOT GLOB '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]'").fetchone()[0]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    print("\nFinal checks (database after Test B)")
    results = [
        (f"one row per student per day ({rows} rows, {distinct} distinct)", rows == distinct),
        ("every row has a confidence score", no_conf == 0),
        ("every row has a HH:MM:SS timestamp", bad_ts == 0),
        (f"PRAGMA integrity_check = {integrity}", integrity == "ok"),
        (f"journal mode = {mode} (WAL lets sync read while the app writes)", mode == "wal"),
    ]
    for text, good in results:
        print(f"  {'PASS' if good else 'FAIL'}  {text}")
    return all(g for _, g in results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--students", type=int, default=40)
    ap.add_argument("--calls", type=int, default=50, help="calls per thread")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        make_db(tmp, a.students)
        a_ok = test_raw_writes(a.threads, a.students, a.calls * 4)
        b_ok, today = test_api(a.threads, a.students, a.calls)
        v_ok = verify(today)
    print("\nRESULT:", "PASS - SQLite handled the load" if (a_ok and b_ok and v_ok) else "FAIL - see above")
    sys.exit(0 if (a_ok and b_ok and v_ok) else 1)


if __name__ == "__main__":
    main()
