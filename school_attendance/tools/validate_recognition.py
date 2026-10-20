"""Validate recognition results against the seeded student records + check the attendance log.

    python tools/validate_recognition.py --class-id 1
    python tools/validate_recognition.py --class-id 1 --photos test_photos/     (optional, needs face models)

1. Leave-one-out test (no models needed): every stored face photo of a seeded student is used as a
   "probe" and matched against all the OTHER stored photos, using exactly the same rule as the app
   (average of the top-3 scores, MATCH_THRESHOLD and MATCH_MARGIN). Result for each probe:
     correct    -> matched the right student
     wrong      -> matched a different student  (false accept, the dangerous one)
     unmatched  -> not accepted                 (false reject)
2. --photos: fresh photos in folders named <roll>_<name> (same layout as import_dataset.py). Each is
   recognised against the trained gallery and compared with the roll number in the folder name.
3. Attendance-log integrity: confidence / time / date / student-id rules on the `attendance` table.

A per-student CSV is saved to data/validation_report.csv (attach it to the weekly report).
"""
import argparse
import csv
import os
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REPORT = os.path.join(A.BASE, "data", "validation_report.csv")


def decide(sims, ids):
    """Same rule as /api/recognize. sims/ids = similarity and student id of every gallery photo.
    Returns (student_id or None, best_score)."""
    scores = {}
    for sid in np.unique(ids):
        s = np.sort(sims[ids == sid])[::-1][:3]
        scores[int(sid)] = float(s.mean())
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    if best < A.MATCH_THRESHOLD or (best - second) < A.MATCH_MARGIN:
        return None, best
    return best_id, best


def load_class(db, class_id):
    rows = db.execute(
        "SELECT f.student_id, f.vector FROM face_embeddings f JOIN students s ON s.id=f.student_id "
        "WHERE s.class_id=? ORDER BY f.id", (class_id,)).fetchall()
    X = np.array([np.frombuffer(r[1], np.float32) for r in rows], np.float32).reshape(-1, 128)
    y = np.array([r[0] for r in rows])
    return X, y


def leave_one_out(X, y):
    """{student_id: [correct, wrong, unmatched]}"""
    res = {int(s): [0, 0, 0] for s in np.unique(y)}
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        if len(np.unique(y[keep])) < 2:
            continue
        got, _ = decide(X[keep] @ X[i], y[keep])
        k = 0 if got == y[i] else (2 if got is None else 1)
        res[int(y[i])][k] += 1
    return res


def photo_test(db, class_id, folder, X, y):
    """{student_id: [correct, wrong, unmatched]} using fresh photos; also counts skipped photos."""
    import cv2
    A.ensure_models()
    by_roll = {r[1]: r[0] for r in db.execute("SELECT id, roll_no FROM students WHERE class_id=?", (class_id,))}
    res, skipped = {}, 0
    for d in sorted(os.listdir(folder)):
        path = os.path.join(folder, d)
        roll = d.split("_", 1)[0]
        if not os.path.isdir(path) or roll not in by_roll:
            continue
        sid = by_roll[roll]
        res.setdefault(sid, [0, 0, 0])
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith(EXT):
                continue
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                skipped += 1
                continue
            s = A.MAX_IMAGE_SIDE / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            found = A.detect_and_embed(img)
            if len(found) != 1:
                skipped += 1
                continue
            got, _ = decide(X @ found[0][1], y)
            res[sid][0 if got == sid else (2 if got is None else 1)] += 1
    return res, skipped


def check_attendance_log(db):
    """Integrity rules for the attendance-log table. Returns list of (rule, bad_row_count)."""
    q = lambda sql: db.execute(sql).fetchone()[0]  # noqa: E731
    checks = [
        ("camera rows must have a confidence score",
         q("SELECT COUNT(*) FROM attendance WHERE source='camera' AND confidence IS NULL")),
        ("confidence must be between 0 and 1",
         q("SELECT COUNT(*) FROM attendance WHERE confidence IS NOT NULL AND (confidence<0 OR confidence>1)")),
        ("every row must point to an existing student",
         q("SELECT COUNT(*) FROM attendance WHERE student_id NOT IN (SELECT id FROM students)")),
        ("date must look like YYYY-MM-DD",
         q("SELECT COUNT(*) FROM attendance WHERE date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'")),
        ("camera rows must have a HH:MM:SS timestamp",
         q("SELECT COUNT(*) FROM attendance WHERE source='camera' AND time NOT GLOB '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]'")),
        ("synced must be 0 or 1",
         q("SELECT COUNT(*) FROM attendance WHERE synced NOT IN (0,1)")),
        ("every attendance date needs a row in sessions",
         q("SELECT COUNT(*) FROM attendance a JOIN students s ON s.id=a.student_id "
           "WHERE NOT EXISTS (SELECT 1 FROM sessions x WHERE x.class_id=s.class_id AND x.date=a.date)")),
    ]
    return checks


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--db", default=A.DB_PATH)
    ap.add_argument("--photos", help="folder of fresh test photos: <roll>_<name>/img.jpg")
    a = ap.parse_args()

    db = sqlite3.connect(a.db, timeout=30)
    if db.execute("SELECT 1 FROM classes WHERE id=?", (a.class_id,)).fetchone() is None:
        raise SystemExit(f"No class with id {a.class_id}.")
    X, y = load_class(db, a.class_id)
    if len(np.unique(y)) < 2:
        raise SystemExit("Need at least 2 students with face data. Run tools/import_dataset.py first.")
    names = {r[0]: (r[1], r[2]) for r in db.execute("SELECT id, roll_no, name FROM students WHERE class_id=?", (a.class_id,))}

    print(f"Class {a.class_id}: {len(np.unique(y))} seeded students, {len(X)} stored face photos")
    print(f"Rule: threshold {A.MATCH_THRESHOLD}, margin {A.MATCH_MARGIN}\n")

    res = leave_one_out(X, y)
    title = "Leave-one-out validation"
    skipped = 0
    if a.photos:
        res, skipped = photo_test(db, a.class_id, a.photos, X, y)
        title = f"Fresh-photo validation ({a.photos})"

    tc = sum(v[0] for v in res.values())
    tw = sum(v[1] for v in res.values())
    tu = sum(v[2] for v in res.values())
    total = tc + tw + tu
    print(f"== {title} ==")
    print(f"Probes {total}: correct {tc} ({pct(tc, total)}), WRONG student {tw} ({pct(tw, total)}), "
          f"not matched {tu} ({pct(tu, total)})" + (f", skipped photos {skipped}" if skipped else ""))

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Roll No", "Name", "Probes", "Correct", "Wrong student", "Not matched", "Accuracy %"])
        weak = []
        for sid, (c, wr, u) in sorted(res.items(), key=lambda kv: names[kv[0]][0].zfill(12)):
            n = c + wr + u
            acc = round(100 * c / n, 1) if n else 0
            w.writerow([names[sid][0], names[sid][1], n, c, wr, u, acc])
            if n and acc < 90:
                weak.append((names[sid][0], names[sid][1], acc))
    print(f"Per-student report saved: {REPORT}")
    if weak:
        print("Students below 90% (capture more / better photos):")
        for roll, name, acc in weak:
            print(f"   Roll {roll:>4} {name:<25} {acc}%")

    print("\n== Attendance-log integrity ==")
    bad_total = 0
    n_rows = db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    print(f"{n_rows} rows in the attendance table")
    for rule, bad in check_attendance_log(db):
        bad_total += bad
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {rule}" + (f"  ({bad} bad rows)" if bad else ""))
    print("\nRESULT:", "all checks passed" if bad_total == 0 and tw == 0 else
          "review the points above (wrong-student matches must be 0: raise MATCH_THRESHOLD if not)")


if __name__ == "__main__":
    main()
