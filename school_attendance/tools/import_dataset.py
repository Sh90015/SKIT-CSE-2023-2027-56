"""Bulk-import a folder of face photos (your ~500-image dataset) into the app.

Folder layout (one sub-folder per student, named  <roll>_<name>):
    dataset/
        1_Asha Verma/   img001.jpg img002.jpg ...
        2_Ravi Kumar/   ...

Run:   python tools/import_dataset.py dataset --class-id 1
Photos with no face, several faces or a tiny face are skipped and counted.
Afterwards press "Train faces" on the class page.
Needs the face models (first run of `python app.py` downloads them).
"""
import argparse
import os
import sqlite3
import sys
import uuid
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--class-id", type=int, required=True)
    a = ap.parse_args()

    A.ensure_models()
    db = sqlite3.connect(A.DB_PATH, timeout=30)
    db.execute("PRAGMA foreign_keys=ON")
    if db.execute("SELECT 1 FROM classes WHERE id=?", (a.class_id,)).fetchone() is None:
        raise SystemExit(f"No class with id {a.class_id}. Create it in the app first.")

    tot = dict(ok=0, noface=0, multi=0, other=0)
    for folder in sorted(os.listdir(a.folder)):
        path = os.path.join(a.folder, folder)
        if not os.path.isdir(path) or "_" not in folder:
            continue
        roll, name = folder.split("_", 1)
        roll, name, err = A.clean_student_fields(roll, name.replace("_", " "))
        if err:
            print(f"skip folder '{folder}': {err}")
            continue
        row = db.execute("SELECT id FROM students WHERE class_id=? AND roll_no=?", (a.class_id, roll)).fetchone()
        sid = row[0] if row else db.execute(
            "INSERT INTO students(uid,class_id,roll_no,name) VALUES (?,?,?,?)",
            (uuid.uuid4().hex, a.class_id, roll, name)).lastrowid
        have = db.execute("SELECT COUNT(*) FROM face_embeddings WHERE student_id=?", (sid,)).fetchone()[0]
        c = dict(ok=0, noface=0, multi=0)
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith(EXT) or have + c["ok"] >= A.MAX_FACE_IMAGES:
                continue
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                tot["other"] += 1
                continue
            s = A.MAX_IMAGE_SIDE / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            found = A.detect_and_embed(img)
            if not found:
                c["noface"] += 1
            elif len(found) > 1:
                c["multi"] += 1
            else:
                db.execute("INSERT INTO face_embeddings(student_id,vector,created_at) VALUES (?,?,?)",
                           (sid, found[0][1].astype(np.float32).tobytes(), datetime.now().isoformat(timespec="seconds")))
                c["ok"] += 1
        db.commit()
        for k in c:
            tot[k] += c[k]
        print(f"Roll {roll:>4} {name:<25} added {c['ok']:>3}  (no face {c['noface']}, several faces {c['multi']})")
    print(f"\nDone: {tot['ok']} photos imported, {tot['noface']} without a face, {tot['multi']} with several faces.")
    print("Now open the class page and press 'Train faces'.")


if __name__ == "__main__":
    main()
