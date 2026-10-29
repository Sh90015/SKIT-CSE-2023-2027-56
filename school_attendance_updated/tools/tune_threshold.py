"""Pick the best MATCH_THRESHOLD from the face data already stored in attendance.db.

Run after enrolling students:   python tools/tune_threshold.py [--db attendance.db] [--class-id 1]

For every stored face photo it checks
  * genuine score  : photo vs the AVERAGE of the same student's OTHER photos
  * impostor score : photo vs the average face of every OTHER student
then sweeps thresholds and prints
  * FRR  (false reject: the right student not recognised)
  * FAR  (false accept: a wrong student marked present)
and suggests the lowest threshold that keeps false accepts under --max-far (default 1%).
It does not need the face models, only the saved embeddings, so it also runs on a laptop.
"""
import argparse
import os
import sqlite3

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(BASE, "attendance.db"))
    ap.add_argument("--class-id", type=int)
    ap.add_argument("--max-far", type=float, default=1.0, help="allowed false-accept rate in %% (default 1)")
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    q = "SELECT e.student_id, e.vector FROM face_embeddings e JOIN students s ON s.id=e.student_id"
    args = ()
    if a.class_id:
        q += " WHERE s.class_id=?"
        args = (a.class_id,)
    per = {}
    for sid, blob in db.execute(q, args):
        per.setdefault(sid, []).append(unit(np.frombuffer(blob, np.float32)))
    per = {k: np.array(v) for k, v in per.items() if len(v) >= 3}
    if len(per) < 2:
        raise SystemExit("Need at least 2 students with 3+ face photos each.")

    means = {k: unit(v.mean(axis=0)) for k, v in per.items()}
    genuine, impostor = [], []
    for sid, vecs in per.items():
        n = len(vecs)
        total = vecs.sum(axis=0)
        for v in vecs:
            loo = unit((total - v) / (n - 1))
            genuine.append(float(v @ loo))
            impostor += [float(v @ m) for o, m in means.items() if o != sid]
    g, i = np.array(genuine), np.array(impostor)

    print(f"{len(per)} students, {len(g)} genuine comparisons, {len(i)} impostor comparisons\n")
    print(" threshold |  FRR %  |  FAR %")
    best = None
    eer = (9e9, 0)
    for t in np.arange(0.20, 0.801, 0.02):
        frr, far = 100 * float((g < t).mean()), 100 * float((i >= t).mean())
        print(f"   {t:.2f}    | {frr:6.2f}  | {far:6.2f}")
        if far <= a.max_far and best is None:
            best = (t, frr, far)
        if abs(frr - far) < eer[0]:
            eer = (abs(frr - far), t)
    print()
    if best:
        print(f"Suggested MATCH_THRESHOLD = {best[0]:.2f}  (FRR {best[1]:.2f}%, FAR {best[2]:.2f}%)")
    else:
        print("No threshold reaches the FAR limit. Capture more / clearer photos.")
    print(f"Equal-error point is near {eer[1]:.2f}. Put your chosen value in app.py (MATCH_THRESHOLD).")
    print("Tip: wrong student marked -> choose a higher value. Right student missed -> lower it.")


if __name__ == "__main__":
    main()
