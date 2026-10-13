"""Measure face detection + recognition speed (use this on the low-end device you plan to deploy on).

    python tools/benchmark.py path/to/some/face/photos [--runs 3]

Prints average / median / 95th-percentile milliseconds per photo. For the full browser -> server
-> browser round trip, use the app, then open  /api/health  (shows avg and p95 of /api/recognize).
Rule of thumb for a smooth attendance flow: under ~600 ms per frame.
"""
import argparse
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    files = [os.path.join(r, f) for r, _, fs in os.walk(a.folder) for f in fs
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))][:100]
    if not files:
        raise SystemExit("No images found.")
    A.ensure_models()
    imgs = []
    for f in files:
        im = cv2.imread(f)
        if im is not None:
            s = A.MAX_IMAGE_SIDE / max(im.shape[:2])
            imgs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im)
    A.detect_and_embed(imgs[0])           # warm-up (loads the models)
    times, found = [], 0
    for _ in range(a.runs):
        for im in imgs:
            t = time.perf_counter()
            found += bool(A.detect_and_embed(im))
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    print(f"{len(times)} runs on {len(imgs)} photos; face found in {100 * found / len(times):.0f}%")
    print(f"average {statistics.mean(times):.0f} ms | median {statistics.median(times):.0f} ms | "
          f"95th percentile {times[int(0.95 * (len(times) - 1))]:.0f} ms | ~{1000 / statistics.mean(times):.1f} photos/second")


if __name__ == "__main__":
    main()
