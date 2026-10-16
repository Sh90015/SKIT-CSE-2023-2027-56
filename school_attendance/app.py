import base64
import csv
import io
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.request
import uuid
from collections import deque
from datetime import date, datetime
from functools import wraps
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
from flask import (Flask, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash


BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "attendance.db")
FACES_DIR = os.path.join(BASE, "data", "faces")   # only used to migrate old .npy files
LOG_PATH = os.path.join(BASE, "data", "sync.log")
MODELS_DIR = os.path.join(BASE, "data", "models")
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.42   # cosine similarity needed to accept a face. Wrong student marked? raise to 0.50. Not recognised? lower to 0.36
MATCH_MARGIN = 0.06      # best student must beat the 2nd best by this much (stops look-alike mix-ups)
CONFIRM_FRAMES = 2       # same student must be recognised this many times in a row before marking
DUPLICATE_THRESHOLD = 0.50  # same face registered under two students is blocked above this
MIN_FACE_PX = 70         # ignore faces smaller than this (too far from the camera)
MAX_FACE_IMAGES = 60     # max photos stored per student
LOW_ATTENDANCE = 75      # attendance below this % is shown in red
SIGNUP_CODE = os.environ.get("SIGNUP_CODE", "")  # if set, signup requires this code
MAX_IMAGE_SIDE = 640     # frames larger than this are shrunk before detection (faster on weak servers)

app = Flask(__name__)

_key_file = os.path.join(BASE, "data", "secret.key")
if not os.path.exists(_key_file):
    with open(_key_file, "wb") as f:
        f.write(os.urandom(32))
with open(_key_file, "rb") as f:
    app.secret_key = f.read()

SCHEMA = """
CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT,
    username TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT,
    teacher_id INTEGER NOT NULL REFERENCES teachers(id),
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT,
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    roll_no TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(class_id, roll_no)
);
-- one row per captured face; `vector` is the 128-number face embedding (float32 bytes)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
CREATE TABLE IF NOT EXISTS sessions (
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    PRIMARY KEY (class_id, date)
);
-- the attendance log: who, when, how sure the model was, and whether it reached the cloud yet
CREATE TABLE IF NOT EXISTS attendance (
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    confidence REAL,
    source TEXT NOT NULL DEFAULT 'camera',
    created_at TEXT,
    synced INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (student_id, date)
);
"""


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=30)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _columns(db, table):
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def migrate(db):
    """Upgrade databases created by older versions of this app."""
    for t in ("teachers", "classes", "students"):
        if "uid" not in _columns(db, t):
            db.execute(f"ALTER TABLE {t} ADD COLUMN uid TEXT")
        for (rid,) in db.execute(f"SELECT id FROM {t} WHERE uid IS NULL OR uid=''").fetchall():
            db.execute(f"UPDATE {t} SET uid=? WHERE id=?", (uuid.uuid4().hex, rid))
        db.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{t}_uid ON {t}(uid)")
    have = _columns(db, "attendance")
    for col, ddl in (("confidence", "REAL"), ("source", "TEXT NOT NULL DEFAULT 'camera'"),
                     ("created_at", "TEXT"), ("synced", "INTEGER NOT NULL DEFAULT 0")):
        if col not in have:
            db.execute(f"ALTER TABLE attendance ADD COLUMN {col} {ddl}")
    db.execute("UPDATE attendance SET source='manual' WHERE time='manual' AND source='camera'")
    migrate_embeddings(db)


def migrate_embeddings(db):
    """Older versions kept face data as .npy files; move them into the face_embeddings table."""
    if not os.path.isdir(FACES_DIR):
        return
    moved = 0
    for sid in os.listdir(FACES_DIR):
        d = os.path.join(FACES_DIR, sid)
        if not (sid.isdigit() and os.path.isdir(d)):
            continue
        if db.execute("SELECT 1 FROM students WHERE id=?", (int(sid),)).fetchone() is None:
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npy"):
                continue
            try:
                v = np.load(os.path.join(d, f)).astype(np.float32).reshape(-1)
            except Exception:
                continue
            if v.shape[0] == 128:
                db.execute("INSERT INTO face_embeddings(student_id,vector,created_at) VALUES (?,?,?)",
                           (int(sid), v.tobytes(), now_iso()))
                moved += 1
    db.commit()
    if moved:
        try:
            os.rename(FACES_DIR, FACES_DIR + "_migrated_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        except OSError:
            pass


def init_db():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.execute("PRAGMA journal_mode = WAL")   # lets the background sync read while the app writes
    db.executescript(SCHEMA)
    migrate(db)
    db.commit()
    db.close()


init_db()

# ------------------------------------------------------------------ input checks

ROLL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/]{0,11}$")


def clean_student_fields(roll, name):
    """Return (roll, name, error). Same rules are repeated in the browser for instant feedback."""
    roll, name = (roll or "").strip(), " ".join((name or "").split())
    if not ROLL_RE.match(roll):
        return roll, name, "Roll number must be 1-12 letters/digits (a dash or slash is allowed)."
    if not 2 <= len(name) <= 60 or any(ch.isdigit() for ch in name):
        return roll, name, "Name must be 2-60 characters and cannot contain digits."
    return roll, name, None


def int_arg(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# ------------------------------------------------------------------ auth


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "teacher_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify(error="Please sign in first"), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def own_class(class_id):
    row = get_db().execute(
        "SELECT * FROM classes WHERE id=? AND teacher_id=?",
        (class_id, session["teacher_id"])).fetchone()
    if row is None:
        abort(404)
    return row


def own_student(student_id):
    row = get_db().execute(
        "SELECT s.* FROM students s JOIN classes c ON c.id=s.class_id "
        "WHERE s.id=? AND c.teacher_id=?",
        (student_id, session["teacher_id"])).fetchone()
    if row is None:
        abort(404)
    return row


@app.route("/")
def home():
    return redirect(url_for("dashboard" if "teacher_id" in session else "login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"].strip()
        username = request.form["username"].strip().lower()
        password = request.form["password"]
        if SIGNUP_CODE and request.form.get("code", "") != SIGNUP_CODE:
            flash("Incorrect school code.", "err")
        elif not name or not username or len(password) < 6:
            flash("Enter your name and username, and use a password of at least 6 characters.", "err")
        else:
            try:
                get_db().execute(
                    "INSERT INTO teachers(uid,username,name,password_hash) VALUES (?,?,?,?)",
                    (uuid.uuid4().hex, username, name, generate_password_hash(password)))
                get_db().commit()
                flash("Account created. Please sign in.", "ok")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is already taken.", "err")
    return render_template("signup.html", need_code=bool(SIGNUP_CODE))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        t = get_db().execute("SELECT * FROM teachers WHERE username=?",
                             (request.form["username"].strip().lower(),)).fetchone()
        if t and check_password_hash(t["password_hash"], request.form["password"]):
            session.clear()
            session["teacher_id"] = t["id"]
            session["teacher_name"] = t["name"]
            return redirect(url_for("dashboard"))
        flash("Incorrect username or password.", "err")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------------------------------------------------------------ classes


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    db = get_db()
    if request.method == "POST":
        name = request.form["name"].strip()
        if name:
            db.execute("INSERT INTO classes(uid,teacher_id,name) VALUES (?,?,?)",
                       (uuid.uuid4().hex, session["teacher_id"], name))
            db.commit()
        return redirect(url_for("dashboard"))
    classes = db.execute(
        "SELECT c.*, (SELECT COUNT(*) FROM students s WHERE s.class_id=c.id) AS n, "
        "(SELECT COUNT(*) FROM attendance a JOIN students s ON s.id=a.student_id "
        " WHERE s.class_id=c.id AND a.date=?) AS today "
        "FROM classes c WHERE teacher_id=? ORDER BY c.name",
        (date.today().isoformat(), session["teacher_id"])).fetchall()
    return render_template("dashboard.html", classes=classes)


def face_count(student_id):
    return get_db().execute("SELECT COUNT(*) FROM face_embeddings WHERE student_id=?",
                            (student_id,)).fetchone()[0]


@app.route("/class/<int:class_id>", methods=["GET", "POST"])
@login_required
def class_page(class_id):
    c = own_class(class_id)
    db = get_db()
    if request.method == "POST":
        roll, name, err = clean_student_fields(request.form.get("roll_no"), request.form.get("name"))
        if err:
            flash(err, "err")
        else:
            try:
                db.execute("INSERT INTO students(uid,class_id,roll_no,name) VALUES (?,?,?,?)",
                           (uuid.uuid4().hex, class_id, roll, name))
                db.commit()
                flash(f"{name} added. Now capture their face.", "ok")
            except sqlite3.IntegrityError:
                flash("This roll number already exists in this class.", "err")
        return redirect(url_for("class_page", class_id=class_id))
    students = db.execute("SELECT * FROM students WHERE class_id=? "
                          "ORDER BY CAST(roll_no AS INTEGER), roll_no", (class_id,)).fetchall()
    rows = [dict(s, faces=face_count(s["id"])) for s in students]
    trained = os.path.exists(model_path(class_id))
    return render_template("class.html", c=c, students=rows, trained=trained,
                           target=30)


@app.route("/student/<int:student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    s = own_student(student_id)
    db = get_db()
    db.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
    db.execute("DELETE FROM face_embeddings WHERE student_id=?", (student_id,))
    db.execute("DELETE FROM students WHERE id=?", (student_id,))
    db.commit()
    flash(f"{s['name']} removed. Remember to train faces again.", "ok")
    return redirect(url_for("class_page", class_id=s["class_id"]))

# ------------------------------------------------------------------ face

YUNET_PATH = os.path.join(BASE, "models", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(BASE, "models", "face_recognition_sface_2021dec.onnx")
_cv_lock = threading.Lock()
_detector = None
_embedder = None
_galleries = {}
_streaks = {}


_BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODEL_FILES = {   # path: (download url, minimum plausible size in bytes)
    YUNET_PATH: (_BASE_URL + "face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
    SFACE_PATH: (_BASE_URL + "face_recognition_sface/face_recognition_sface_2021dec.onnx", 10_000_000),
}


def ensure_models():
    """Download the face models on first run if they are missing."""
    os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
    for path, (url, min_size) in MODEL_FILES.items():
        if os.path.exists(path) and os.path.getsize(path) >= min_size:
            continue
        print(f"Downloading {os.path.basename(path)} (one time only, needs internet)...")
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < min_size:
                raise RuntimeError("downloaded file is too small")
            os.replace(tmp, path)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(
                f"Could not download {os.path.basename(path)} ({e}). Download it manually "
                f"from {url} and save it in the 'models' folder.")


def _load_models():
    global _detector, _embedder
    if _detector is None:
        ensure_models()
        _detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.85, 0.3, 5000)
        _embedder = cv2.FaceRecognizerSF.create(SFACE_PATH, "")


def model_path(class_id):
    return os.path.join(MODELS_DIR, f"class_{class_id}.npz")


def decode_image(data_url):
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is not None and max(img.shape[:2]) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def detect_and_embed(img):
    """Return [(face_row, unit_embedding)] for every usable face, biggest first."""
    out = []
    with _cv_lock:
        _load_models()
        h, w = img.shape[:2]
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img)
        if faces is None:
            return out
        for f in faces:
            if f[2] < MIN_FACE_PX or f[3] < MIN_FACE_PX:
                continue
            emb = _embedder.feature(_embedder.alignCrop(img, f)).flatten().astype(np.float32)
            out.append((f, emb / (np.linalg.norm(emb) + 1e-9)))
    out.sort(key=lambda t: -(t[0][2] * t[0][3]))
    return out


def student_embeddings(student_id, limit=None):
    q = "SELECT vector FROM face_embeddings WHERE student_id=? ORDER BY id DESC"
    args = [student_id]
    if limit:
        q += " LIMIT ?"
        args.append(limit)
    rows = get_db().execute(q, args).fetchall()
    return np.array([np.frombuffer(r["vector"], np.float32) for r in rows], np.float32).reshape(-1, 128)


def mean_unit(x):
    m = x.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def train_class(class_id):
    students = get_db().execute("SELECT id, name FROM students WHERE class_id=?",
                                (class_id,)).fetchall()
    xs, ys, means = [], [], []
    for s in students:
        e = student_embeddings(s["id"])
        if len(e):
            xs.append(e)
            ys += [s["id"]] * len(e)
            means.append((s["name"], mean_unit(e)))
    if not xs:
        return 0, 0, []
    np.savez(model_path(class_id), X=np.vstack(xs), y=np.array(ys))
    _galleries.pop(class_id, None)
    warnings = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            if float(means[i][1] @ means[j][1]) >= DUPLICATE_THRESHOLD:
                warnings.append(f"{means[i][0]} and {means[j][0]} look like the same person. "
                                f"Remove one of them and capture again.")
    return len(means), len(ys), warnings


def load_gallery(class_id):
    """(X, y, groups): all embeddings, their student ids, and row indexes grouped per student."""
    p = model_path(class_id)
    if not os.path.exists(p):
        return None
    mtime = os.path.getmtime(p)
    cached = _galleries.get(class_id)
    if cached and cached[0] == mtime:
        return cached[1]
    data = np.load(p)
    X, y = data["X"], data["y"]
    groups = {int(sid): np.where(y == sid)[0] for sid in np.unique(y)}
    gal = (X, y, groups)
    _galleries[class_id] = (mtime, gal)
    return gal


def find_duplicate(student, emb):
    """Name of another student in the class whose face already matches emb, else None."""
    others = get_db().execute("SELECT id, name FROM students WHERE class_id=? AND id<>?",
                              (student["class_id"], student["id"])).fetchall()
    for o in others:
        e = student_embeddings(o["id"], limit=10)
        if len(e) and float(mean_unit(e) @ emb) >= DUPLICATE_THRESHOLD:
            return o["name"]
    return None


@app.route("/student/<int:student_id>/capture")
@login_required
def capture_page(student_id):
    s = own_student(student_id)
    return render_template("capture.html", s=s, count=face_count(student_id),
                           target=30)


@app.route("/api/capture/<int:student_id>", methods=["POST"])
@login_required
def api_capture(student_id):
    s = own_student(student_id)
    img = decode_image((request.get_json(silent=True) or {}).get("image", ""))
    if img is None:
        return jsonify(error="Could not read the photo"), 400
    n = face_count(student_id)
    try:
        found = detect_and_embed(img)
    except RuntimeError as e:
        return jsonify(error=str(e)), 500
    if not found:
        return jsonify(detected=False, saved=n)
    if len(found) > 1:
        return jsonify(detected=True, multiple=True, saved=n)
    if n >= MAX_FACE_IMAGES:
        return jsonify(detected=True, saved=n, full=True)
    emb = found[0][1]
    if n < 3:
        other = find_duplicate(s, emb)
        if other:
            return jsonify(detected=True, conflict=other, saved=n)
    get_db().execute("INSERT INTO face_embeddings(student_id,vector,created_at) VALUES (?,?,?)",
                     (student_id, emb.astype(np.float32).tobytes(), now_iso()))
    get_db().commit()
    return jsonify(detected=True, saved=n + 1)


@app.route("/api/train/<int:class_id>", methods=["POST"])
@login_required
def api_train(class_id):
    own_class(class_id)
    n_students, n_samples, warnings = train_class(class_id)
    if n_samples == 0:
        return jsonify(error="No face data found. Capture faces first."), 400
    return jsonify(students=n_students, images=n_samples, warnings=warnings)

# ------------------------------------------------------------------ attendance


@app.route("/class/<int:class_id>/attendance")
@login_required
def attendance_page(class_id):
    c = own_class(class_id)
    total = get_db().execute("SELECT COUNT(*) FROM students WHERE class_id=?",
                             (class_id,)).fetchone()[0]
    return render_template("attendance.html", c=c, total=total,
                           trained=os.path.exists(model_path(class_id)),
                           today=date.today().strftime("%d-%m-%Y"))


@app.route("/api/recognize/<int:class_id>", methods=["POST"])
@login_required
def api_recognize(class_id):
    t0 = time.perf_counter()
    own_class(class_id)
    gal = load_gallery(class_id)
    if gal is None:
        return jsonify(error="Train the class faces first."), 400
    img = decode_image((request.get_json(silent=True) or {}).get("image", ""))
    if img is None:
        return jsonify(error="Could not read the photo"), 400
    try:
        found = detect_and_embed(img)
    except RuntimeError as e:
        return jsonify(error=str(e)), 500

    X, _y, groups = gal
    db = get_db()
    today = date.today().isoformat()
    now_s = datetime.now().strftime("%H:%M:%S")
    now_t = time.time()
    results = []
    for _face, emb in found:
        sims = X @ emb
        scores = {sid: float(np.sort(sims[idx])[::-1][:3].mean()) for sid, idx in groups.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        student = db.execute("SELECT * FROM students WHERE id=? AND class_id=?",
                             (best_id, class_id)).fetchone()
        if student is None or best < MATCH_THRESHOLD or (best - second) < MATCH_MARGIN:
            results.append(dict(known=False, score=round(best, 2)))
            continue
        key = (class_id, best_id)
        prev = _streaks.get(key)
        count = prev[0] + 1 if prev and now_t - prev[1] <= 3 else 1
        _streaks[key] = (count, now_t)
        confirmed = count >= CONFIRM_FRAMES
        new, marked_at = False, None
        if confirmed:
            db.execute("INSERT OR IGNORE INTO sessions(class_id,date) VALUES (?,?)",
                       (class_id, today))
            cur = db.execute("INSERT OR IGNORE INTO attendance"
                             "(student_id,date,time,confidence,source,created_at,synced) "
                             "VALUES (?,?,?,?,'camera',?,0)",
                             (best_id, today, now_s, round(best, 4), now_iso()))
            new = cur.rowcount == 1
            marked_at = now_s if new else db.execute(
                "SELECT time FROM attendance WHERE student_id=? AND date=?",
                (best_id, today)).fetchone()["time"]
        results.append(dict(known=True, confirmed=confirmed, id=best_id,
                            name=student["name"], roll_no=student["roll_no"],
                            new=new, time=marked_at, score=round(best, 2)))
    db.commit()
    present = db.execute(
        "SELECT COUNT(*) FROM attendance a JOIN students s ON s.id=a.student_id "
        "WHERE s.class_id=? AND a.date=?", (class_id, today)).fetchone()[0]
    ms = round((time.perf_counter() - t0) * 1000)
    return jsonify(faces=results, present=present, ms=ms)


# ------------------------------------------------------------------ enrollment (one-page form + camera)


@app.route("/class/<int:class_id>/enroll")
@login_required
def enroll_page(class_id):
    c = own_class(class_id)
    return render_template("enroll.html", c=c, target=30)


@app.route("/api/enroll", methods=["POST"])
@login_required
def api_enroll():
    """Create a student record. The browser then sends the face photos to /api/capture/<id>."""
    data = request.get_json(silent=True) or {}
    c = own_class(int_arg(data.get("class_id")))
    roll, name, err = clean_student_fields(data.get("roll_no"), data.get("name"))
    if err:
        return jsonify(error=err), 400
    db = get_db()
    try:
        cur = db.execute("INSERT INTO students(uid,class_id,roll_no,name) VALUES (?,?,?,?)",
                         (uuid.uuid4().hex, c["id"], roll, name))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify(error="This roll number already exists in this class."), 409
    return jsonify(student_id=cur.lastrowid, capture_url=url_for("api_capture", student_id=cur.lastrowid)), 201

if __name__ == "__main__":
    ensure_models()
    host = os.environ.get("HOST", "127.0.0.1")          # HOST=0.0.0.0 lets phones on the Wi-Fi connect
    port = int(os.environ.get("PORT", "5000"))
    use_https = os.environ.get("HTTPS") == "1"           # phones only allow the camera over HTTPS
    scheme = "https" if use_https else "http"
    print(f"Open in your browser:  {scheme}://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False, threaded=True,
            ssl_context="adhoc" if use_https else None)
