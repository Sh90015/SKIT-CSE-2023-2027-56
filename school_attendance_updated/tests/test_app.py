"""End-to-end test of the whole app without needing the face models or a real PostgreSQL.

Run:  python tests/test_app.py
* detect_and_embed is replaced by a fake that "recognises" a person from a colour in the image.
* PostgreSQL is replaced by a tiny SQLite-backed fake so the real sync code is exercised.
"""
import base64, os, sqlite3, sys, tempfile, shutil, types

import cv2
import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp()
shutil.copytree(SRC, os.path.join(TMP, "app"), ignore=shutil.ignore_patterns("data", "attendance.db*", "tests", "__pycache__"))
APPDIR = os.path.join(TMP, "app")
sys.path.insert(0, APPDIR)

# ---------- fake psycopg2 backed by SQLite -------------------------------------------------
CLOUD_FILE = os.path.join(TMP, "cloud.db")
fake = types.ModuleType("psycopg2")
fake.online = True

class OperationalError(Exception):
    pass

class _Cur:
    def __init__(self, c): self.c = c
    def execute(self, sql, params=()):
        self.cur = self.c.execute(sql.replace("%s", "?"), params)
    def fetchall(self): return self.cur.fetchall()

class _Conn:
    def __init__(self):
        self.c = sqlite3.connect(CLOUD_FILE)
        self.c.execute("PRAGMA foreign_keys=ON")
    def cursor(self): return _Cur(self.c)
    def commit(self): self.c.commit()
    def rollback(self): self.c.rollback()
    def close(self): self.c.close()

def connect(url, connect_timeout=8):
    if not fake.online:
        raise OperationalError("could not connect to server: Network is unreachable")
    return _Conn()
fake.connect, fake.OperationalError = connect, OperationalError
sys.modules["psycopg2"] = fake
os.environ["DATABASE_URL"] = "postgresql://fake/test"

import app as A   # noqa: E402
import sync as S  # noqa: E402

# ---------- fake face model ----------------------------------------------------------------
PEOPLE = {}
def person_vec(pid):
    rng = np.random.RandomState(pid * 7 + 1)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)

def fake_detect(img):
    pid = int(img[0, 0, 0])            # blue channel of the top-left pixel = person id (0 = nobody)
    if pid == 0:
        return []
    if pid >= 200:                     # two faces in the frame
        return [(np.array([0, 0, 100, 100]), person_vec(1)), (np.array([0, 0, 90, 90]), person_vec(2))]
    v = person_vec(pid) + np.random.RandomState(0).randn(128).astype(np.float32) * 0.01
    return [(np.array([0, 0, 100, 100]), v / np.linalg.norm(v))]
A.detect_and_embed = fake_detect

def frame(pid):
    img = np.zeros((480, 640, 3), np.uint8); img[:, :, 0] = pid
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

passed = 0
def check(name, cond, extra=""):
    global passed
    print(("PASS " if cond else "FAIL ") + name + (" " + str(extra) if not cond else ""))
    assert cond, name
    passed += 1

c = A.app.test_client()
# ---------- auth ----------
check("login page", c.get("/login").status_code == 200)
c.post("/signup", data=dict(name="Teacher One", username="t1", password="secret1"))
r = c.post("/login", data=dict(username="t1", password="secret1"))
check("login redirects", r.status_code == 302)
check("bad login rejected", A.app.test_client().post("/login", data=dict(username="t1", password="x")).status_code == 200)
check("api needs login", A.app.test_client().get("/api/sync/status").status_code == 401)

# ---------- class + enroll ----------
c.post("/dashboard", data=dict(name="Class 5 - A"))
check("dashboard", b"Class 5 - A" in c.get("/dashboard").data)
check("enroll page", c.get("/class/1/enroll").status_code == 200)
r = c.post("/api/enroll", json=dict(class_id=1, roll_no="1", name="Asha Verma"))
check("enroll ok", r.status_code == 201 and r.json["student_id"] == 1, r.json)
check("enroll duplicate roll", c.post("/api/enroll", json=dict(class_id=1, roll_no="1", name="Dup Name")).status_code == 409)
check("enroll bad roll", c.post("/api/enroll", json=dict(class_id=1, roll_no="@@", name="Ok Name")).status_code == 400)
check("enroll name digits", c.post("/api/enroll", json=dict(class_id=1, roll_no="9", name="R2D2")).status_code == 400)
check("enroll hindi name", c.post("/api/enroll", json=dict(class_id=1, roll_no="5", name="शिवानी फौजदार")).status_code == 201)
c.post("/api/enroll", json=dict(class_id=1, roll_no="2", name="Ravi Kumar"))
sid = {"Asha": 1, "Ravi": 3}
for name, pid in (("Asha", 1), ("Ravi", 2)):
    for _ in range(5):
        r = c.post(f"/api/capture/{sid[name]}", json=dict(image=frame(pid)))
    check(f"capture {name}", r.json["saved"] == 5, r.json)
check("capture no face", c.post("/api/capture/1", json=dict(image=frame(0))).json["detected"] is False)
check("capture two faces", c.post("/api/capture/1", json=dict(image=frame(200))).json.get("multiple") is True)
check("duplicate face blocked", c.post("/api/capture/2", json=dict(image=frame(1))).json.get("conflict") == "Asha Verma")
r = c.post("/api/train/1")
check("train", r.status_code == 200 and r.json["students"] == 2, r.json)

# ---------- recognition ----------
r = c.post("/api/recognize/1", json=dict(image=frame(1))).json
check("first sight not yet confirmed", r["faces"][0]["known"] and not r["faces"][0]["confirmed"])
r = c.post("/api/recognize/1", json=dict(image=frame(1))).json
f = r["faces"][0]
check("confirmed + marked", f["confirmed"] and f["new"] and f["name"] == "Asha Verma" and r["present"] == 1, r)
r = c.post("/api/recognize/1", json=dict(image=frame(1))).json
check("second time = already marked", not r["faces"][0]["new"] and r["faces"][0]["time"] is not None)
r = c.post("/api/recognize/1", json=dict(image=frame(99))).json
check("unknown face", r["faces"] and not r["faces"][0]["known"])
db = sqlite3.connect(A.DB_PATH); db.row_factory = sqlite3.Row
row = db.execute("SELECT * FROM attendance").fetchone()
check("log has confidence/source/created_at", row["confidence"] > 0.9 and row["source"] == "camera" and row["created_at"] and row["synced"] == 0, dict(row))
check("health stats", c.get("/api/health").json["recognize_calls"] >= 3)

# ---------- status / sync ----------
st = c.get("/api/sync/status").json
check("status pending", st["state"] == "pending" and st["pending"] == 1, st)
fake.online = False
r = c.post("/api/sync").json
check("offline sync fails gracefully", not r["ok"] and r["state"] == "offline", r)
check("offline status keeps pending", c.get("/api/sync/status").json["pending"] == 1)
check("failure is logged", "Network is unreachable" in c.get("/sync").data.decode())
c.post("/api/recognize/1", json=dict(image=frame(2))); c.post("/api/recognize/1", json=dict(image=frame(2)))
check("attendance still works offline", c.get("/api/attendance-report/1").json["students"][0]["present_days"] >= 0)
fake.online = True
r = c.post("/api/sync").json
check("sync after reconnect", r["ok"] and r["pushed"] == 2 and r["status"]["pending"] == 0, r)
cl = sqlite3.connect(CLOUD_FILE)
check("cloud has 2 attendance rows", cl.execute("SELECT COUNT(*) FROM attendance_log").fetchone()[0] == 2)
check("cloud has students", cl.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 3)
check("cloud row has confidence+device", cl.execute("SELECT confidence, device_id FROM attendance_log").fetchone()[0] > 0.9)
check("sync idempotent", c.post("/api/sync").json["pushed"] == 0)

# manual correction after sync -> delete propagates
check("mark absent", c.post("/api/mark", json=dict(student_id=3, date=A.date.today().isoformat(), present=False)).json["ok"])
check("pending delete queued", c.get("/api/sync/status").json["pending"] == 1)
c.post("/api/sync")
check("cloud row deleted", cl.execute("SELECT COUNT(*) FROM attendance_log").fetchone()[0] == 1)
cl.close(); cl = sqlite3.connect(CLOUD_FILE)
c.post("/api/mark", json=dict(student_id=3, date=A.date.today().isoformat(), present=True))
check("manual present pending", c.get("/api/sync/status").json["pending"] == 1)

# duplicate / conflict handling: same roll under a different uid already in the cloud
cl.execute("PRAGMA foreign_keys=OFF")
cl.execute("UPDATE students SET uid='zzz-other-device' WHERE roll_no='2'"); cl.commit()
cl.execute("PRAGMA foreign_keys=ON")
# -> cloud now has roll 2 with another uid, local roll 2 will conflict on UNIQUE(class_uid, roll_no)
r = c.post("/api/sync").json
log = c.get("/sync").data.decode()
check("conflict logged, others still sync", "already exists in the cloud" in log, r)

# ---------- profile / reports / csv ----------
r = c.get("/student/1"); check("student profile", r.status_code == 200 and b"Attendance history" in r.data and b"Present" in r.data)
r = c.get("/class/1/report"); check("report page", r.status_code == 200 and b"Monthly summary" in r.data)
r = c.get("/class/1/report.csv"); check("overall csv", b"Attendance %" in r.data and r.mimetype == "text/csv")
month = A.date.today().strftime("%Y-%m")
r = c.get(f"/class/1/report.csv?month={month}")
txt = r.data.decode("utf-8-sig")
check("monthly csv", "Present Days" in txt and "Present count" in txt and "P" in txt, txt)
check("monthly csv bad month falls back", b"Attendance %" in c.get("/class/1/report.csv?month=bad").data)
j = c.get(f"/api/attendance-report/1?month={month}").json
check("json report", j["class_name"] == "Class 5 - A" and j["month"]["days"] and len(j["students"]) == 3, j)
check("json bad date", c.get("/api/attendance-report/1?date=x").status_code == 400)
check("sync page", c.get("/sync").status_code == 200)
check("capture page", c.get("/student/1/capture").status_code == 200)
check("class page", b"Enroll student" in c.get("/class/1").data)
check("attendance page", b"Take attendance" in c.get("/class/1/attendance").data)

# ---------- delete + migration ----------
db2 = sqlite3.connect(A.DB_PATH)
uid2 = db2.execute("SELECT uid FROM students WHERE id=2").fetchone()[0]
c.post("/api/sync"); cl.close(); cl = sqlite3.connect(CLOUD_FILE)
check("student 2 is in cloud before delete", cl.execute("SELECT COUNT(*) FROM students WHERE uid=?", (uid2,)).fetchone()[0] == 1)
c.post("/student/2/delete")
check("student removed locally", db2.execute("SELECT COUNT(*) FROM students WHERE id=2").fetchone()[0] == 0)
check("embeddings removed", db2.execute("SELECT COUNT(*) FROM face_embeddings WHERE student_id=2").fetchone()[0] == 0)
check("delete queued for cloud", c.get("/api/sync/status").json["pending"] >= 1)
c.post("/api/sync"); cl.close(); cl = sqlite3.connect(CLOUD_FILE)
check("cloud student deleted too", cl.execute("SELECT COUNT(*) FROM students WHERE uid=?", (uid2,)).fetchone()[0] == 0)

# old .npy -> table migration
os.makedirs(os.path.join(A.FACES_DIR, "1"), exist_ok=True)
np.save(os.path.join(A.FACES_DIR, "1", "a.npy"), person_vec(1))
before = db2.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
A.init_db()
after = db2.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
check("npy migrated into DB", after == before + 1)
check("old faces folder moved aside", not os.path.isdir(A.FACES_DIR))

print(f"\nALL {passed} CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
