# Student Attendance System (Face Recognition) — offline-first

Teacher login → classes → enroll students with the camera → face recognition marks attendance →
reports, monthly CSV, student profiles. Works with **no internet** (saved in local SQLite) and
**syncs automatically to a cloud PostgreSQL database** when the internet is back.

## How to run
1. Open this folder in VS Code and open a Terminal
2. `pip install -r requirements.txt`
3. `python app.py`  (first run needs internet once to download the face model, ~38 MB)
4. Open http://127.0.0.1:5000

### Use it from phones (camera needs HTTPS on phones)
Run on the school computer/laptop, connected to the same Wi-Fi as the phones:
```
set HOST=0.0.0.0         (Windows)      export HOST=0.0.0.0     (Mac/Linux)
set HTTPS=1                             export HTTPS=1
python app.py
```
Open `https://<computer-ip>:5000` on the phone. The browser shows a "not secure certificate"
warning (self-signed) → choose *Advanced → Continue*. Then allow the camera.
`localhost` on the same computer works without HTTPS.

## How to use
1. **Create an account** (teacher), sign in, **add a class**
2. **Enroll student (camera)**: type roll no. + name, the camera takes ~30 face photos by itself
   (or use the quick *Add a student* form + *Capture face*)
3. Press **Train faces** on the class page (again whenever you add students)
4. **Take attendance** → Start → each student stands in front of the camera. A big green
   *Marked Present* screen confirms; red/amber screens explain unrecognised faces, camera
   problems or a lost connection
5. **Reports**: by date, overall %, **Overall CSV** and **Monthly CSV** (P/A for every day).
   Click Present/Absent to fix by hand. Click a student's name for the **profile + attendance history**

## Cloud sync (PostgreSQL on Render / Railway)
1. Create a free PostgreSQL database on Render or Railway and copy its **External Database URL**
2. Set it before starting the app:
   `set DATABASE_URL=postgresql://user:pass@host:5432/dbname` (Windows) /
   `export DATABASE_URL=...` (Mac/Linux)
3. Start `python app.py`. The tables are created automatically.

* The top bar shows a badge: **Synced ✓ / N pending sync / Offline · N pending / Local only**.
  Click it for the **Sync log** page (every failed attempt is logged there and in `data/sync.log`).
* While offline, attendance is saved locally with `synced = 0`. The app retries every minute
  (`SYNC_INTERVAL` seconds) and you can press **Sync now**.
* After pushing, each record is **read back from the cloud**; only then is it marked synced.
* Duplicates are ignored (first record wins); a roll number that clashes with a different student
  in the cloud is logged as a *conflict* and skipped without blocking the rest.
* Manual corrections and deleted students are also removed from the cloud on the next sync.
* Face embeddings stay on the school device only (biometric data of children is not uploaded).

## REST API (all need the teacher to be logged in)
| Route | Purpose |
|---|---|
| `POST /api/enroll` `{class_id, roll_no, name}` | create a student, returns `student_id` |
| `POST /api/capture/<student_id>` `{image}` | add a face photo (base64 data URL) |
| `POST /api/train/<class_id>` | build the recognition model for the class |
| `POST /api/recognize/<class_id>` `{image}` | recognise faces in a frame and mark attendance |
| `POST /api/sync` · `GET /api/sync/status` | sync now · pending count / state |
| `GET /api/attendance-report/<class_id>?date=YYYY-MM-DD&month=YYYY-MM` | JSON report |
| `GET /api/health` | average / p95 recognition time (ms) |

## Database
SQLite (`attendance.db`): `teachers, classes, students` (profile: roll no, name, class),
`face_embeddings` (128-number vector per photo), `sessions`, `attendance`
(student, date, time, **confidence**, source, created_at, **synced**), `pending_deletes`, `sync_log`.
Cloud PostgreSQL: `teachers, classes, students, sessions, attendance_log`.
Old databases/`.npy` face files from earlier versions are upgraded automatically.

## Settings (top of app.py)
- `MATCH_THRESHOLD = 0.42` — wrong student marked? raise (e.g. 0.50). Faces not recognised? lower (0.36)
- `MATCH_MARGIN = 0.06`, `CONFIRM_FRAMES = 2`, `LOW_ATTENDANCE = 75`, `MAX_IMAGE_SIDE = 640`
- Stop strangers creating accounts: `set SIGNUP_CODE=myschool123` before `python app.py`

## Tools (`tools/`)
- `python tools/import_dataset.py dataset --class-id 1` — bulk-import a folder of face photos
  (`dataset/<roll>_<name>/*.jpg`), e.g. your ~500-image dataset
- `python tools/tune_threshold.py` — shows false-accept / false-reject for every threshold using the
  enrolled faces and suggests the best `MATCH_THRESHOLD`
- `python tools/benchmark.py <photo-folder>` — recognition speed in ms (run it on the low-end phone/PC)
- `python tests/test_app.py` — automatic end-to-end test (fake face model + fake cloud DB)

## Good to know
- One face can belong to only one student; **Train faces** warns if two students look alike.
- Capture in good light, one person in the frame.
- Data is stored in this folder: `attendance.db` and `data/`. Copy them to back up.
- Get parents'/school permission before storing children's face data.
- If the model download fails, save https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx into `models/`.
