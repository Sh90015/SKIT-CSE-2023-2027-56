# Sprint form → project mapping

Status: ✅ done in code · 🧪 code/tool provided, you must run it on real data/devices · 👥 team activity

## Data preparation & model (member: ML/backend lead)
| Task from form | Where |
|---|---|
| Collect ~500 face images, varied light/angles | 🧪 `tools/import_dataset.py` imports the folder; the enroll/capture pages collect ~30 per student |
| OpenCV face-detection pipeline | ✅ `detect_and_embed()` in app.py (YuNet detector) |
| Pretrained CNN → recognition prototype | ✅ SFace CNN embeddings + cosine matching in app.py |
| Tune confidence threshold (false accept/reject) | ✅ `MATCH_THRESHOLD`, `MATCH_MARGIN`; 🧪 `tools/tune_threshold.py` |
| Benchmark speed on low-end Android | 🧪 `tools/benchmark.py`, `/api/health` |
| Integrate model into attendance flow | ✅ `/api/recognize` + attendance page |
| Flask REST API: enroll, recognize, sync, attendance-report | ✅ `/api/enroll`, `/api/recognize`, `/api/sync`, `/api/attendance-report` |
| Local SQLite + cloud DB integration | ✅ `sync.py` |
| Teacher-account authentication | ✅ `/signup`, `/login` (hashed passwords) |
| Connect API to frontend, optimise response time, log failed syncs | ✅ frames shrunk to ≤640 px client+server, cached gallery, `/api/health`; failed syncs in `sync_log` + `data/sync.log` |
| End-to-end offline testing, auto-sync on reconnect | ✅ `tests/test_app.py` simulates offline → reconnect; 🧪 repeat on the real device |
| Pilot at partner rural school, final testing, submission | 👥 |

## Database (member 1)
| Task | Where |
|---|---|
| Student profile + facial-embedding schema (SQLite + PostgreSQL), fields: name, roll no, class, embedding | ✅ `SCHEMA` in app.py, `CLOUD_SCHEMA` in sync.py |
| Begin labelling the face dataset | 🧪 folder naming `<roll>_<name>` for `import_dataset.py` |
| Attendance-log table: timestamp, student ID, confidence score; stress-test writes | ✅ `attendance` table; writes tested in `tests/test_app.py` |
| PostgreSQL cloud DB + sync script with duplicate/conflict handling | ✅ `sync.py` (`ON CONFLICT`, conflict log) |
| Dashboard queries on PostgreSQL + monthly CSV export | ✅ `/api/attendance-report`, `/class/<id>/report.csv?month=YYYY-MM` (reports read the local copy, which is synced to PostgreSQL) |
| Test sync accuracy under intermittent connectivity, final verification | ✅ read-back verification in `sync.py`; 🧪 repeat on real PostgreSQL after the pilot |

## UI (member 2)
| Task | Where |
|---|---|
| Enrollment form + phone-browser camera capture, dashboard wireframes | ✅ `templates/enroll.html` (wireframes: 👥) |
| Recognition result screen ("Marked Present"), error states, camera-permission handling | ✅ `templates/attendance.html`, `openCamera()` in base.html |
| Attendance history + student profile screens, client-side validation | ✅ `templates/student.html`, validation in enroll.html / class.html |
| Mobile-friendly teacher dashboard connected to APIs | ✅ responsive CSS in base.html, dashboard shows present-today |
| Test across smartphones / low bandwidth, offline indicators with pending-sync badges, final presentation | ✅ badge + offline banner (base.html); 🧪 device testing; 👥 presentation |
