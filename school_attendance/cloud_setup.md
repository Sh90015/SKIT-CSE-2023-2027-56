# PostgreSQL Cloud Database Setup

**Project:** Automated Attendance System for Rural Schools (SKIT-CSE-2023-2027-56)
**Sprint task:** Set up PostgreSQL cloud database and write sync script (1 Nov 2026 - 2 Jan 2027)
**Author:** Shivani Faujdar (Database developer)
**Branch:** shivani

## 1. What was done
- Created a PostgreSQL cloud database on **Render**.
- Connected the Flask app to it through the `DATABASE_URL` environment variable.
- Ran the first sync from the app's `/sync` page (local SQLite to cloud PostgreSQL).

## 2. Database details
| Item | Value |
|---|---|
| Provider | Render (PostgreSQL) |
| Database name | `school-attendance-db` |
| PostgreSQL version | 18 |
| Region | Singapore (Southeast Asia) |
| Plan | Free |
| Created | 11 Nov 2026 |
| Status | Available |

**Note on the free plan:** a free Render database expires 30 days after creation
(around 11 Dec 2026), with a short grace period before it is deleted. All attendance data
is also kept in the local SQLite database, so if the cloud database expires a new one can
be created and the data synced again.

## 3. How the app connects
The connection string is read from the `DATABASE_URL` environment variable. It is
**never stored in the code, in GitHub or in screenshots** because it contains a password.

Steps used on Windows PowerShell (run inside the `school_attendance` folder):
```powershell
$env:DATABASE_URL="<External Database URL from the Render dashboard>"
python app.py
```
When the app starts it prints `Cloud sync: ON (DATABASE_URL found)`.
If `DATABASE_URL` is not set, it prints `Cloud sync: OFF` and the app keeps working
offline: attendance is saved in local SQLite and synced later.

The **External** Database URL is used because the app runs on a laptop, outside Render.

## 4. Cloud tables
`sync.py` creates these tables automatically on the first sync:

| Table | Purpose |
|---|---|
| `teachers` | Teacher accounts (no passwords are synced) |
| `classes` | Classes of each teacher |
| `students` | Student profiles (roll number, name, class) |
| `sessions` | Days on which attendance was taken |
| `attendance_log` | Attendance records with date, time and confidence score |

Face embeddings stay on the local device and are not uploaded.

## 5. Sync behaviour
- Attendance is always written to local SQLite first (works with no internet).
- A background job and the `/sync` page push pending records to PostgreSQL when the
  internet is available.
- Conflict handling uses `ON CONFLICT ... DO UPDATE`, so a record synced twice does not
  create a duplicate row.
- Records deleted offline are remembered (`pending_deletes`) and removed from the cloud on
  the next sync.

## 6. First sync
- Created a class and students in the app, marked attendance, and pressed **Sync now** on
  the `/sync` page on 18 Nov 2026.
- Screenshots are saved in this `docs/` folder.

## 7. Next steps
- Verify row counts in the cloud tables (`tools/cloud_check.py`).
- Test duplicate sync, offline-to-online sync and deletion sync, and record the results in
  `docs/cloud_sync_results.md`.
- Update `FORM_MAPPING.md` and the README with the cloud setup notes.
