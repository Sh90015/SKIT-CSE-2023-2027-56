"""
Cloud Database Verification Tool
Checks local SQLite records and cloud PostgreSQL records.
Run from the project root:
    python tools/cloud_check.py
"""

import os
import sys
import sqlite3

# Allow importing sync.py from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import sync


LOCAL_DB = os.path.join(BASE_DIR, "attendance.db")

LOCAL_TABLES = ["teachers", "classes", "students", "sessions"]
CLOUD_TABLES = [
    "teachers",
    "classes",
    "students",
    "sessions",
    "attendance_log"
]


def local_count(db, table):
    return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def cloud_count(cursor, table):
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]


def main():
    print("\n======================================")
    print(" SCHOOL ATTENDANCE - CLOUD CHECK")
    print("======================================")

    # Check local database
    if not os.path.exists(LOCAL_DB):
        print("[ERROR] Local attendance.db not found.")
        return 1

    local = sqlite3.connect(LOCAL_DB)

    try:
        print("\n[1] LOCAL SQLITE DATABASE")
        print("--------------------------------------")

        for table in LOCAL_TABLES:
            print(f"{table}: {local_count(local, table)}")

        total_attendance = local_count(local, "attendance")
        synced = local.execute(
            "SELECT COUNT(*) FROM attendance WHERE synced=1"
        ).fetchone()[0]
        pending = local.execute(
            "SELECT COUNT(*) FROM attendance WHERE synced=0"
        ).fetchone()[0]
        pending_deletes = local_count(local, "pending_deletes")

        print(f"Total attendance: {total_attendance}")
        print(f"Synced attendance: {synced}")
        print(f"Pending attendance: {pending}")
        print(f"Pending deletions: {pending_deletes}")

        # Check cloud configuration
        print("\n[2] CLOUD POSTGRESQL DATABASE")
        print("--------------------------------------")

        if not sync.cloud_url():
            print("[ERROR] DATABASE_URL is not configured.")
            print("Local database is working, but cloud cannot be checked.")
            return 1

        cloud = sync._connect_cloud(sync.cloud_url())
        cursor = cloud.cursor()

        try:
            for table in CLOUD_TABLES:
                count = cloud_count(cursor, table)
                print(f"{table}: {count}")

            print("\n[3] SYNC STATUS")
            print("--------------------------------------")

            status = sync.status(LOCAL_DB)

            print(f"State: {status['state']}")
            print(f"Cloud configured: {status['configured']}")
            print(f"Pending records: {status['pending']}")
            print(f"Last sync: {status['last_sync_at']}")

            if status["pending"] == 0 and status["state"] == "synced":
                print("\n[SUCCESS] No pending records. Sync status is synced.")
            elif status["pending"] > 0:
                print("\n[INFO] Some records are still pending sync.")
            else:
                print("\n[INFO] Check the sync status and logs.")

            print("\n======================================")
            print(" CLOUD CHECK COMPLETED")
            print("======================================")

        finally:
            cursor.close()
            cloud.close()

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        return 1

    finally:
        local.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
