
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "support_tickets.db"

if not DB_PATH.exists():
    print("Database not found:", DB_PATH)
else:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
    """)

    tables = cursor.fetchall()

    print("\nTables in database:")
    for table in tables:
        print("\nTable:", table[0])

        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()

        print("Columns:")
        for column in columns:
            print(f"  {column[1]} ({column[2]})")

        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        print("Rows:", cursor.fetchone()[0])

    conn.close()
