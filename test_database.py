
import sqlite3
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "support_tickets.db"


class TestSupportDatabase(unittest.TestCase):

    def setUp(self):
        """Connect to the existing database."""

        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()

    def tearDown(self):
        """Close the database connection."""

        self.conn.close()

    def test_database_exists(self):
        """Check whether the database exists."""

        self.assertTrue(DB_PATH.exists())

    def test_tickets_table_exists(self):
        """Check whether the tickets table exists."""

        self.cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name='tickets'
        """)

        result = self.cursor.fetchone()

        self.assertIsNotNone(result)

    def test_required_columns_exist(self):
        """Verify the ticket table schema."""

        self.cursor.execute(
            "PRAGMA table_info(tickets)"
        )

        columns = [
            row[1]
            for row in self.cursor.fetchall()
        ]

        required_columns = [
            "ticket_id",
            "customer_message",
            "category",
            "department",
            "priority",
            "status",
            "summary",
            "created_at"
        ]

        for column in required_columns:
            self.assertIn(
                column,
                columns,
                f"Missing column: {column}"
            )

    def test_ticket_records_are_readable(self):
        """Check whether ticket records can be retrieved."""

        self.cursor.execute("""
            SELECT ticket_id, status
            FROM tickets
        """)

        records = self.cursor.fetchall()

        self.assertIsInstance(records, list)

        print(
            f"\nTickets currently in database: {len(records)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
