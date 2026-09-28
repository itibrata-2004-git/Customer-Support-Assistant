
import sqlite3
from pathlib import Path
from datetime import datetime
from uuid import uuid4


# ----------------------------------
# Database configuration
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "support_tickets.db"


# ----------------------------------
# Support department mapping
# ----------------------------------

DEPARTMENT_MAP = {
    "billing": "Billing Support",
    "delivery": "Delivery Support",
    "returns": "Returns and Refunds",
    "technical": "Technical Support",
    "general": "General Customer Support",
}


# ----------------------------------
# Initialize SQLite database
# ----------------------------------

def initialize_database():
    """Create or update the support ticket table."""

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        # Create table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                customer_message TEXT NOT NULL,
                category TEXT NOT NULL,
                department TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Migration for existing databases
        cursor.execute("PRAGMA table_info(tickets)")

        columns = [
            column[1]
            for column in cursor.fetchall()
        ]

        if "summary" not in columns:
            cursor.execute(
                "ALTER TABLE tickets ADD COLUMN summary TEXT"
            )

        connection.commit()

    finally:
        connection.close()


# ----------------------------------
# Create support ticket
# ----------------------------------

def create_ticket(
    customer_message: str,
    category: str,
    priority: str = "medium",
    summary: str = "",
):
    """Create a support ticket and save it in SQLite."""

    initialize_database()

    # Validate category
    category = category.lower().strip()

    if category not in DEPARTMENT_MAP:
        category = "general"

    department = DEPARTMENT_MAP[category]

    # Validate priority
    allowed_priorities = {
        "low",
        "medium",
        "high",
        "urgent",
    }

    priority = priority.lower().strip()

    if priority not in allowed_priorities:
        priority = "medium"

    # Generate unique ticket ID
    ticket_id = "SS-" + uuid4().hex[:8].upper()

    # Generate timestamp
    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

    # Save ticket to database
    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO tickets (
                ticket_id,
                customer_message,
                category,
                department,
                priority,
                status,
                summary,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                customer_message,
                category,
                department,
                priority,
                "open",
                summary,
                created_at,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    # Return complete ticket details
    return {
        "ticket_id": ticket_id,
        "customer_message": customer_message,
        "category": category,
        "department": department,
        "priority": priority,
        "status": "open",
        "summary": summary,
        "created_at": created_at,
    }


# ----------------------------------
# Retrieve all support tickets
# ----------------------------------

def get_all_tickets():
    """Return all tickets, newest first."""

    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM tickets
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ----------------------------------
# Retrieve one support ticket
# ----------------------------------

def get_ticket(ticket_id: str):
    """Find a ticket using its ticket ID."""

    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM tickets
            WHERE ticket_id = ?
            """,
            (ticket_id,),
        )

        row = cursor.fetchone()

        return dict(row) if row else None

    finally:
        connection.close()


# ----------------------------------
# Update support ticket status
# ----------------------------------

def update_ticket_status(
    ticket_id: str,
    status: str
):
    """Update the status of an existing support ticket."""

    allowed_statuses = {
        "open",
        "in progress",
        "resolved",
        "closed",
    }

    status = status.lower().strip()

    if status not in allowed_statuses:
        raise ValueError(
            f"Invalid status: {status}. "
            f"Allowed statuses: {sorted(allowed_statuses)}"
        )

    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE tickets
            SET status = ?
            WHERE ticket_id = ?
            """,
            (status, ticket_id),
        )

        updated = cursor.rowcount > 0

        connection.commit()

        return updated

    finally:
        connection.close()


# ----------------------------------
# Run database test
# ----------------------------------

if __name__ == "__main__":

    print("Initializing support ticket database...")

    initialize_database()

    print("Database initialized successfully.")

    # Create a sample ticket for testing
    result = create_ticket(
        customer_message=(
            "My order is delayed. "
            "Please connect me to support."
        ),
        category="delivery",
        priority="high",
        summary=(
            "Customer reports a delayed order "
            "and requests human support."
        ),
    )

    print("\nTicket created successfully:")
    print(result)

    print("\nAll tickets:")

    for ticket in get_all_tickets():
        print(ticket)

    print("\nDatabase test completed.")

