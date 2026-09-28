
from agent.tools import get_all_tickets

# Fetch all saved support tickets
tickets = get_all_tickets()

if not tickets:
    print("No support tickets found.")

else:
    print(f"\nTotal tickets: {len(tickets)}")

    for ticket in tickets:
        print("-" * 50)

        print("Ticket ID:", ticket["ticket_id"])
        print("Category:", ticket["category"])
        print("Department:", ticket["department"])
        print("Priority:", ticket["priority"])
        print("Status:", ticket["status"])
        print("Message:", ticket["customer_message"])
        print("Created:", ticket["created_at"])
