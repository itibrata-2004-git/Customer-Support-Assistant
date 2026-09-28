
import sys
import csv
import uuid
from pathlib import Path
from datetime import datetime

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.graph import run_support_agent

REPORT_PATH = BASE_DIR / "evaluation_report.csv"

results = []


# =====================================================
# 2. EVALUATION HELPER
# =====================================================

def record_result(test_name, passed, details):
    """Store and display an individual test result."""

    status = "PASS" if passed else "FAIL"

    results.append({
        "test_name": test_name,
        "status": status,
        "details": str(details),
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        )
    })

    print(f"\n{test_name}: {status}")
    print(f"Details: {details}")


def get_response_text(response):
    """Extract the answer from the agent response."""

    if isinstance(response, dict):
        return str(response.get("answer", "")).strip()

    return str(response).strip()


def is_agent_error(response):
    """Detect common agent failure messages."""

    text = get_response_text(response).lower()

    error_phrases = [
        "classification error",
        "rag retrieval error",
        "failed to connect to ollama",
        "connection refused",
        "internal server error"
    ]

    return any(
        phrase in text
        for phrase in error_phrases
    )


# =====================================================
# 3. TEST 1 - RETURN POLICY RAG
# =====================================================

def test_return_policy():

    print("\nTesting return policy retrieval...")

    response = run_support_agent(
        question=(
            "What is the return and refund policy? "
            "Please use the company knowledge base."
        ),
        thread_id=f"eval_return_{uuid.uuid4().hex[:8]}"
    )

    answer = get_response_text(response)

    sources = (
        response.get("sources", [])
        if isinstance(response, dict)
        else []
    )

    needs_human = (
        response.get("needs_human", False)
        if isinstance(response, dict)
        else False
    )

    ticket = (
        response.get("ticket", {})
        if isinstance(response, dict)
        else {}
    )

    valid_source = any(
        "return_policy.txt" in str(source).lower()
        for source in sources
    )

    no_unnecessary_ticket = (
        not needs_human and not ticket
    )

    passed = (
        bool(answer)
        and not is_agent_error(response)
        and valid_source
        and no_unnecessary_ticket
    )

    record_result(
        "Return Policy RAG",
        passed,
        {
            "answer_received": bool(answer),
            "correct_source_found": valid_source,
            "unnecessary_ticket_created": not no_unnecessary_ticket
        }
    )


# =====================================================
# 4. TEST 2 - SHIPPING POLICY RAG
# =====================================================

def test_shipping_policy():

    print("\nTesting shipping policy retrieval...")

    response = run_support_agent(
        question=(
            "According to the shipping policy, "
            "what should a customer do if an order is delayed?"
        ),
        thread_id=f"eval_shipping_{uuid.uuid4().hex[:8]}"
    )

    answer = get_response_text(response)

    sources = (
        response.get("sources", [])
        if isinstance(response, dict)
        else []
    )

    valid_source = any(
        "shipping_policy.txt" in str(source).lower()
        for source in sources
    )

    passed = (
        bool(answer)
        and not is_agent_error(response)
        and valid_source
    )

    record_result(
        "Shipping Policy RAG",
        passed,
        {
            "answer_received": bool(answer),
            "correct_source_found": valid_source
        }
    )


# =====================================================
# 5. TEST 3 - CONVERSATION MEMORY
# =====================================================

def test_conversation_memory():

    print("\nTesting conversation memory...")

    thread_id = f"eval_memory_{uuid.uuid4().hex[:8]}"

    run_support_agent(
        question=(
            "My order number is ORD-12345. "
            "My package has not arrived."
        ),
        thread_id=thread_id
    )

    response = run_support_agent(
        question=(
            "What was the order number I mentioned?"
        ),
        thread_id=thread_id
    )

    answer = get_response_text(response)

    passed = (
        "ord-12345" in answer.lower()
        and not is_agent_error(response)
    )

    record_result(
        "Conversation Memory",
        passed,
        {
            "expected_order_number": "ORD-12345",
            "order_number_recalled": (
                "ord-12345" in answer.lower()
            )
        }
    )


# =====================================================
# 6. TEST 4 - NEW THREAD ISOLATION
# =====================================================

def test_new_thread_isolation():

    print("\nTesting new conversation isolation...")

    new_thread_id = (
        f"eval_isolation_{uuid.uuid4().hex[:8]}"
    )

    response = run_support_agent(
        question=(
            "What was the order number I mentioned earlier?"
        ),
        thread_id=new_thread_id
    )

    answer = get_response_text(response)

    passed = (
        "ord-12345" not in answer.lower()
        and not is_agent_error(response)
    )

    record_result(
        "New Thread Isolation",
        passed,
        {
            "previous_order_number_recalled": (
                "ord-12345" in answer.lower()
            )
        }
    )


# =====================================================
# 7. TEST 5 - EXPLICIT HUMAN ESCALATION
# =====================================================

def test_human_escalation():

    print("\nTesting explicit human escalation...")

    response = run_support_agent(
        question=(
            "Please escalate my unresolved issue "
            "to a human support agent. "
            "I have contacted support multiple times "
            "and need assistance."
        ),
        thread_id=f"eval_escalation_{uuid.uuid4().hex[:8]}"
    )

    needs_human = (
        response.get("needs_human", False)
        if isinstance(response, dict)
        else False
    )

    ticket = (
        response.get("ticket", {})
        if isinstance(response, dict)
        else {}
    )

    ticket_id = (
        ticket.get("ticket_id", "")
        if isinstance(ticket, dict)
        else ""
    )

    passed = (
        bool(needs_human)
        and bool(ticket_id)
        and not is_agent_error(response)
    )

    record_result(
        "Human Escalation",
        passed,
        {
            "escalation_requested": True,
            "needs_human": needs_human,
            "ticket_id_returned": bool(ticket_id)
        }
    )


# =====================================================
# 8. SAVE CSV REPORT
# =====================================================

def save_report():

    if not results:
        print("No evaluation results to save.")
        return

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_name",
                "status",
                "details",
                "timestamp"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation report saved to: {REPORT_PATH}")


# =====================================================
# 9. RUN ALL TESTS
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMARTSUPPORT AI - RELIABILITY EVALUATION")
    print("=" * 60)

    tests = [
        test_return_policy,
        test_shipping_policy,
        test_conversation_memory,
        test_new_thread_isolation,
        test_human_escalation
    ]

    for test_function in tests:

        try:
            test_function()

        except Exception as error:

            record_result(
                test_function.__name__,
                False,
                f"Exception: {error}"
            )

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    total = len(results)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"{result['test_name']}: "
            f"{result['status']}"
        )

    print(f"\nPassed: {passed}/{total}")

    if total:
        print(
            f"Pass rate: {passed / total * 100:.1f}%"
        )

    save_report()

    print("\nEvaluation completed.")

