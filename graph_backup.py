
import json
from typing import TypedDict, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from rag.chains import answer_question
from agent.tools import create_ticket


# ----------------------------------
# Agent configuration
# ----------------------------------

CHAT_MODEL = "llama3.2:3b"

llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
)


# ----------------------------------
# Agent state
# ----------------------------------

class SupportState(TypedDict):

    question: str

    category: str
    priority: str
    human_requested: bool

    answer: str
    sources: list

    needs_human: bool

    ticket: Optional[dict]


# ----------------------------------
# Node 1: Classify customer intent
# ----------------------------------

def classify_intent(state: SupportState):

    question = state["question"]

    prompt = f"""
You are an intent classifier for ShopEase customer support.

Classify the customer's message.

Categories:
- billing
- delivery
- returns
- technical
- general

Priority:
- low
- medium
- high
- urgent

Human requested:
- true if the customer asks to speak to a human,
  representative, or live support agent.
- false otherwise.

Return ONLY a valid JSON object in this format:

{{
  "category": "delivery",
  "priority": "medium",
  "human_requested": false
}}

Customer message:
{question}
"""

    try:
        response = llm.invoke(prompt)

        content = response.content.strip()

        # Remove possible markdown code fences
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON returned")

        data = json.loads(content[start:end + 1])

        category = str(
            data.get("category", "general")
        ).lower().strip()

        priority = str(
            data.get("priority", "medium")
        ).lower().strip()

        human_requested = data.get(
            "human_requested",
            False
        )

        if category not in {
            "billing",
            "delivery",
            "returns",
            "technical",
            "general",
        }:
            category = "general"

        if priority not in {
            "low",
            "medium",
            "high",
            "urgent",
        }:
            priority = "medium"

        if not isinstance(human_requested, bool):
            human_requested = False

    except Exception as e:

        print("Classification fallback:", e)

        # Safe fallback if the local model returns
        # invalid JSON or is unavailable.

        lower_question = question.lower()

        if any(word in lower_question for word in [
            "refund",
            "return",
            "damaged",
            "broken",
            "defective",
        ]):
            category = "returns"

        elif any(word in lower_question for word in [
            "delivery",
            "shipping",
            "tracking",
            "delayed",
            "order",
        ]):
            category = "delivery"

        elif any(word in lower_question for word in [
            "payment",
            "billing",
            "charged",
            "invoice",
        ]):
            category = "billing"

        elif any(word in lower_question for word in [
            "technical",
            "app error",
            "website",
            "login",
        ]):
            category = "technical"

        else:
            category = "general"

        priority = "medium"

        human_requested = any(
            phrase in lower_question
            for phrase in [
                "human",
                "real person",
                "representative",
                "live agent",
                "speak to someone",
                "talk to someone",
            ]
        )

    return {
        "category": category,
        "priority": priority,
        "human_requested": human_requested,
        "needs_human": human_requested,
    }


# ----------------------------------
# Node 2: Retrieve RAG answer
# ----------------------------------

def retrieve_answer(state: SupportState):

    result = answer_question(
        state["question"]
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "needs_human": result["needs_human"],
    }


# ----------------------------------
# Node 3: Create human support ticket
# ----------------------------------

def escalate_to_human(state: SupportState):

    ticket = create_ticket(
        customer_message=state["question"],
        category=state["category"],
        priority=state["priority"],
        summary=(
            "Customer issue: "
            + state["question"]
        ),
    )

    if state["human_requested"]:

        answer = (
            "I understand that you would like "
            "to speak to a human support agent.\n\n"
            f"Your support ticket is "
            f"{ticket['ticket_id']}.\n\n"
            f"Department: {ticket['department']}\n"
            f"Priority: {ticket['priority'].title()}\n"
            "Status: Open\n\n"
            "Your request has been recorded for "
            "the human support team."
        )

    else:

        answer = (
            "I'm sorry, but I couldn't find "
            "sufficient information to answer "
            "your question confidently.\n\n"
            f"I've created support ticket "
            f"{ticket['ticket_id']}.\n\n"
            f"Department: {ticket['department']}\n"
            f"Priority: {ticket['priority'].title()}\n"
            "Status: Open\n\n"
            "The support team can review your issue."
        )

    return {
        "ticket": ticket,
        "answer": answer,
        "needs_human": True,
    }


# ----------------------------------
# Routing logic
# ----------------------------------

def route_after_classification(state: SupportState):

    if state["human_requested"]:
        return "escalate"

    return "retrieve"


def route_after_retrieval(state: SupportState):

    if state["needs_human"]:
        return "escalate"

    return "finish"


# ----------------------------------
# Build LangGraph workflow
# ----------------------------------

workflow = StateGraph(SupportState)

workflow.add_node(
    "classify",
    classify_intent,
)

workflow.add_node(
    "retrieve",
    retrieve_answer,
)

workflow.add_node(
    "escalate",
    escalate_to_human,
)


workflow.add_edge(
    START,
    "classify",
)

workflow.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "retrieve": "retrieve",
        "escalate": "escalate",
    },
)

workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "escalate": "escalate",
        "finish": END,
    },
)

workflow.add_edge(
    "escalate",
    END,
)


support_agent = workflow.compile()


# ----------------------------------
# Public function for the UI
# ----------------------------------

def run_support_agent(question: str):

    initial_state = {
        "question": question,
        "category": "general",
        "priority": "medium",
        "human_requested": False,
        "answer": "",
        "sources": [],
        "needs_human": False,
        "ticket": None,
    }

    result = support_agent.invoke(
        initial_state
    )

    return result


# ----------------------------------
# Test the agent directly
# ----------------------------------

if __name__ == "__main__":

    while True:

        question = input(
            "\nCustomer: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit",
        ]:
            break

        if not question:
            continue

        result = run_support_agent(question)

        print("\nSmartSupport AI:")
        print(result["answer"])

        print("\nCategory:", result["category"])
        print("Human handoff:", result["needs_human"])

        if result["ticket"]:
            print("Ticket:", result["ticket"])
