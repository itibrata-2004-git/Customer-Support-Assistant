
from typing import TypedDict, Annotated
from pathlib import Path
import sqlite3
import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from rag.chains import answer_question
from agent.tools import create_ticket


# --------------------------------------------------
# 1. PROJECT PATHS AND DATABASE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MEMORY_DB_PATH = BASE_DIR / "conversation_memory.sqlite"

# Keep the SQLite connection alive for the lifetime
# of the application.
connection = sqlite3.connect(
    str(MEMORY_DB_PATH),
    check_same_thread=False
)

checkpointer = SqliteSaver(connection)
checkpointer.setup()


# --------------------------------------------------
# 2. LOCAL LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# --------------------------------------------------
# 3. AGENT STATE
# --------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    question: str
    category: str
    priority: str
    human_requested: bool

    answer: str
    sources: list
    needs_human: bool
    ticket: dict | None


# --------------------------------------------------
# 4. CLASSIFY CUSTOMER INTENT
# --------------------------------------------------

def classify_intent(state: SupportState):

    question = state["question"]

    prompt = f"""
You are an intent classification system
for a customer support chatbot.

Classify the customer's latest message.

Conversation history:
{format_history(state.get("messages", []))}

Latest customer message:
{question}

Return ONLY a valid JSON object with these fields:

category: one of billing, delivery, returns,
technical, general

priority: one of low, medium, high

human_requested: true or false

Rules:
- If the customer asks for a human, live agent,
  or customer representative, human_requested is true.
- If the customer reports a serious problem,
  choose high priority.
- Otherwise choose a suitable category and priority.
- Do not include markdown or explanations.
"""

    try:
        response = llm.invoke(prompt).content.strip()

        # Remove possible markdown formatting
        response = response.replace("```json", "")
        response = response.replace("```", "").strip()

        result = json.loads(response)

        category = result.get("category", "general").lower()
        priority = result.get("priority", "medium").lower()
        human_requested = result.get("human_requested", False)

        valid_categories = [
            "billing", "delivery", "returns",
            "technical", "general"
        ]

        valid_priorities = ["low", "medium", "high"]

        if category not in valid_categories:
            category = "general"

        if priority not in valid_priorities:
            priority = "medium"

        if isinstance(human_requested, str):
            human_requested = human_requested.lower() == "true"

    except Exception:
        # Simple fallback if the model returns invalid JSON
        text = question.lower()

        category = "general"
        priority = "medium"

        if any(word in text for word in [
            "refund", "payment", "bill", "charged"
        ]):
            category = "billing"

        elif any(word in text for word in [
            "delivery", "shipping", "order", "delayed"
        ]):
            category = "delivery"

        elif any(word in text for word in [
            "return", "replace", "exchange"
        ]):
            category = "returns"

        elif any(word in text for word in [
            "error", "technical", "not working", "bug"
        ]):
            category = "technical"

        human_requested = any(word in text for word in [
            "human", "live agent", "representative",
            "customer support agent", "speak to someone"
        ])

    return {
        "category": category,
        "priority": priority,
        "human_requested": human_requested
    }


# --------------------------------------------------
# 5. FORMAT CONVERSATION HISTORY
# --------------------------------------------------

def format_history(messages):

    history = []

    # Use recent conversation messages for context
    for message in messages[-10:]:

        if isinstance(message, HumanMessage):
            role = "Customer"

        elif isinstance(message, AIMessage):
            role = "Assistant"

        else:
            continue

        content = str(message.content)

        history.append(f"{role}: {content}")

    return "\n".join(history)


# --------------------------------------------------
# 6. REWRITE FOLLOW-UP QUESTIONS
# --------------------------------------------------

def create_standalone_question(state: SupportState):

    question = state["question"]
    messages = state.get("messages", [])

    # A first question does not need rewriting
    if len(messages) <= 1:
        return question

    history = format_history(messages[:-1])

    prompt = f"""
Rewrite the latest customer question as a
standalone question using the conversation history.

Conversation history:
{history}

Latest question:
{question}

Rules:
- Preserve the customer's actual intent.
- Resolve references such as "it", "that order",
  "the previous issue", and "what about that".
- Do not answer the question.
- Return only the rewritten question.
"""

    try:
        response = llm.invoke(prompt).content.strip()

        if response:
            return response

    except Exception:
        pass

    return question


# --------------------------------------------------
# 7. RETRIEVE ANSWER USING RAG
# --------------------------------------------------

def retrieve_answer(state: SupportState):

    question = state["question"]

    standalone_question = create_standalone_question(state)

    # Use your existing RAG pipeline
    result = answer_question(standalone_question)

    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "sources": [],
            "needs_human": False
        }

    answer = result.get(
        "answer",
        "I could not generate an answer."
    )

    sources = result.get("sources", [])

    needs_human = result.get("needs_human", False)

    return {
        "answer": answer,
        "sources": sources,
        "needs_human": needs_human
    }


# --------------------------------------------------
# 8. CREATE SUPPORT TICKET
# --------------------------------------------------

def escalate_to_human(state: SupportState):

    question = state["question"]
    category = state["category"]
    priority = state["priority"]

    # Create ticket in the existing SQLite ticket system
    ticket = create_ticket(
        customer_message=question,
        category=category,
        priority=priority
    )

    ticket_id = ticket.get("ticket_id", "N/A")

    message = (
        "I have created a support ticket for you.\n\n"
        f"Ticket ID: {ticket_id}\n"
        f"Category: {category.title()}\n"
        f"Priority: {priority.title()}\n\n"
        "Your conversation has been recorded locally. "
        "Please keep your ticket ID for reference."
    )

    return {
        "ticket": ticket,
        "answer": message,
        "messages": [
            AIMessage(content=message)
        ]
    }


# --------------------------------------------------
# 9. ROUTING LOGIC
# --------------------------------------------------

def route_after_classification(state: SupportState):

    if state.get("human_requested", False):
        return "escalate"

    return "retrieve"


def route_after_retrieval(state: SupportState):

    if state.get("needs_human", False):
        return "escalate"

    return "finish"


# --------------------------------------------------
# 10. FINISH CONVERSATION
# --------------------------------------------------

def finish_conversation(state: SupportState):

    answer = state.get("answer", "")

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


# --------------------------------------------------
# 11. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

workflow = StateGraph(SupportState)

workflow.add_node("classify", classify_intent)
workflow.add_node("retrieve", retrieve_answer)
workflow.add_node("escalate", escalate_to_human)
workflow.add_node("finish", finish_conversation)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "escalate": "escalate",
        "retrieve": "retrieve"
    }
)

workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "escalate": "escalate",
        "finish": "finish"
    }
)

workflow.add_edge("finish", END)
workflow.add_edge("escalate", END)


# --------------------------------------------------
# 12. COMPILE GRAPH WITH PERSISTENT MEMORY
# --------------------------------------------------

support_agent = workflow.compile(
    checkpointer=checkpointer
)


# --------------------------------------------------
# 13. RUN SUPPORT AGENT
# --------------------------------------------------

def run_support_agent(
    question: str,
    thread_id: str = "default"
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {
        "messages": [
            HumanMessage(content=question)
        ],

        "question": question,
        "category": "general",
        "priority": "medium",
        "human_requested": False,

        "answer": "",
        "sources": [],
        "needs_human": False,
        "ticket": None
    }

    result = support_agent.invoke(
        initial_state,
        config=config
    )

    return result

