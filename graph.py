
import json
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
)
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.tools import create_ticket
from rag.chains import answer_question


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MEMORY_DB_PATH = BASE_DIR / "conversation_memory.sqlite"

MODEL_NAME = "llama3.2:3b"


# ============================================================
# 2. INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# ============================================================
# 3. CONVERSATION MEMORY STATE
# ============================================================

class SupportState(TypedDict, total=False):

    # Persistent conversation messages
    messages: Annotated[list[AnyMessage], add_messages]

    # Current customer question
    question: str

    # Intent classification
    category: str
    priority: str
    human_requested: bool

    # Response information for the current turn
    answer: str
    sources: list
    needs_human: bool

    # Current turn's ticket information
    ticket: dict


# ============================================================
# 4. SQLITE CHECKPOINTER
# ============================================================

# Create the SQLite database if it does not exist.
# If it already exists, reuse it.

MEMORY_DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

memory_connection = sqlite3.connect(
    str(MEMORY_DB_PATH),
    check_same_thread=False,
)

checkpointer = SqliteSaver(memory_connection)

# Create checkpoint tables if they do not exist.
checkpointer.setup()


# ============================================================
# 5. CLASSIFY CUSTOMER INTENT
# ============================================================

def classify_intent(state: SupportState):

    question = state.get("question", "").strip()

    messages = state.get("messages", [])

    # Build recent conversation history
    history = []

    for message in messages[-8:]:

        if isinstance(message, HumanMessage):
            role = "Customer"

        elif isinstance(message, AIMessage):
            role = "Assistant"

        else:
            continue

        history.append(
            f"{role}: {message.content}"
        )

    history_text = "\n".join(history)

    prompt = f"""
You are an intent classification assistant
for a customer support company.

Analyze the customer's latest message using
the conversation history.

Conversation history:
{history_text}

Latest customer message:
{question}

Classify the latest message into one category:

billing
delivery
returns
technical
general

Priority must be one of:

low
medium
high
urgent

Set human_requested to true if the customer
explicitly requests a human, a real person,
a human support agent, or escalation.

Set human_requested to false otherwise.

Do not create a ticket just because a customer
asks a normal question.

Return ONLY valid JSON in this format:

{{
    "category": "delivery",
    "priority": "medium",
    "human_requested": false
}}
"""

    try:

        response = llm.invoke(prompt)

        response_text = response.content.strip()

        # Remove Markdown code fences if returned
        if response_text.startswith("```"):

            response_text = (
                response_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        result = json.loads(response_text)

        category = str(
            result.get("category", "general")
        ).lower().strip()

        priority = str(
            result.get("priority", "medium")
        ).lower().strip()

        human_requested = (
            result.get("human_requested", False)
            is True
        )

        valid_categories = {
            "billing",
            "delivery",
            "returns",
            "technical",
            "general",
        }

        valid_priorities = {
            "low",
            "medium",
            "high",
            "urgent",
        }

        if category not in valid_categories:
            category = "general"

        if priority not in valid_priorities:
            priority = "medium"

    except Exception as error:

        print("Classification error:", error)

        # Fallback classification
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in [
                "refund",
                "return",
                "exchange",
            ]
        ):
            category = "returns"

        elif any(
            word in question_lower
            for word in [
                "payment",
                "billing",
                "invoice",
                "charge",
            ]
        ):
            category = "billing"

        elif any(
            word in question_lower
            for word in [
                "delivery",
                "shipping",
                "order",
                "tracking",
                "delayed",
            ]
        ):
            category = "delivery"

        elif any(
            word in question_lower
            for word in [
                "technical",
                "error",
                "not working",
                "bug",
            ]
        ):
            category = "technical"

        else:
            category = "general"

        priority = "medium"

        human_requested = any(
            phrase in question_lower
            for phrase in [
                "human agent",
                "speak to a human",
                "talk to a human",
                "real person",
                "human support",
                "connect me to support",
                "talk to an agent",
                "speak to an agent",
                "customer support agent",
                "escalate this",
            ]
        )

    return {
        "category": category,
        "priority": priority,
        "human_requested": human_requested,
    }


# ============================================================
# 6. BUILD CONTEXT FOR FOLLOW-UP QUESTIONS
# ============================================================

def build_contextual_question(state: SupportState):

    question = state.get("question", "").strip()

    messages = state.get("messages", [])

    # Exclude the latest customer message itself
    previous_messages = messages[:-1]

    # Keep a limited recent conversation window
    recent_messages = previous_messages[-8:]

    if not recent_messages:
        return question

    history_text = "\n".join(
        (
            "Customer: "
            if isinstance(message, HumanMessage)
            else "Assistant: "
        )
        + str(message.content)

        for message in recent_messages

        if isinstance(
            message,
            (HumanMessage, AIMessage)
        )
    )

    prompt = f"""
You are a question rewriting assistant.

Rewrite the latest customer question so it
can be understood independently by a
customer support knowledge base.

Use the conversation history only to resolve
references and understand the question.

Do not answer the customer's question.

Do not invent order numbers, customer details,
policies, or facts.

If the latest question is already standalone,
return it unchanged.

Conversation history:
{history_text}

Latest question:
{question}

Return only the rewritten question.
"""

    try:

        response = llm.invoke(prompt)

        rewritten_question = response.content.strip()

        if rewritten_question:
            return rewritten_question

    except Exception as error:

        print("Question rewriting error:", error)

    return question


# ============================================================
# 7. RETRIEVE KNOWLEDGE BASE ANSWER
# ============================================================

def retrieve_answer(state: SupportState):

    contextual_question = build_contextual_question(
        state
    )

    try:

        result = answer_question(
            contextual_question
        )

        if not isinstance(result, dict):

            raise ValueError(
                "answer_question() must return a dictionary."
            )

        answer = result.get(
            "answer",
            "I could not find an answer in the knowledge base."
        )

        sources = result.get("sources", [])

        needs_human = bool(
            result.get("needs_human", False)
        )

        if not answer:

            answer = (
                "I could not find an answer in "
                "the available support documents."
            )

    except Exception as error:

        print("RAG retrieval error:", error)

        answer = (
            "I'm sorry, I encountered an issue while "
            "searching the support knowledge base. "
            "You can request a human support agent "
            "for further assistance."
        )

        sources = []

        needs_human = True

    return {
        "answer": answer,
        "sources": sources,
        "needs_human": needs_human,

        # Save the assistant's response to conversation history
        "messages": [
            AIMessage(content=str(answer))
        ],
    }


# ============================================================
# 8. SUMMARIZE THE CONVERSATION
# ============================================================

def summarize_conversation(messages):

    conversation_text = "\n".join(
        (
            "Customer: "
            if isinstance(message, HumanMessage)
            else "Assistant: "
        )
        + str(message.content)

        for message in messages

        if isinstance(
            message,
            (HumanMessage, AIMessage)
        )
    )

    if not conversation_text.strip():

        return "Customer requested support."

    prompt = f"""
You are a customer support summarization assistant.

Summarize this conversation for a human
customer support agent.

Include:
1. The customer's main issue.
2. Relevant information provided.
3. Actions or troubleshooting already attempted.
4. What assistance the customer needs.

Do not invent facts, order numbers, or resolutions.

Keep the summary concise, within 100 words.

Conversation:
{conversation_text}

Summary:
"""

    try:

        response = llm.invoke(prompt)

        summary = response.content.strip()

        if summary:
            return summary

    except Exception as error:

        print("Conversation summary error:", error)

    # Fallback if summarization fails
    return conversation_text[-1500:]


# ============================================================
# 9. CREATE SUPPORT TICKET / HUMAN HANDOFF
# ============================================================

def escalate_to_human(state: SupportState):

    question = state.get("question", "").strip()

    category = state.get(
        "category",
        "general"
    )

    priority = state.get(
        "priority",
        "medium"
    )

    messages = state.get("messages", [])

    summary = summarize_conversation(messages)

    try:

        ticket = create_ticket(
            customer_message=question,
            category=category,
            priority=priority,
            summary=summary,
        )

        answer = (
            "I've created a support ticket for you.\n\n"
            f"**Ticket ID:** {ticket['ticket_id']}\n\n"
            f"**Department:** {ticket['department']}\n\n"
            f"**Priority:** {ticket['priority']}\n\n"
            "**Status:** Open\n\n"
            "Your conversation summary has been saved "
            "with the ticket for the support team."
        )

    except Exception as error:

        print("Ticket creation error:", error)

        ticket = {}

        answer = (
            "I'm sorry, I couldn't create your "
            "support ticket because of a database error. "
            "Please try again."
        )

    return {
        "ticket": ticket,
        "answer": answer,
        "sources": [],
        "needs_human": True,

        # Save the handoff response to conversation history
        "messages": [
            AIMessage(content=answer)
        ],
    }


# ============================================================
# 10. ROUTING LOGIC
# ============================================================

def route_after_classification(state: SupportState):

    if state.get("human_requested", False):
        return "escalate"

    return "retrieve"


def route_after_retrieval(state: SupportState):

    if state.get("needs_human", False):
        return "escalate"

    return END


# ============================================================
# 11. BUILD LANGGRAPH WORKFLOW
# ============================================================

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
        END: END,
    },
)

workflow.add_edge(
    "escalate",
    END,
)

# Compile with persistent SQLite memory
support_agent = workflow.compile(
    checkpointer=checkpointer
)


# ============================================================
# 12. RUN SUPPORT AGENT
# ============================================================

def run_support_agent(
    question: str,
    thread_id: str = "default",
):

    question = question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "needs_human": False,
            "ticket": {},
            "category": "general",
            "priority": "medium",
        }

    # Every conversation must have its own thread ID
    config = {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }

    # Only send the new message.
    # LangGraph restores previous conversation messages
    # automatically for this thread_id.
    #
    # Reset the response fields for each new turn so
    # old tickets or answers are not returned again.

    initial_state = {
        "question": question,

        "messages": [
            HumanMessage(content=question)
        ],

        "category": "general",
        "priority": "medium",
        "human_requested": False,

        "answer": "",
        "sources": [],
        "needs_human": False,
        "ticket": {},
    }

    result = support_agent.invoke(
        initial_state,
        config=config,
    )

    return {
        "answer": result.get(
            "answer",
            "I could not generate a response."
        ),

        "sources": result.get(
            "sources",
            []
        ),

        "needs_human": result.get(
            "needs_human",
            False
        ),

        "ticket": result.get(
            "ticket",
            {}
        ),

        "category": result.get(
            "category",
            "general"
        ),

        "priority": result.get(
            "priority",
            "medium"
        ),
    }


# ============================================================
# 13. TEST ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_thread_id = "smart-support-test"

    response = run_support_agent(
        "What is your shipping policy?",
        thread_id=test_thread_id,
    )

    print("\nAgent response:")
    print(response["answer"])

    print("\nSources:")
    print(response["sources"])

    print("\nConversation memory database:")
    print(MEMORY_DB_PATH)



