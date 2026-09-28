
# ============================================================
# SmartSupport AI - Customer Support Chatbot
# File: App/frontend/app.py
# Phase 3: Persistent Conversation Memory
# ============================================================

import sys
import uuid
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartSupport AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. IMPORT LANGGRAPH AGENT
# ============================================================

try:
    from agent.graph import run_support_agent

except Exception as e:
    st.error("Could not import the SmartSupport AI agent.")
    st.code(str(e))
    st.stop()


# ============================================================
# 4. INITIALIZE SESSION STATE
# ============================================================

# Every conversation receives a unique ID.
# LangGraph uses this ID to store and retrieve memory
# from conversation_memory.sqlite.

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Streamlit stores the visible conversation here.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track whether the agent is currently processing.
if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 16px;
        color: #64748b;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    div[data-testid="stChatInput"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎧 SmartSupport AI")

    st.caption(
        "AI-powered customer support assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
        key="new_chat_button",
    ):

        # Create a completely separate conversation.
        st.session_state.thread_id = str(uuid.uuid4())

        # Clear only the visible messages.
        # Existing SQLite conversations remain saved.
        st.session_state.messages = []

        st.session_state.processing = False

        st.rerun()

    st.subheader("Features")

    st.markdown(
        """
        - 🤖 Local LLM with Ollama
        - 📚 RAG knowledge retrieval
        - 🧠 LangGraph agent workflow
        - 💾 Persistent conversation memory
        - 🎫 Automatic support ticket creation
        - 👨‍💻 Human support escalation
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
        key="clear_chat_button",
    ):

        # Start a fresh thread so previous context
        # cannot leak into the next question.
        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.session_state.processing = False

        st.rerun()

    st.divider()

    st.caption(
        "SmartSupport AI | Phase 3"
    )

    st.caption(
        "Powered by LangGraph, Ollama, "
        "ChromaDB, SQLite and Streamlit"
    )


# ============================================================
# 7. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'SmartSupport AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Your intelligent customer support assistant. '
    'Ask questions about company policies, orders, '
    'returns, billing and more.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# 8. DISPLAY WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message("assistant"):

        st.markdown(
            """
            Hello! 👋 Welcome to SmartSupport AI.

            I'm your AI customer support assistant.

            I can help you with:

            - Shipping and delivery policies
            - Return and refund policies
            - Company information
            - Billing-related questions
            - Connecting you with human support

            **How can I help you today?**
            """
        )


# ============================================================
# 9. DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    with st.chat_message(role):

        st.markdown(content)

        # ----------------------------------------------------
        # DISPLAY SAVED TICKET DETAILS
        # ----------------------------------------------------

        ticket = message.get("ticket")

        if ticket:

            st.success(
                "🎫 Ticket created: "
                + str(
                    ticket.get(
                        "ticket_id",
                        "N/A"
                    )
                )
            )

            st.write(
                "**Department:** "
                + str(
                    ticket.get(
                        "department",
                        "N/A"
                    )
                )
            )

            st.write(
                "**Priority:** "
                + str(
                    ticket.get(
                        "priority",
                        "medium"
                    )
                ).title()
            )

            st.write(
                "**Status:** "
                + str(
                    ticket.get(
                        "status",
                        "open"
                    )
                ).title()
            )

        # ----------------------------------------------------
        # DISPLAY KNOWLEDGE SOURCES
        # ----------------------------------------------------

        sources = message.get(
            "sources",
            []
        )

        if sources:

            with st.expander(
                "📚 View Knowledge Sources"
            ):

                for source in sources:

                    st.write(f"- {source}")

        # ----------------------------------------------------
        # DISPLAY HUMAN ESCALATION NOTICE
        # ----------------------------------------------------

        if message.get(
            "needs_human",
            False
        ):

            st.warning(
                "This question may require human support."
            )


# ============================================================
# 10. CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask SmartSupport AI a question..."
)


# ============================================================
# 11. PROCESS CUSTOMER QUESTION
# ============================================================

if user_input:

    # --------------------------------------------------------
    # A. SAVE CUSTOMER MESSAGE IN STREAMLIT SESSION
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Display customer message immediately.
    with st.chat_message("user"):

        st.markdown(user_input)

    # --------------------------------------------------------
    # B. RUN LANGGRAPH SUPPORT AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "SmartSupport AI is thinking..."
        ):

            try:

                # Use the current conversation's thread ID.
                # The LangGraph SQLite checkpointer restores
                # the history associated with this thread.

                result = run_support_agent(
                    question=user_input,
                    thread_id=st.session_state.thread_id,
                )

                # --------------------------------------------
                # EXTRACT RESPONSE
                # --------------------------------------------

                answer = result.get(
                    "answer",
                    "Sorry, I could not generate an answer."
                )

                sources = result.get(
                    "sources",
                    []
                )

                needs_human = result.get(
                    "needs_human",
                    False
                )

                ticket = result.get(
                    "ticket",
                    None
                )

                # Normalize sources.
                if sources is None:
                    sources = []

                # --------------------------------------------
                # DISPLAY ASSISTANT ANSWER
                # --------------------------------------------

                st.markdown(answer)

                # --------------------------------------------
                # DISPLAY TICKET INFORMATION
                # --------------------------------------------

                if ticket:

                    st.success(
                        "🎫 Ticket created: "
                        + str(
                            ticket.get(
                                "ticket_id",
                                "N/A"
                            )
                        )
                    )

                    st.write(
                        "**Department:** "
                        + str(
                            ticket.get(
                                "department",
                                "General Customer Support"
                            )
                        )
                    )

                    st.write(
                        "**Priority:** "
                        + str(
                            ticket.get(
                                "priority",
                                "medium"
                            )
                        ).title()
                    )

                    st.write(
                        "**Status:** "
                        + str(
                            ticket.get(
                                "status",
                                "open"
                            )
                        ).title()
                    )

                # --------------------------------------------
                # DISPLAY KNOWLEDGE SOURCES
                # --------------------------------------------

                if sources:

                    with st.expander(
                        "📚 View Knowledge Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"- {source}"
                            )

                # --------------------------------------------
                # HUMAN ESCALATION WARNING
                # --------------------------------------------

                if needs_human:

                    st.warning(
                        "This question may require "
                        "human support."
                    )

            except Exception as e:

                # --------------------------------------------
                # HANDLE ERRORS SAFELY
                # --------------------------------------------

                answer = (
                    "Sorry, something went wrong "
                    "while processing your request. "
                    "Please try again."
                )

                sources = []

                needs_human = False

                ticket = None

                st.error(
                    "An error occurred while "
                    "processing your request."
                )

                with st.expander(
                    "View technical details"
                ):

                    st.code(str(e))

                st.info(
                    "Check that Ollama is running, "
                    "the local model is installed, "
                    "and the RAG database is available."
                )

    # ========================================================
    # 12. SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "needs_human": needs_human,
            "ticket": ticket,
        }
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartSupport AI may occasionally generate "
    "incorrect information. Please contact human "
    "support for account-specific or sensitive issues."
)
