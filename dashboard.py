
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional dependency for calling Ollama's local API
import requests


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartSupport AI | Admin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DATABASE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DB_PATH = BASE_DIR / "support_tickets.db"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 0;
    }

    .sub-title {
        color: #a1a1aa;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #60a5fa;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. DATABASE FUNCTIONS
# =========================================================

def get_connection():
    """Connect to the existing support ticket database."""
    return sqlite3.connect(DB_PATH)


def load_tickets():
    """Load all tickets from the existing SQLite database."""

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        return pd.DataFrame()

    try:
        conn = get_connection()

        df = pd.read_sql_query(
            """
            SELECT
                id,
                ticket_id,
                customer_message,
                category,
                department,
                priority,
                status,
                summary,
                created_at
            FROM tickets
            ORDER BY id DESC
            """,
            conn
        )

        conn.close()

        return df

    except Exception as e:
        st.error(f"Error loading tickets: {e}")
        return pd.DataFrame()


def update_ticket_status(ticket_id, new_status):
    """Update the status of an existing ticket."""

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE tickets
            SET status = ?
            WHERE ticket_id = ?
            """,
            (new_status, ticket_id)
        )

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        st.error(f"Unable to update ticket: {e}")
        return False


# =========================================================
# 5. AI SUPPORT INSIGHTS
# =========================================================

def generate_ai_insights(df):
    """
    Generate support analytics using the locally running
    Ollama model.
    """

    if df.empty:
        return "There are no support tickets available to analyze."

    ticket_data = df[
        [
            "ticket_id",
            "category",
            "department",
            "priority",
            "status",
            "summary"
        ]
    ].fillna("Not specified")

    ticket_text = ticket_data.to_string(index=False)

    prompt = f"""
You are an AI customer support analytics assistant.

Analyze the following customer support ticket data.

TICKET DATA:
{ticket_text}

Provide a concise business report containing:

1. Most common customer issues.
2. Recurring problems that may require attention.
3. High-priority or unresolved support concerns.
4. Practical recommendations for improving customer support.

Rules:
- Use only the information in the ticket data.
- Do not invent ticket counts or customer details.
- Clearly state when information is insufficient.
- Use simple professional language.
- Format the response with clear headings and bullet points.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result.get(
            "response",
            "The AI did not return a response."
        )

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running or is unavailable. "
            "Start Ollama and try again."
        )

    except requests.exceptions.Timeout:
        return (
            "The AI request timed out. "
            "Try again after ensuring Ollama is responsive."
        )

    except Exception as e:
        return f"AI insight generation failed: {e}"


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🛠️ SmartSupport AI")

    st.caption("Admin Control Center")

    st.divider()

    st.subheader("Dashboard Controls")

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True
    ):
        st.rerun()

    if st.button(
        "🧠 Generate AI Insights",
        use_container_width=True
    ):
        st.session_state.generate_insights = True

    st.divider()

    st.caption("SmartSupport AI")
    st.caption("Local AI Customer Support System")


# =========================================================
# 7. LOAD DATA
# =========================================================

df = load_tickets()


# =========================================================
# 8. DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Support Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Monitor customer tickets, track support performance, '
    'and analyze recurring issues with AI.'
    '</div>',
    unsafe_allow_html=True
)


if df.empty:
    st.warning(
        "No tickets found in the database. "
        "Create a support ticket using the chatbot first."
    )

    st.stop()


# =========================================================
# 9. CLEAN DATA
# =========================================================

for column in [
    "category",
    "department",
    "priority",
    "status",
    "summary",
    "customer_message"
]:
    df[column] = df[column].fillna("Not specified")

df["priority"] = df["priority"].astype(str).str.strip()
df["status"] = df["status"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()
df["department"] = df["department"].astype(str).str.strip()


# =========================================================
# 10. KPI METRICS
# =========================================================

total_tickets = len(df)

open_tickets = df["status"].str.lower().isin(
    ["open", "pending", "in progress", "in_progress"]
).sum()

resolved_tickets = df["status"].str.lower().isin(
    ["resolved", "closed"]
).sum()

high_priority_tickets = df["priority"].str.lower().isin(
    ["high", "urgent", "critical"]
).sum()

resolution_rate = (
    (resolved_tickets / total_tickets) * 100
    if total_tickets > 0
    else 0
)


st.markdown(
    '<div class="section-title">📌 Key Performance Indicators</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Tickets",
        total_tickets
    )

with col2:
    st.metric(
        "Open / Pending",
        int(open_tickets)
    )

with col3:
    st.metric(
        "Resolved / Closed",
        int(resolved_tickets)
    )

with col4:
    st.metric(
        "High Priority",
        int(high_priority_tickets)
    )

with col5:
    st.metric(
        "Resolution Rate",
        f"{resolution_rate:.1f}%"
    )


st.divider()


# =========================================================
# 11. FILTERS
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Search and Filter Tickets</div>',
    unsafe_allow_html=True
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    search_text = st.text_input(
        "Search by ticket ID or customer message",
        placeholder="Enter ticket ID or keyword"
    )

with filter_col2:
    status_options = ["All"] + sorted(
        df["status"].unique().tolist()
    )

    selected_status = st.selectbox(
        "Filter by Status",
        status_options
    )

with filter_col3:
    priority_options = ["All"] + sorted(
        df["priority"].unique().tolist()
    )

    selected_priority = st.selectbox(
        "Filter by Priority",
        priority_options
    )


filtered_df = df.copy()

if search_text:
    search_mask = (
        filtered_df["ticket_id"].astype(str).str.contains(
            search_text,
            case=False,
            na=False
        )
        |
        filtered_df["customer_message"].astype(str).str.contains(
            search_text,
            case=False,
            na=False
        )
    )

    filtered_df = filtered_df[search_mask]


if selected_status != "All":
    filtered_df = filtered_df[
        filtered_df["status"] == selected_status
    ]


if selected_priority != "All":
    filtered_df = filtered_df[
        filtered_df["priority"] == selected_priority
    ]


st.caption(
    f"Showing {len(filtered_df)} of {total_tickets} tickets"
)


# =========================================================
# 12. TICKET TABLE
# =========================================================

st.markdown(
    '<div class="section-title">🎫 Customer Support Tickets</div>',
    unsafe_allow_html=True
)

display_columns = [
    "ticket_id",
    "category",
    "department",
    "priority",
    "status",
    "summary",
    "created_at"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 13. UPDATE TICKET STATUS
# =========================================================

st.markdown(
    '<div class="section-title">✏️ Manage Ticket Status</div>',
    unsafe_allow_html=True
)

ticket_options = filtered_df["ticket_id"].astype(str).tolist()

if ticket_options:

    selected_ticket = st.selectbox(
        "Select a ticket to update",
        ticket_options
    )

    current_status = filtered_df.loc[
        filtered_df["ticket_id"].astype(str) == selected_ticket,
        "status"
    ].iloc[0]

    status_choices = [
        "Open",
        "In Progress",
        "Pending",
        "Resolved",
        "Closed"
    ]

    if current_status not in status_choices:
        status_choices.insert(0, current_status)

    default_index = status_choices.index(current_status)

    new_status = st.selectbox(
        "New Status",
        status_choices,
        index=default_index
    )

    if st.button(
        "💾 Update Ticket",
        type="primary"
    ):

        if update_ticket_status(
            selected_ticket,
            new_status
        ):
            st.success(
                f"Ticket {selected_ticket} updated to {new_status}."
            )

            st.rerun()

else:
    st.info("No tickets match the selected filters.")


st.divider()


# =========================================================
# 14. ANALYTICS CHARTS
# =========================================================

st.markdown(
    '<div class="section-title">📈 Support Analytics</div>',
    unsafe_allow_html=True
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Tickets by Priority")

    priority_counts = (
        df["priority"]
        .value_counts()
        .rename_axis("Priority")
        .reset_index(name="Tickets")
    )

    st.bar_chart(
        priority_counts.set_index("Priority"),
        y="Tickets"
    )

with chart_col2:
    st.subheader("Tickets by Status")

    status_counts = (
        df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Tickets")
    )

    st.bar_chart(
        status_counts.set_index("Status"),
        y="Tickets"
    )


chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Tickets by Category")

    category_counts = (
        df["category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Tickets")
    )

    st.bar_chart(
        category_counts.set_index("Category"),
        y="Tickets"
    )

with chart_col4:
    st.subheader("Tickets by Department")

    department_counts = (
        df["department"]
        .value_counts()
        .rename_axis("Department")
        .reset_index(name="Tickets")
    )

    st.bar_chart(
        department_counts.set_index("Department"),
        y="Tickets"
    )


# =========================================================
# 15. AI-POWERED SUPPORT INSIGHTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 AI Support Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">'
    'Analyze existing support tickets using your local '
    'Llama 3.2 model. Your ticket data is processed '
    'through the locally running Ollama service.'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.get("generate_insights", False):

    with st.spinner(
        "AI is analyzing support tickets. Please wait..."
    ):

        insights = generate_ai_insights(df)

    st.session_state.ai_insights = insights
    st.session_state.generate_insights = False


if st.session_state.get("ai_insights"):

    st.subheader("AI-Generated Report")

    st.markdown(
        st.session_state.ai_insights
    )

else:
    st.info(
        "Click 'Generate AI Insights' in the sidebar "
        "to analyze your current tickets."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.caption(
    "SmartSupport AI | Local AI Customer Support & Analytics"
)


