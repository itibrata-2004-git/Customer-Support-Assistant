# Customer-Support-Assistant

An AI-powered customer support system built with **Ollama, RAG, LangGraph, Streamlit, and SQLite**. SmartSupport AI answers customer questions using a knowledge base, maintains conversation context, creates support tickets, and provides an analytics dashboard for administrators.

## Project Overview

SmartSupport AI is designed to simulate a customer support workflow using a locally running large language model. Customers can ask questions about supported topics such as shipping, returns, billing, and company policies. The system retrieves relevant information from its knowledge base and generates a response.

When a request needs human assistance, the system supports escalation through ticket creation. The admin dashboard provides ticket summaries, filtering, status management, and support analytics.

## Features

- **Local LLM:** Uses Ollama with the `llama3.2:3b` model.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant content from a local knowledge base to ground responses.
- **LangGraph agent workflow:** Coordinates the support workflow and available tools.
- **Conversation memory:** Stores conversation history and supports separate chat sessions using thread IDs.
- **Automatic support tickets:** Creates tickets for customer support requests and escalation.
- **Ticket management:** View tickets and update their status from the dashboard.
- **Support analytics:** Visualize tickets by priority, status, category, and department, along with key performance indicators.
- **Evaluation and tests:** Includes database tests and an evaluation script for checking agent behavior.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Local language model | Ollama (`llama3.2:3b`) |
| Agent workflow | LangGraph |
| Knowledge retrieval | RAG, ChromaDB |
| Data storage | SQLite |
| Data processing | Python, Pandas |
| Model / data utilities | Scikit-learn, Joblib |
| Testing | Pytest |

## Application Screenshots

Add your screenshots to the `screenshots/` folder using the filenames below. These Markdown image links will display once the files are committed to the repository.

### Chatbot

![SmartSupport AI Chatbot]<img width="1917" height="970" alt="Screenshot 2026-09-28 124022" src="https://github.com/user-attachments/assets/9fac665b-40f5-4a12-95a5-1d213124a93c" />


### Support Analytics Dashboard

![Support Analytics Dashboard]<img width="1917" height="962" alt="Screenshot 2026-09-28 124103" src="https://github.com/user-attachments/assets/940bc7b9-0d28-4015-b4bf-2c87438392e3" />
<img width="1917" height="976" alt="Screenshot 2026-09-28 124152" src="https://github.com/user-attachments/assets/58d1d742-221e-4a14-a284-114cb325670d" />
### Ticket Management

![Ticket Management]<img width="1917" height="910" alt="Screenshot 2026-09-28 124117" src="https://github.com/user-attachments/assets/636769ec-94c3-4b89-8830-51195d8c2cbc" />

## Project Structure

```text
SmartSupport-AI/
├── App/
│   └── frontend/
│       ├── app.py                 # Streamlit customer chatbot
│       └── dashboard.py           # Admin dashboard and analytics
├── agent/
│   ├── graph.py                   # LangGraph agent workflow
│   └── tools.py                   # Agent tools and support actions
├── rag/                            # Knowledge base and RAG components
├── tests/
│   ├── __init__.py
│   └── test_database.py            # Database tests
├── evaluate_agent.py               # Agent evaluation script
├── check_database.py               # Database inspection utility
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

The exact files may vary depending on the version of the project you upload.

## Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by your dependencies
- Git
- [Ollama](https://ollama.com/download)
- A local Ollama model: `llama3.2:3b`

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartSupport-AI.git
cd SmartSupport-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Ollama and download the model

Install Ollama, then run:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running. You can verify the installed model with:

```bash
ollama list
```

### 5. Run the customer support chatbot

From the project root, open a terminal and run:

```bash
python -m streamlit run App/frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Run the admin dashboard

Open a **second terminal** in the same project folder. Activate the virtual environment there, then run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run App/frontend/dashboard.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502) in your browser.

Keep both Streamlit terminals and the Ollama service running while using the application. The chatbot and dashboard run as separate local applications.

## Usage

1. Start Ollama and ensure the configured model is available.
2. Launch the chatbot and ask a question about a supported customer service topic.
3. Start a new chat when you want a separate conversation.
4. If the workflow creates a support ticket, open the admin dashboard.
5. Refresh the dashboard to load current ticket data.
6. Search or filter tickets and update ticket status using the dashboard controls.
7. Use the analytics and AI insights features to review support activity.

## Testing and Evaluation

Run the database tests from the project root:

```bash
python -m pytest tests/ -v
```

Run the agent evaluation script:

```bash
python evaluate_agent.py
```

The evaluation script checks selected agent behaviors. Automated keyword or source checks are useful for basic testing, but they do not guarantee that every generated answer is factually correct. Review responses and retrieved sources when validating changes.

## Data and Privacy

This project uses local SQLite databases for ticket records and conversation memory, and may use local ChromaDB storage for retrieval.

- Local database files and private data should not be committed to a public repository.
- Do not upload API keys, passwords, `.env` files, private customer information, or conversation logs.
- The `.gitignore` file should exclude local databases, vector-store data, model artifacts, and secrets.
- A fresh clone may require initializing its databases and knowledge base through the project's setup or application workflow.

The repository contains application source code; it does not include the Ollama model itself. The model must be downloaded separately through Ollama.

## Current Project Status

The application has been run locally with the chatbot and admin dashboard on separate Streamlit ports. The project includes RAG-based responses, ticket handling, conversation memory, dashboard analytics, and basic testing/evaluation utilities.

## Future Improvements

- Add authentication and role-based access for administrators.
- Improve automated evaluation with a larger set of test questions and expected answers.
- Add deployment configuration for a hosted environment.
- Add monitoring, structured logging, and more detailed reporting.
- Expand the support knowledge base and ticket categories.

## Author

**Itibrata Sahoo**

GitHub: [Itibrata-2004-git](https://github.com/Itibrata-2004-git)

---

If you find this project useful, feel free to star th
