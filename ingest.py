
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# -------------------------------
# Project configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "company_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "shopease_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"


def load_documents():
    """Load PDF, TXT, and Markdown files."""

    documents = []

    # Load text files
    for extension in ["*.txt", "*.md"]:
        loader = DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob=extension,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )

        documents.extend(loader.load())

    # Load PDF files
    for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        raise ValueError(
            f"No documents found in {DOCUMENTS_DIR}"
        )

    return documents


def create_vector_database():
    """Chunk documents, embed them, and save to Chroma."""

    print("Loading company documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} document pages/files.")

    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    # Initialize local embedding model
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Create persistent Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("Vector database created successfully.")
    print(f"Database location: {CHROMA_DIR}")

    return vectorstore


if __name__ == "__main__":
    create_vector_database()
