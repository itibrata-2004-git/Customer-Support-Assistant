
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "shopease_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"


def get_vectorstore():
    """Connect to the existing Chroma database."""

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vectorstore


def retrieve_documents(question: str, k: int = 4):
    """Retrieve the most relevant document chunks."""

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    return results


if __name__ == "__main__":
    query = input("Enter a test question: ")

    results = retrieve_documents(query)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Score:", score)
        print(doc.page_content)
