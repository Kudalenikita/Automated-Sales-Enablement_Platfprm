# rag/rag_engine.py
# Fully updated & working version for Python 3.13 + Streamlit Cloud (Jan 2026)
# Uses chromadb-client (HttpClient) + external Chroma server → NO local persistence, NO NumPy/build issues

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIGURATION =========================
# →→→ UPDATE THESE WITH YOUR EXTERNAL CHROMA INSTANCE DETAILS ←←←

CHROMA_HOST = os.getenv("CHROMA_HOST", "https://your-chroma-instance.trychroma.com")  # e.g. from trychroma.com or Render
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "443"))                                 # 443 for HTTPS, 8000 for local/http
CHROMA_SSL = os.getenv("CHROMA_SSL", "True").lower() == "true"                     # True for HTTPS
CHROMA_AUTH_TOKEN = os.getenv("CHROMA_AUTH_TOKEN", None)                          # Bearer token if required (trychroma.com free tier needs it)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "sales_features"  # Change if you use a different collection

# ==================================================================

def get_vector_client_and_collection():
    """Initialize Chroma HTTP client and return client, collection, embedding_func"""
    
    headers = {}
    if CHROMA_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {CHROMA_AUTH_TOKEN}"
    # Optional: add this header for trychroma.com free tier
    # headers["X-Chroma-Project"] = "default"  # sometimes needed

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        ssl=CHROMA_SSL,
        headers=headers
    )

    # Test connection (optional but helpful on first deploy)
    try:
        client.heartbeat()  # This will raise if cannot connect
        print(f"Connected to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
    except Exception as e:
        print(f"Failed to connect to Chroma server: {e}")
        raise

    embedding_func = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    return client, collection, embedding_func


# Global instances (initialized once at import time)
try:
    client, collection, embedding_func = get_vector_client_and_collection()
    print("Chroma vector DB initialized successfully via HTTP client.")
except Exception as e:
    print(f"Failed to initialize Chroma client: {e}")
    raise


def ingest_to_vector_db(text: str, metadata: dict):
    """
    Ingest a single text chunk with metadata into the vector DB
    """
    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    print(f"Ingested document with id: {doc_id[:12]}...")


def query_vector_db(
    query: str,
    customer_filter: str = None,
    n_results: int = 10
):
    """
    Query the vector database.
    Returns: list of dicts → [{"text": "...", "metadata": {...}}, ...]
    Compatible with pitch_deck.py
    """
    where = {"customer_name": customer_filter} if customer_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    retrieved = []
    for doc_text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text": doc_text,
            "metadata": meta or {},
            "distance": distance  # optional: useful for debugging/ranking
        })

    return retrieved


# Optional: utility to clear collection (for testing)
def reset_collection():
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} deleted.")
    global collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    print("New empty collection created.")
