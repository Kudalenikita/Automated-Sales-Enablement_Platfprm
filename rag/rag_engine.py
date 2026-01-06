# rag/rag_engine.py
# Updated: Returns list of dictionaries → compatible with pitch_deck.py

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_vector_client_and_collection():
    """Initialize Chroma client and collection"""
    client = chromadb.PersistentClient(path="data/chroma")
    
    embedding_func = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    
    collection = client.get_or_create_collection(
        name="sales_features",
        embedding_function=embedding_func
    )
    
    return client, collection, embedding_func


# Global instances (used across the app)
client, collection, embedding_func = get_vector_client_and_collection()


def ingest_to_vector_db(vector_client, embedding_func, text: str, metadata: dict):
    """
    Ingest a single text chunk with metadata
    """
    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )


def query_vector_db(
    vector_client,
    embedding_func,
    query: str,
    customer_filter: str = None,
    n_results: int = 10
):
    """
    Query the vector database.
    Returns: list of dicts → [{"text": "...", "metadata": {...}}, ...]
    """
    where = {"customer_name": customer_filter} if customer_filter else None
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas"]
    )
    
    if not results.get("documents") or not results["documents"][0]:
        return []
    
    retrieved = []
    for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": doc_text,
            "metadata": meta or {}
        })
    
    return retrieved