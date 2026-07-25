"""Retrieval-Augmented Generation (RAG) logic for the AI Research Assistant.

This module provides a clean RAG backend using ChromaDB for vector storage and retrieval.
Handles creating/loading persistent collections, indexing document chunks with embeddings and metadata,
and retrieving top-k relevant chunks for a query.
"""

import hashlib
import logging
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

from embeddings import generate_embeddings

logger = logging.getLogger(__name__)

# Global variables for ChromaDB singleton pattern
_chroma_client: Any = None
_collection: Any = None
# In-memory set to avoid indexing duplicate chunks within the same session
_indexed_chunk_hashes: set = set()

# Collection name and persistence directory
COLLECTION_NAME = "ai_research_assistant"
PERSIST_DIRECTORY = ".chroma"


def _get_chroma_client() -> chromadb.PersistentClient:
    """Lazy load and return the ChromaDB persistent client.

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance.
    """
    global _chroma_client
    if _chroma_client is None:
        try:
            logger.info(f"Initializing ChromaDB persistent client at {PERSIST_DIRECTORY}")
            _chroma_client = chromadb.PersistentClient(
                path=PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Could not initialize ChromaDB client: {e}") from e
    return _chroma_client


def _get_or_create_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection for storing document chunks.

    Returns:
        chromadb.Collection: The ChromaDB collection instance.
    """
    global _collection
    if _collection is None:
        try:
            client = _get_chroma_client()
            logger.info(f"Getting or creating collection: {COLLECTION_NAME}")
            _collection = client.get_or_create_collection(name=COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' ready with {_collection.count()} existing items")
        except Exception as e:
            logger.error(f"Failed to get or create ChromaDB collection: {e}")
            raise RuntimeError(f"Could not access ChromaDB collection: {e}") from e
    return _collection


def _compute_chunk_hash(chunk: Dict[str, Any]) -> str:
    """Compute a unique hash for a chunk based on its text, source, and page.
    Used to avoid indexing duplicate chunks within the same session.

    Args:
        chunk: A document chunk dict with 'text' and 'metadata' (containing 'source' and 'page').

    Returns:
        str: A hexadecimal hash string representing the chunk.
    """
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown")
    page = str(metadata.get("page", 0))

    hash_string = f"{text}|{source}|{page}"
    return hashlib.md5(hash_string.encode()).hexdigest()


def index_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Index document chunks with their embeddings and metadata into ChromaDB.
    Resets the collection before indexing to ensure only the currently uploaded documents are stored.
    Avoids duplicate indexing of the same chunk (based on text, source, page) within the same session.
    Uses each chunk's chunk_id as the ChromaDB document ID.
    Generates embeddings in batches for efficiency.

    Args:
        chunks: A list of document chunk dicts, each with:
            - 'text': str (the chunk text)
            - 'metadata': dict with keys 'source' (str), 'page' (int), 'chunk_id' (str)
    """
    global _collection, _indexed_chunk_hashes
    if not chunks:
        logger.warning("No chunks provided for indexing")
        return

    try:
        client = _get_chroma_client()
        # Delete the existing collection to start fresh
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception as e:
            # Collection might not exist, which is fine
            logger.info(f"No existing collection to delete, or error: {e}")

        # Create a new collection
        collection = client.create_collection(name=COLLECTION_NAME)
        logger.info(f"Created new collection: {COLLECTION_NAME}")
        _collection = collection

        # Reset the duplicate chunk hash set for this indexing session
        _indexed_chunk_hashes = set()

        texts = []
        metadatas = []
        ids = []

        # Log the unique sources being indexed for verification
        sources_indexed = set()
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            sources_indexed.add(source)

            chunk_hash = _compute_chunk_hash(chunk)
            if chunk_hash in _indexed_chunk_hashes:
                logger.debug(f"Skipping duplicate chunk: {source} page {chunk['metadata']['page']}")
                continue

            texts.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk["metadata"]["chunk_id"])
            _indexed_chunk_hashes.add(chunk_hash)

        logger.info(f"Indexing documents from sources: {', '.join(sources_indexed)}")

        if not texts:
            logger.info("All chunks were duplicates, nothing to index")
            return

        logger.info(f"Generating embeddings for {len(texts)} chunks in batch")
        embeddings_list = generate_embeddings(texts)

        logger.info(f"Indexing {len(texts)} new chunks into ChromaDB collection")
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=texts,
            ids=ids
        )
        logger.info(f"Successfully indexed {len(texts)} chunks. Total items in collection: {collection.count()}")

    except Exception as e:
        logger.error(f"Error indexing chunks: {e}")
        raise RuntimeError(f"Failed to index chunks in ChromaDB: {e}") from e


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the top-k most relevant document chunks for a query using vector similarity.
    Returns similarity scores (computed as 1 - distance) along with the retrieved chunks.

    Args:
        query: The query text string.
        top_k: Number of top relevant chunks to retrieve (default: 5).

    Returns:
        A list of dicts, each containing:
            - 'text': str (the chunk text)
            - 'metadata': dict with keys 'source' (str), 'page' (int), 'chunk_id' (str)
            - 'similarity': float (similarity score, higher means more similar)
        Sorted by relevance (most relevant first).
    """
    if not query.strip():
        logger.warning("Empty query provided for retrieval")
        return []

    try:
        collection = _get_or_create_collection()
        if collection.count() == 0:
            logger.warning("ChromaDB collection is empty, no chunks to retrieve")
            return []

        logger.debug(f"Generating embedding for query: {query[:100]}...")
        query_embedding = generate_embeddings([query])[0]

        logger.debug(f"Querying ChromaDB for top {top_k} similar chunks")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        relevant_chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                chunk = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": similarity
                }
                relevant_chunks.append(chunk)

            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
        else:
            logger.info("No relevant chunks found for query")

        return relevant_chunks

    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        raise RuntimeError(f"Failed to retrieve chunks from ChromaDB: {e}") from e