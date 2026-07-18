"""Embedding utilities for the AI Research Assistant.

This module defines a function for generating vector embeddings for text data.
Uses the all-MiniLM-L6-v2 SentenceTransformer model loaded as a singleton.
"""

import logging
from typing import List, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global variable to hold the model instance (singleton pattern)
_model: Any = None


def _get_model() -> SentenceTransformer:
    """Lazy load and return the SentenceTransformer model.

    Returns:
        SentenceTransformer: The all-MiniLM-L6-v2 model instance.
    """
    global _model
    if _model is None:
        try:
            logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e
    return _model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text strings.

    Args:
        texts: A list of text strings to convert into embeddings.

    Returns:
        A list of numeric embeddings (each embedding is a list of floats).

    Example:
        >>> embeddings = generate_embeddings(["Hello world", "Machine learning"])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])  # all-MiniLM-L6-v2 produces 384-dim embeddings
        384
    """
    if not texts:
        return []

    try:
        model = _get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        # Convert numpy array to list of lists for JSON serialization compatibility
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e