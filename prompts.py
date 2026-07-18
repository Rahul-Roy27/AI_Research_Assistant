"""Prompt management for the AI Research Assistant.

This module provides functions for building prompts used by the model.
"""

from typing import List, Dict, Any


def load_prompt_templates() -> Dict[str, str]:
    """Load prompt templates from configuration or files.

    Returns:
        A dictionary of prompt template names to template strings.
    """
    # Placeholder implementation - can be extended to load from files
    return {}


def build_prompt(context_chunks: List[Dict[str, Any]], question: str) -> str:
    """Build a complete prompt for the model using retrieved context and the question.

    Args:
        context_chunks: List of retrieved chunks, each with:
            - 'text': str (the chunk text)
            - 'metadata': dict with keys 'source' (str), 'page' (int), 'chunk_id' (str)
        question: The user query.

    Returns:
        A formatted prompt string.
    """
    if not context_chunks:
        return f"Answer the following question based on your knowledge: {question}"

    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk['metadata'].get('source', 'Unknown')
        page = chunk['metadata'].get('page', '?')
        text = chunk['text']
        context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AI research assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know based on the provided information.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def format_response(response: str) -> str:
    """Format the model response for display in the app.

    Args:
        response: The raw response from the model.

    Returns:
        A formatted response string.
    """
    # Placeholder implementation - can be extended for formatting
    return response.strip()