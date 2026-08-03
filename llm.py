"""LLM utilities for the AI Research Assistant using Groq.

This module handles initialization of the Groq model and generation of answers.
"""

import os
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

from groq import Groq

logger = logging.getLogger(__name__)

# Global variable for the Groq client (singleton pattern)
_groq_client: Any = None

# Model name - using Groq's fast model
MODEL_NAME = "llama-3.3-70b-versatile"


def _get_groq_client():
    """Lazy load and return the Groq client.

    Returns:
        Groq: The Groq client instance.
    """
    global _groq_client
    if _groq_client is None:
        # Try GROQ_API_KEY first, then fall back to GOOGLE_API_KEY for backward compatibility
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it to use Groq (you can put your Groq API key here)."
                " For backward compatibility, GOOGLE_API_KEY is also accepted."
            )
        try:
            logger.info("Initializing Groq client")
            _groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError(f"Could not initialize Groq client: {e}") from e
    return _groq_client


def generate_answer(prompt: str) -> str:
    """Generate an answer using the Groq model.

    Args:
        prompt: The prompt string to send to the model.

    Returns:
        str: The generated answer text.

    Raises:
        RuntimeError: If there is an error generating the answer.
    """
    if not prompt.strip():
        logger.warning("Empty prompt provided for generation")
        return ""

    try:
        client = _get_groq_client()
        logger.debug(f"Generating answer for prompt (first 100 chars): {prompt[:100]}...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
        )
        answer = chat_completion.choices[0].message.content
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Check if it's a rate limit or other known error
        if "rate limit" in str(e).lower() or "429" in str(e):
            friendly_msg = ("I'm unable to generate an answer due to API rate limits. "
                            "Please try again later.")
            return friendly_msg
        raise RuntimeError(f"Failed to generate answer: {e}") from e