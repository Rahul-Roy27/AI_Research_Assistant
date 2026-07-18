"""LLM utilities for the AI Research Assistant.

This module handles initialization of the Gemini model and generation of answers.
Uses the latest Google GenAI SDK.
"""

import os
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global variable for the GenAI client (singleton pattern)
_genai_client: Any = None

# Model name - using a current Gemini model
MODEL_NAME = "gemini-2.5-flash"


def _get_genai_client():
    """Lazy load and return the GenAI client.

    Returns:
        genai.Client: The Google GenAI client instance.
    """
    global _genai_client
    if _genai_client is None:
        if not GENAI_AVAILABLE:
            raise ImportError(
                "Google GenAI package not installed. "
                "Please install it with: pip install google-genai"
            )
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it to use Gemini."
            )
        try:
            logger.info("Initializing Google GenAI client")
            _genai_client = genai.Client(api_key=api_key)
            logger.info("GenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GenAI client: {e}")
            raise RuntimeError(f"Could not initialize GenAI client: {e}") from e
    return _genai_client


def generate_answer(prompt: str) -> str:
    """Generate an answer using the Gemini model.

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
        client = _get_genai_client()
        logger.debug(f"Generating answer for prompt (first 100 chars): {prompt[:100]}...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        answer = response.text
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise RuntimeError(f"Failed to generate answer: {e}") from e