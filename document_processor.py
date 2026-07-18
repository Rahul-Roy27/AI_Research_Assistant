"""Document processing utilities for the AI Research Assistant.

This module contains functions for loading PDF documents from Streamlit UploadedFile objects,
extracting text, preprocessing, and splitting into chunks for embedding and retrieval.
"""

import logging
import uuid
from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_documents(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    """Load documents from a list of Streamlit UploadedFile objects.
    Extracts text page by page using PyMuPDF.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects (file-like with .name and .read()).

    Returns:
        A list of document records with text and metadata.
        Each record has keys: 'text' (str) and 'metadata' (dict).
        Metadata contains: 'source' (filename), 'page' (page number starting from 1).
    """
    documents: List[Dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            uploaded_file.seek(0)
            bytes_data = uploaded_file.read()

            doc = fitz.open(stream=bytes_data, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if not text.strip():
                    continue
                record = {
                    "text": text,
                    "metadata": {
                        "source": file_name,
                        "page": page_num + 1,
                    }
                }
                documents.append(record)
            doc.close()
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    return documents


def preprocess_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a single document for retrieval and embedding.

    Args:
        document: A document record with keys 'text' (str) and 'metadata' (dict).

    Returns:
        A document record with normalized text and preserved metadata.
    """
    text = document.get("text", "")
    text = re.sub(r'\s+', ' ', text).strip()
    processed = document.copy()
    processed["text"] = text
    return processed


def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into smaller chunks for embedding and retrieval.

    Args:
        documents: A list of preprocessed document records.
                   Each record must have keys: 'text' (str) and 'metadata' (dict).
                   Metadata must contain: 'source' (str), 'page' (int).

    Returns:
        A list of document chunks.
        Each chunk record has keys: 'text' (str) and 'metadata' (dict).
        Metadata contains: 'source' (str), 'page' (int), 'chunk_id' (str, UUID).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        text = doc.get("text", "")
        source = doc.get("metadata", {}).get("source", "unknown")
        page = doc.get("metadata", {}).get("page", 1)
        if not text:
            continue
        text_chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(text_chunks):
            chunk_record = {
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_id": str(uuid.uuid4()),
                }
            }
            chunks.append(chunk_record)

    return chunks