"""Test script for the RAG module.

This script tests the index_chunks and retrieve_relevant_chunks functions
using dummy data to verify the reset functionality and duplicate avoidance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import index_chunks, retrieve_relevant_chunks
from embeddings import generate_embeddings

def test_reset_and_duplicate_avoidance():
    """Test that index_chunks resets the collection and avoids duplicates within a session."""
    print("=== Testing RAG reset and duplicate avoidance ===")

    # First batch of chunks
    chunks1 = [
        {
            "text": "This is the first chunk about apples.",
            "metadata": {
                "source": "apple.txt",
                "page": 1,
                "chunk_id": "apple-1"
            }
        },
        {
            "text": "This is the second chunk about apples.",
            "metadata": {
                "source": "apple.txt",
                "page": 1,
                "chunk_id": "apple-2"
            }
        },
        {
            "text": "This is a chunk about bananas.",
            "metadata": {
                "source": "banana.txt",
                "page": 2,
                "chunk_id": "banana-1"
            }
        }
    ]

    # Index the first batch
    print("\n--- Indexing first batch ---")
    index_chunks(chunks1)

    # Check collection count
    from rag import _get_or_create_collection
    collection = _get_or_create_collection()
    count_after_first = collection.count()
    print(f"Count after first indexing: {count_after_first}")
    assert count_after_first == 3, f"Expected 3 chunks, got {count_after_first}"

    # Try to index the same chunks again (should avoid duplicates)
    print("\n--- Re-indexing the same batch (duplicate avoidance) ---")
    index_chunks(chunks1)
    collection = _get_or_create_collection()  # Get the current collection after potential reset
    count_after_duplicate = collection.count()
    print(f"Count after re-indexing same batch: {count_after_duplicate}")
    assert count_after_duplicate == 3, f"Expected still 3 chunks (no duplicates), got {count_after_duplicate}"

    # Second batch of chunks (different content)
    chunks2 = [
        {
            "text": "This is about oranges.",
            "metadata": {
                "source": "orange.txt",
                "page": 1,
                "chunk_id": "orange-1"
            }
        },
        {
            "text": "Another orange chunk.",
            "metadata": {
                "source": "orange.txt",
                "page": 2,
                "chunk_id": "orange-2"
            }
        }
    ]

    # Index the second batch (should reset and only have these two)
    print("\n--- Indexing second batch (should reset) ---")
    index_chunks(chunks2)
    collection = _get_or_create_collection()
    count_after_second = collection.count()
    print(f"Count after second indexing: {count_after_second}")
    assert count_after_second == 2, f"Expected 2 chunks after reset, got {count_after_second}"

    # Verify that the first batch is gone by trying to retrieve
    print("\n--- Testing retrieval ---")
    # Query for something from the first batch
    results = retrieve_relevant_chunks("apple", top_k=5)
    print(f"Retrieved chunks for query 'apple': {len(results)}")
    # Should be empty or very low similarity because the collection was reset
    # Actually, we reset the collection, so there should be no apple chunks.
    # However, note that the reset happens in index_chunks, so after indexing chunks2,
    # the collection only has orange chunks.
    for i, chunk in enumerate(results):
        print(f"  Result {i}: {chunk['text'][:50]}... (similarity: {chunk.get('similarity', 0):.2f})")

    # Query for something from the second batch
    results2 = retrieve_relevant_chunks("orange", top_k=5)
    print(f"\nRetrieved chunks for query 'orange': {len(results2)}")
    for i, chunk in enumerate(results2):
        print(f"  Result {i}: {chunk['text'][:50]}... (similarity: {chunk.get('similarity', 0):.2f})")
    assert len(results2) > 0, "Should have retrieved orange chunks"

    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    try:
        test_reset_and_duplicate_avoidance()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)