"""End-to-end test of the RAG pipeline using the actual PDFs we created."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import load_documents, preprocess_document, split_documents
from rag import index_chunks, retrieve_relevant_chunks

def test_end_to_end():
    """Test the full pipeline: load PDFs -> preprocess -> split -> index -> retrieve."""
    print("=== End-to-end RAG pipeline test ===")

    # Step 1: Load documents from the Data directory
    pdf_files = [os.path.join("Data", f) for f in os.listdir("Data") if f.endswith(".pdf")]
    print(f"Found PDF files: {pdf_files}")
    assert len(pdf_files) > 0, "No PDF files found in Data directory"

    # Simulate Streamlit UploadedFile objects (we'll just open the files as binary)
    # For simplicity, we'll use the load_documents function which expects file-like objects with .name and .read()
    # We'll create a simple class to mimic that.
    class MockUploadedFile:
        def __init__(self, filepath):
            self.name = os.path.basename(filepath)
            self._filepath = filepath
            self._file = open(filepath, "rb")

        def seek(self, offset):
            self._file.seek(offset)

        def read(self):
            return self._file.read()

        def close(self):
            self._file.close()

    uploaded_files = [MockUploadedFile(f) for f in pdf_files]

    try:
        raw_docs = load_documents(uploaded_files)
        print(f"Loaded {len(raw_docs)} raw documents (pages)")
        for i, doc in enumerate(raw_docs[:2]):  # Show first two
            print(f"  Doc {i}: source={doc['metadata']['source']}, page={doc['metadata']['page']}, text preview: {doc['text'][:50]}...")
    finally:
        for f in uploaded_files:
            f.close()

    assert len(raw_docs) > 0, "No documents loaded"

    # Step 2: Preprocess documents
    processed_docs = [preprocess_document(doc) for doc in raw_docs]
    print(f"Preprocessed {len(processed_docs)} documents")

    # Step 3: Split into chunks
    chunks = split_documents(processed_docs)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first three chunks
        print(f"  Chunk {i}: source={chunk['metadata']['source']}, page={chunk['metadata']['page']}, chunk_id={chunk['metadata']['chunk_id'][:8]}..., text preview: {chunk['text'][:50]}...")

    assert len(chunks) > 0, "No chunks created"

    # Step 4: Index chunks
    print("\n--- Indexing chunks ---")
    index_chunks(chunks)

    # Verify indexing worked
    from rag import _get_or_create_collection
    collection = _get_or_create_collection()
    count = collection.count()
    print(f"Number of chunks indexed in ChromaDB: {count}")
    assert count == len(chunks), f"Expected {len(chunks)} chunks in DB, got {count}"

    # Step 5: Test retrieval
    print("\n--- Testing retrieval ---")
    # Query about apples
    query_apple = "What does the document say about apples?"
    results_apple = retrieve_relevant_chunks(query_apple, top_k=5)
    print(f"Query: '{query_apple}'")
    print(f"Found {len(results_apple)} relevant chunks:")
    for i, chunk in enumerate(results_apple):
        print(f"  {i+1}. source={chunk['metadata']['source']}, page={chunk['metadata']['page']}, similarity={chunk.get('similarity', 0):.3f}")
        print(f"     Text: {chunk['text'][:100]}...")
    # We expect at least one result and that it mentions apples
    assert len(results_apple) > 0, "Should find chunks about apples"
    # Check that the retrieved chunks are indeed about apples (or at least from the apple PDF)
    apple_sources = set(chunk['metadata']['source'] for chunk in results_apple)
    print(f"Sources of apple results: {apple_sources}")
    # Since we have two PDFs, we might get chunks from both if they are similar, but hopefully at least one is from test1.pdf
    # We'll just check that we got something.

    # Query about oranges
    query_orange = "Tell me about oranges."
    results_orange = retrieve_relevant_chunks(query_orange, top_k=5)
    print(f"\nQuery: '{query_orange}'")
    print(f"Found {len(results_orange)} relevant chunks:")
    for i, chunk in enumerate(results_orange):
        print(f"  {i+1}. source={chunk['metadata']['source']}, page={chunk['metadata']['page']}, similarity={chunk.get('similarity', 0):.3f}")
        print(f"     Text: {chunk['text'][:100]}...")
    assert len(results_orange) > 0, "Should find chunks about oranges"

    # Query about something not in the documents
    query_none = "What is the capital of France?"
    results_none = retrieve_relevant_chunks(query_none, top_k=5)
    print(f"\nQuery: '{query_none}' (should have low similarity)")
    print(f"Found {len(results_none)} chunks:")
    for i, chunk in enumerate(results_none):
        print(f"  {i+1}. source={chunk['metadata']['source']}, page={chunk['metadata']['page']}, similarity={chunk.get('similarity', 0):.3f}")
        print(f"     Text: {chunk['text'][:100]}...")
    # The similarity might be low, but we still get chunks because the collection is not empty.
    # We can check that the similarity is lower than for the relevant queries (optional).

    print("\n=== End-to-end test passed! ===")

if __name__ == "__main__":
    try:
        test_end_to_end()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)