"""Streamlit app entrypoint for the AI Research Assistant."""

import streamlit as st
import logging
from document_processor import load_documents, preprocess_document, split_documents
from rag import index_chunks, retrieve_relevant_chunks
from llm import generate_answer
from prompts import build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Render the main Streamlit interface."""
    st.set_page_config(page_title="AI Research Assistant", layout="wide")

    st.title("AI Research Assistant")
    st.caption("Upload PDFs, ask questions, and get answers with source citations")

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📄 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to process"
        )

        if st.button("Process Documents", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing documents..."):
                try:
                    raw_docs = load_documents(uploaded_files)
                    if not raw_docs:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        return

                    processed_docs = [preprocess_document(doc) for doc in raw_docs]

                    chunks = split_documents(processed_docs)
                    if not chunks:
                        st.error("No chunks were created from the documents.")
                        return

                    index_chunks(chunks)

                    st.session_state.processed = True
                    st.success(f"Successfully processed {len(uploaded_files)} file(s) into {len(chunks)} chunks!")

                    st.session_state.chat_history = []

                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    logger.error(f"Error processing documents: {e}", exc_info=True)

    # Main chat interface
    st.header("Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"📄 {source['source']} (Page {source['page']})")

    if prompt := st.chat_input("Ask a question about your documents"):
        if not st.session_state.processed:
            st.warning("Please upload and process documents first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retrieved_chunks = retrieve_relevant_chunks(prompt, top_k=5)

                    llm_prompt = build_prompt(retrieved_chunks, prompt)

                    answer = generate_answer(llm_prompt)

                    sources = []
                    for chunk in retrieved_chunks:
                        sources.append({
                            "source": chunk['metadata'].get('source', 'Unknown'),
                            "page": chunk['metadata'].get('page', '?')
                        })

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    st.write(answer)

                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.caption(f"📄 {source['source']} (Page {source['page']})")

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logger.error(f"Error generating answer: {e}", exc_info=True)

if __name__ == "__main__":
    main()