# AI Research Assistant

A modular Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions. The system retrieves relevant document chunks using semantic search and generates context-aware answers using Google's Gemini 2.5 Flash.

---

## Live Demo

Streamlit link :- https://ai-research-assistant-by-rahulroy27.streamlit.app/

---

## Features

- Upload and process multiple PDF documents
- Semantic search using SentenceTransformer embeddings
- AI-powered question answering with Gemini 2.5 Flash
- Source citations with filenames and page numbers
- Persistent vector storage using ChromaDB
- Interactive Streamlit chat interface
- Clean, modular architecture

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Text Splitting | LangChain Text Splitters |

---

## Project Structure

```text
ai-research-assistant/
│
├── app.py
├── config.py
├── document_processor.py
├── embeddings.py
├── llm.py
├── prompts.py
├── rag.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-research-assistant
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Usage

1. Upload one or more PDF documents.
2. Click **Process Documents**.
3. Ask questions in natural language.
4. Receive AI-generated answers with supporting source citations.

---

## Architecture

```text
PDF Upload
     │
     ▼
Text Extraction (PyMuPDF)
     │
     ▼
Chunking (LangChain)
     │
     ▼
Embedding Generation (SentenceTransformers)
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Semantic Retrieval
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Answer + Source Citations
```

---

## Future Improvements

- Support for DOCX and TXT documents
- Hybrid search (keyword + vector)
- Conversation memory
- Streaming AI responses
- Literature review generation
- Chat history export
- Docker deployment

---

## License

This project is intended for educational and portfolio purposes.