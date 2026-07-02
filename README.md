# MedPulse AI: Multimodal Document Intelligence Platform

MedPulse AI is a lightweight Retrieval-Augmented Generation (RAG) pipeline for clinical document search and image analysis.

---

## Core Methods & Architecture

1. **Multimodal Ingestion Pipeline:**
   * PDF text extraction and semantic chunking (~800 characters).
   * Standalone/embedded image analysis using **Llama 4 Vision LLM** to generate textual descriptions of diagnostic scans for database indexing.

2. **Conversational Query Rewriting:**
   * An LLM-based query reformulation step that rewrites conversational follow-up questions (e.g., *"what issue i mean"*) into standalone, search-friendly queries using chat history context.

3. **Two-Stage Hybrid Search & Retrieval:**
   * Combines dense vector search (**ChromaDB** with sentence-transformers) and sparse keyword search (**BM25**) using an `EnsembleRetriever` to maximize retrieval coverage.
   * Persistent database reloading bypasses expensive document processing on server restarts.

---

## Tech Stack

* **LLMs:** Groq API (`llama-3.1-8b`, `llama-4-scout-17b`)
* **Database & Ingestion:** ChromaDB, BM25, PyMuPDF
* **Orchestration:** LangChain, FastAPI, Python
