# MedPulse AI: Multimodal Document Intelligence Platform

MedPulse AI is a production-grade, lightweight Retrieval-Augmented Generation (RAG) workspace designed to ingest and analyze medical clinical records (PDFs) and clinical scans (PNG/JPG/JPEG images) simultaneously. By combining textual chunking and Vision LLM-driven image description, MedPulse AI enables users to perform cross-document medical reasoning inside a unified, real-time conversational chat interface.

---

## Key Features

* **Multi-File Ingestion Workspace:** Upload multiple medical records (PDFs) and standalone diagnostic scans (such as chest X-rays or prescriptions) at the same time.
* **Scan & X-Ray Analysis (Vision LLM):** Automatically extracts embedded images from PDFs and analyzes standalone images using **Llama 4 Scout (17B)** to detect clinical abnormalities, pathologies, and fractures.
* **Conversational Query Rewriting:** Reformulates conversational, multi-turn follow-up queries (like *"what issue i mean"*) into standalone database search queries to prevent RAG retrieval failures.
* **Two-Stage Hybrid Search:** Combines dense vector search (**ChromaDB** with sentence-transformers) and sparse keyword search (**BM25**) to maximize retrieval accuracy.
* **Sleek UI with Token Streaming:** A minimal, single-window frontend styled like **Shadcn UI** built using **React** and **Tailwind CSS** that consumes FastAPI token streams for real-time response rendering.
* **File-Lock Safety:** Handles persistent ChromaDB connections dynamically on Windows to prevent resource lock write permission exceptions during workspace resets.

---

## Tech Stack

* **Backend:** FastAPI, Python, LangChain, Uvicorn
* **Database & Retrieval:** ChromaDB, BM25 Retriever
* **Vector Embeddings:** HuggingFace `all-MiniLM-L6-v2` (120MB)
* **Cloud AI Inference:** Groq API
  * **Text Model:** `llama-3.1-8b-instant`
  * **Vision Model:** `meta-llama/llama-4-scout-17b-16e-instruct`
* **Frontend:** React 18, Tailwind CSS

---

## Getting Started

### Prerequisites

Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the FastAPI server:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
👉 **http://localhost:8000**
