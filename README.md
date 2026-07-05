# MedPulse AI: Multimodal Clinical RAG & Document Intelligence

🚀 **[Live Demo on Render](https://medpulse-ai-wpjb.onrender.com)**

MedPulse AI is a Retrieval-Augmented Generation (RAG) platform designed for clinical document search, analysis, and medical image interpretation.

---

## 🚀 Key Highlights

*   **Multimodal Ingestion Pipeline:** Processes PDFs and images (PNG, JPG, JPEG). It performs semantic text chunking and extracts embedded images from clinical reports.
*   **Vision LLM Interpretation:** Utilizes **Llama 4 Vision** to generate detailed descriptions of clinical scans, X-rays, and medical charts, indexing them directly into the search index.
*   **Hybrid Search & Retrieval:** Uses LangChain's `EnsembleRetriever` to fuse dense semantic vector search (ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`) with sparse keyword matching (BM25) to optimize lookup coverage.
*   **Conversational Query Rewriter:** Includes an LLM-driven query reformulation chain to rewrite conversational follow-up inputs into standalone, context-complete search queries.
*   **Production API & Dashboard:** Fast, streaming backend served via FastAPI endpoints and a simple browser-based client workspace.

---

## 🛠️ Project Architecture

```
                       +---------------------------------------+
                       |   PDF Clinical Reports & Medical Img  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +-------------------+-------------------+
                       |        Multimodal Ingestion           |
                       +-------+-----------------------+-------+
                               |                       |
                               | (Text Chunks)         | (Clinical Images)
                               v                       v
                       +---------------+       +---------------+
                       | Semantic      |       | Llama 4       |
                       | Chunking      |       | Vision LLM    |
                       +-------+-------+       +-------+-------+
                               |                       | (Descriptions)
                               +-----------+-----------+
                                           |
                                           v
                       +-------------------+-------------------+
                       |    Hybrid Indexing (ChromaDB + BM25)  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +-------------------+-------------------+
                       |   FastAPI App (Streaming Endpoint)    | <--- Query & History
                       +-------------------+-------------------+
                                           |
                                           v
                       +-------------------+-------------------+
                       | Hybrid Search + Conversational Chains |
                       +-------------------+-------------------+
                                           |
                                           v
                       +-------------------+-------------------+
                       |      MedPulse Clinical Workspace      |
                       +---------------------------------------+
```

---

## 📋 Technology Stack

*   **LLM Engine:** Groq API (`llama-3.1-8b-instant` / `llama-4-scout-17b`)
*   **Database & Indexing:** ChromaDB, BM25 Retriever
*   **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API
*   **Libraries:** LangChain, FastAPI, PyMuPDF (fitz), Python

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.10+ installed
*   Groq API Key
*   HuggingFace Hub Token

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_hub_token_here
```

### 3. Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Running the Server
```powershell
python -m uvicorn main:app --reload
```
*   Access the local workspace interface at: **`http://127.0.0.1:8000/`**
*   Access the live demo on Render: **`https://medpulse-ai-wpjb.onrender.com`**
*   Access the local Swagger API documentation at: **`http://127.0.0.1:8000/docs`**

