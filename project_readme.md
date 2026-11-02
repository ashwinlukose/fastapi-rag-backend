# Local RAG AI Backend Service

A modern, production-grade Retrieval-Augmented Generation (RAG) backend built with **Python**, **FastAPI**, **ChromaDB**, and **Ollama**, paired with an interactive **Streamlit** user interface. 

Designed to run 100% locally and privately on edge hardware (optimized for Apple Silicon / M-Series Macs) with zero external API costs or cloud dependencies.

---

## 🏗️ Architecture & Tech Stack

```
[ Streamlit UI ] --(HTTP / JSON)--> [ FastAPI REST Backend ]
                                          │
                                    (Text Splitting)
                                          │
                                    [ ChromaDB ] <--- (Embeddings: all-MiniLM-L6-v2)
                                          │
                                    (Context Retrieval)
                                          │
                                    [ Local LLM (Ollama / Llama 3) ]
```

* **Core Framework:** FastAPI (Asynchronous REST API with automatic Swagger documentation)
* **Vector Database:** ChromaDB (Persistent local vector storage)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Running locally via PyTorch)
* **LLM Runner:** Ollama executing `Llama 3` locally
* **Frontend Dashboard:** Streamlit (Chat interface and file ingestion pipeline)

---

## 📂 Project Structure

```text
fastapi-rag-backend/
│
├── main.py              # FastAPI application & RAG pipeline logic
├── app.py               # Streamlit frontend dashboard chat interface
├── requirements.txt     # Pinned Python dependencies
├── uploaded_docs/       # Local storage for ingested PDF files
└── vector_db/           # Persistent ChromaDB vector embedding store
```

---

## 🚀 Getting Started Locally

### Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) installed and running locally on your machine.

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/fastapi-rag-backend.git
cd fastapi-rag-backend

# Create and activate Python virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install fastapi uvicorn pydantic python-multipart pypdf langchain langchain-community langchain-ollama langchain-huggingface sentence-transformers "numpy<2.0.0" "chromadb<0.5.0" langchain-chroma streamlit requests
```

### 3. Pull Your Local LLM Model via Ollama
Ensure your local Ollama daemon is running, then pull your desired model:
```bash
ollama pull llama3
```

---

## 🏃 Running the Application

You will need two terminal tabs (both with your virtual environment activated):

### Tab 1: Start the FastAPI Backend
```bash
uvicorn main:app --reload
```
* *API Base URL:* `http://127.0.0.1:8000`
* *Interactive Swagger Docs:* `http://127.0.0.1:8000/docs`

### Tab 2: Start the Streamlit Frontend UI
```bash
streamlit run app.py
```
* *Web Dashboard:* `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | Health check and welcome message |
| **POST** | `/api/v1/upload-pdf` | Accepts multipart PDF upload, chunks text, generates embeddings, and persists to ChromaDB |
| **POST** | `/api/v1/query` | Retrieves top-k matching document chunks via vector similarity and queries local Ollama LLM for grounded answers |

---

## 💡 Future Enhancements
* [ ] Add automated document list endpoints (`/api/v1/documents`)
* [ ] Implement strict exception handling for password-protected or corrupted PDFs
* [ ] Containerize application services using Docker and Docker Compose