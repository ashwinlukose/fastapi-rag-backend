import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# LangChain imports for document processing, vector store, and local LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

app = FastAPI(
    title="Local RAG Backend API",
    description="A modern FastAPI service for document-grounded AI question answering using ChromaDB and Ollama",
    version="1.0.0"
)

# Configuration directories
UPLOAD_DIR = "./uploaded_docs"
DB_DIR = "./vector_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize local embedding model & LLM via Ollama (running locally on your Mac)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
local_llm = ChatOllama(model="llama3", temperature=0.2)

# Initialize Chroma vector store
vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: list[str]

@app.get("/")
def read_root():
    return {"message": "FastAPI Local RAG Service is running! Go to /docs to test endpoints."}

@app.post("/api/v1/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Accepts a PDF file upload.
    2. Saves it locally.
    3. Splits the text into chunks and stores embeddings in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Load and split PDF text
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Add chunks to ChromaDB vector store
    vector_store.add_documents(chunks)
    
    return {
        "filename": file.filename,
        "status": "Success",
        "chunks_indexed": len(chunks),
        "message": "PDF uploaded, chunked, and embedded into ChromaDB successfully!"
    }

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag_system(request: QueryRequest):
    """
    1. Searches ChromaDB for relevant text chunks.
    2. Passes context to the local Ollama LLM.
    3. Returns a grounded answer.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Retrieve relevant documents from ChromaDB (top 3 matches)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(request.question)
    
    if not relevant_docs:
        return QueryResponse(
            question=request.question,
            answer="I couldn't find any relevant context in the uploaded documents to answer this.",
            source_documents=[]
        )
    
    # Extract context text
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    source_sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in relevant_docs]))
    
    # Construct prompt for the local LLM
    prompt = f"""You are a helpful assistant. Answer the question strictly using the provided context below. If you don't know the answer, state that you don't know.

Context:
{context}

Question: {request.question}
Answer:"""

    # Generate response from local Ollama model
    response = local_llm.invoke(prompt)
    
    return QueryResponse(
        question=request.question,
        answer=response.content,
        source_documents=source_sources
    )