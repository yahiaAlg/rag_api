import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
import chromadb
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from datetime import datetime
import uuid
from fastapi import Request, Depends
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG QA API",
    description="API for RAG-based Question Answering using Ollama and ChromaDB",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Load environment variables
COLLECTION_NAME = "documents"
PERSIST_DIRECTORY = "./chroma_db"
MODEL_NAME = "eas/dragon-mistral-v0"

# Initialize ChromaDB
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Initialize embeddings (changed to CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize Ollama
# Update the Ollama initialization in app.py
llm = Ollama(model=MODEL_NAME, base_url="http://host.docker.internal:11435")


class QuestionRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None


class DocumentResponse(BaseModel):
    collection_id: str
    message: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


def process_pdf(file: bytes) -> str:
    """Extract text from PDF file"""
    from io import BytesIO

    # Create a file-like object from bytes
    pdf_file = BytesIO(file)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_text(file: bytes) -> str:
    """Process text file"""
    return file.decode("utf-8")


# Update the rate limiter decorators to include request dependency:
@app.post("/api/documents", response_model=DocumentResponse)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    try:
        collection_id = str(uuid.uuid4())
        content = await file.read()

        if file.filename.endswith(".pdf"):
            text = process_pdf(content)
        elif file.filename.endswith(".txt"):
            text = process_text(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=f"{COLLECTION_NAME}_{collection_id}",
            persist_directory=PERSIST_DIRECTORY,
        )

        return DocumentResponse(
            collection_id=collection_id, message="Document processed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=AnswerResponse)
@limiter.limit("30/minute")
async def ask_question(request: Request, question_request: QuestionRequest):
    try:
        if not question_request.collection_id:
            raise HTTPException(status_code=400, detail="Collection ID is required")

        vectorstore = Chroma(
            collection_name=f"{COLLECTION_NAME}_{question_request.collection_id}",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None, qa_chain, question_request.question
        )

        return AnswerResponse(
            answer=result["result"],
            sources=[
                doc.page_content[:200] + "..." for doc in result["source_documents"]
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add these to your existing app.py
from datetime import datetime


@app.get("/")
async def root():
    return {
        "message": "RAG QA API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RAG QA API",
    }
