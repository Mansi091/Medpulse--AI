from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os

from rag_pipeline import (
    build_rag_pipeline,
    build_hybrid_retriever_from_disk,
    ask_question,
    ask_question_stream,
    process_workspace_files,
    init_embeddings,
    init_llm,
    init_vision_llm
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")

    app.state.embeddings = init_embeddings()
    app.state.llm = init_llm()
    app.state.vision_llm = init_vision_llm()

    yield

    print("Server shutting down...")

    del app.state.embeddings
    del app.state.llm
    del app.state.vision_llm


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

retriever_store = {}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    question: str
    chat_history: str = ""


@app.post("/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...)
):
    saved_paths = []
    filenames = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}"
            )
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        saved_paths.append(file_path)
        filenames.append(file.filename)

    embeddings = request.app.state.embeddings
    vision_llm = request.app.state.vision_llm
    persist_dir = os.path.join(CHROMA_DB_DIR, "active_workspace")

    retriever = process_workspace_files(
        saved_paths,
        embeddings,
        vision_llm,
        persist_dir
    )

    retriever_store["active_workspace"] = retriever

    return {
        "message": "Workspace initialized successfully",
        "filenames": filenames
    }


@app.post("/ask")
async def ask_pdf(
    request: Request,
    chat_request: ChatRequest
):
    question = chat_request.question
    chat_history = chat_request.chat_history

    if "active_workspace" not in retriever_store:
        persist_dir = os.path.join(CHROMA_DB_DIR, "active_workspace")
        if os.path.exists(persist_dir):
            embeddings = request.app.state.embeddings
            retriever = build_hybrid_retriever_from_disk(
                embeddings,
                persist_dir
            )
            retriever_store["active_workspace"] = retriever
        else:
            raise HTTPException(
                status_code=404,
                detail="Workspace not initialized. Please upload files first."
            )

    retriever = retriever_store["active_workspace"]
    llm = request.app.state.llm

    return StreamingResponse(
        ask_question_stream(question, retriever, llm, chat_history),
        media_type="text/event-stream"
    )
