from docx import Document
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

from RAGHelper_local import RAGHelperLocal
from pymilvus import Collection, connections
import random
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from uuid import UUID
chat_memory = {}
app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Initialize RAG helper

raghelper = RAGHelperLocal(logger)


@app.post("/create_title")
async def create_title(request: Request):
    data = await request.json()
    question = data.get("question")
    llm = raghelper.get_llm()
    response = llm.invoke([{"role": "user", "content": f"Write a succinct title for this: {question}"}])
    return {"title": response.content if hasattr(response, "content") else response}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    session_id = data.get("session_id")
    docs = data.get("docs", [])

    if not session_id:
        return JSONResponse(content={"error": "Missing session_id"}, status_code=400)

    end_string = os.getenv("llm_assistant_token")
    history = chat_memory.get(session_id, [])

    # Manually append the new user message BEFORE sending to RAG
    history.append({"role": "user", "content": prompt})

    # Get answer, passing in full history (including this user turn)
    thread, response = raghelper.handle_user_interaction(prompt, history.copy())

    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        text = response.get("text", "")
    else:
        raise ValueError(f"Unexpected response type from LLM: {type(response)}")

    try:
        index = text.rindex(end_string)
        reply = text[index + len(end_string):]
    except ValueError:
        print("[WARN] Assistant token not found in response. Returning full response.")
        reply = text

    # Append assistant reply AFTER LLM call
    history.append({"role": "assistant", "content": reply})
    chat_memory[session_id] = history  # Update full session memory

    if not docs or 'docs' in response:
        docs = response['docs']

    return {
        "reply": reply,
        "history": history,
        "documents": [
            {
                "source": doc.metadata.get("source"),
                "pk": doc.metadata.get("pk"),
                "provenance": float(doc.metadata["provenance"]) if doc.metadata.get("provenance") is not None else None,
                "sharepoint_url": doc.metadata.get("sharepoint_url")
            }
            for doc in docs if hasattr(doc, "metadata")
        ],
        "question": prompt,
    }


@app.get("/get_documents")
async def get_documents():
    data_dir = os.getenv("data_directory")
    file_types = os.getenv("file_types", "").split(",")
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]
    return files


@app.post("/get_document")
async def get_document(request: Request):
    data = await request.json()
    filename = data.get("filename")
    file_path = os.path.join(os.getenv("data_directory"), filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)

@app.post("/load_sharepoint_documents")
async def load_sharepoint_documents():
    sharepoint_dir = os.getenv("sharepoint_data_directory")
    if not sharepoint_dir or not os.path.exists(sharepoint_dir):
        return JSONResponse(content={"error": "SharePoint folder not found."}, status_code=404)

    count = 0
    for file in os.listdir(sharepoint_dir):
        if file.endswith(".docx"):
            filepath = os.path.join(sharepoint_dir, file)
            raghelper.add_document(filepath)
            count += 1

    return {"loaded_documents": count}

@app.get("/load_sharepoint_docs")
async def load_sharepoint_docs():
    sharepoint_path = os.getenv("sharepoint_data_directory")
    if not sharepoint_path:
        return {"error": "sharepoint_data_directory not set in .env"}

    loaded_files = []
    for root, dirs, files in os.walk(sharepoint_path):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt', '.docx')):  # Add more types if needed
                full_path = os.path.join(root, file)
                try:
                    raghelper.add_document(full_path)
                    loaded_files.append(full_path)
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

    return {"loaded": loaded_files, "count": len(loaded_files)}




# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5001, reload=True)
