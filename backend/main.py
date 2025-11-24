# Load environment variables from .env file FIRST, before any imports
import os
import pickle
import tempfile
import uuid
import shutil
from rag_indexer import extract_text_from_pdf, chunk_text, embed_texts, build_faiss_index, retrieve

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, manually load .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import List, Optional, Dict, Any
from models.openai_chat import get_openai_streaming_response, format_messages, query_openai_api
from kernel_manager import get_kernel_manager

app = FastAPI(title="Alzheimer's Analysis Pipeline API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session-based storage for PDFs and FAISS indexes
# Structure: {session_id: {pdf_path, pdf_name, chunks, embeddings, faiss_index}}
session_storage: Dict[str, Dict[str, Any]] = {}

# Temporary directory for PDF uploads (cleared on restart)
TEMP_DIR = tempfile.mkdtemp(prefix="gene_ide_pdfs_")

# PDF summary response format
PDF_SUMMARY_FORMAT = """
Format the response using markdown with the following sections:

## Summary Structure

### 1. Prevalence of the Issue
Describe the scope and importance of the problem being addressed.

### 2. Impact & Innovation
Explain the paper's contribution and innovative aspects.

### 3. Existing Landscape
Summarize the current state of research in this area.

### 4. Hypothesis
State the paper's main hypothesis or research question.

### 5. Datasets
List the datasets used (specify if open, closed, or proprietary).

### 6. Methods & Approach
Describe the methodology (e.g., deep learning, machine learning, statistics).
Use **bold** for key techniques and `code` for specific algorithms or frameworks.

### 7. Results
Summarize the main findings and their significance.
Use tables or bullet points for quantitative results when appropriate.

"""

def cleanup_session(session_id: str):
    """Clean up session data including PDF file and FAISS index"""
    if session_id in session_storage:
        session_data = session_storage[session_id]
        # Remove PDF file if it exists
        if 'pdf_path' in session_data and os.path.exists(session_data['pdf_path']):
            try:
                os.remove(session_data['pdf_path'])
            except Exception as e:
                print(f"Error removing PDF file: {e}")
        # Remove from session storage
        del session_storage[session_id]
        print(f"Cleaned up session: {session_id}")

def is_document_summary_query(query: str) -> bool:
    """
    Detect if a query is a document summary query.
    Returns True if the query is a document summary query.
    """
    query_lower = query.lower()
    instruction = "You are an assistant who is going to tell me if the query is a document summary query. If it is, return True, otherwise return False."
    response = query_openai_api(instruction, query_lower)
    return "true" in response.lower()

def is_document_query(query: str) -> bool:
    """
    Detect if a query is document-related using keyword/heuristic approach.
    Returns True if the query seems to be asking about document content.
    """
    query_lower = query.lower()
    
    # Document-related keywords
    doc_keywords = [
        'paper', 'document', 'pdf', 'article', 'study', 'research',
        'author', 'abstract', 'conclusion', 'method', 'result', 'finding',
        'section', 'chapter', 'page', 'figure', 'table', 'reference',
        'cite', 'source', 'publication'
    ]
    
    # Interrogative patterns
    interrogatives = [
        'what', 'why', 'how', 'when', 'where', 'who', 'which',
        'explain', 'describe', 'summarize', 'summary', 'tell me about',
        'according to', 'does the', 'is the', 'are the', 'can you explain'
    ]
    
    # Check for document keywords
    has_doc_keyword = any(keyword in query_lower for keyword in doc_keywords)
    
    # Check for interrogative patterns
    has_interrogative = any(pattern in query_lower for pattern in interrogatives)
    
    # Check for question marks
    has_question_mark = '?' in query
    
    # Return True if it has interrogative patterns OR (question mark AND doc keywords)
    return has_interrogative or (has_question_mark and has_doc_keyword)

# Request/Response models for API endpoints
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ExecuteRequest(BaseModel):
    code: str
    cell_id: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "Gene Analysis IDE API"}

@app.post("/api/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload a PDF file and create FAISS index for RAG.
    Replaces any existing PDF for this session.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Clean up any existing session data
        if session_id in session_storage:
            cleanup_session(session_id)
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        pdf_path = os.path.join(TEMP_DIR, unique_filename)
        
        # Save uploaded file
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"PDF saved to: {pdf_path}")
        
        # Process PDF with RAG indexer
        try:
            # Extract text from PDF
            texts = extract_text_from_pdf(pdf_path)
            if not texts:
                raise ValueError("Could not extract text from PDF")
            
            # Chunk text
            chunks = chunk_text(texts)
            if not chunks:
                raise ValueError("Could not create chunks from PDF text")
            
            # Generate embeddings
            embeddings = embed_texts(chunks)
            
            # Build FAISS index
            faiss_index = build_faiss_index(embeddings)
            
            # Store in session
            session_storage[session_id] = {
                'pdf_path': pdf_path,
                'pdf_name': file.filename,
                'chunks': chunks,
                'embeddings': embeddings,
                'faiss_index': faiss_index,
                'num_pages': len(texts),
                'num_chunks': len(chunks)
            }
            
            print(f"Successfully indexed PDF for session {session_id}: {file.filename}")
            
            return {
                "status": "success",
                "message": "PDF uploaded and indexed successfully",
                "pdf_name": file.filename,
                "num_pages": len(texts),
                "num_chunks": len(chunks)
            }
            
        except Exception as e:
            # Clean up file if processing failed
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with RAG support"""
    await websocket.accept()
    
    # Extract session_id from query params
    session_id = websocket.query_params.get("session_id", None)
    print(f"WebSocket connected with session_id: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming message
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                chat_history = message_data.get("history", [])
                
                if not user_message.strip():
                    await websocket.send_text("<<<ERROR>>>")
                    continue
                
                # Check if session has indexed PDF and if query is document-related
                use_rag = False
                rag_context = ""
                
                if session_id and session_id in session_storage:
                    session_data = session_storage[session_id]
                    if is_document_query(user_message):
                        use_rag = True
                        print(f"Document query detected, using RAG for: {user_message}")
                        
                        # Retrieve relevant chunks from FAISS
                        try:
                            results = retrieve(
                                session_data['faiss_index'],
                                session_data['embeddings'],
                                session_data['chunks'],
                                user_message,
                                k=5
                            )
                            
                            # Format context from retrieved chunks
                            context_parts = []
                            for i, result in enumerate(results, 1):
                                page = result['meta']['page']
                                text = result['text']
                                score = result['score']
                                context_parts.append(
                                    f"[Context {i} - Page {page}, Relevance: {score:.2f}]\n{text}\n"
                                )
                            
                            rag_context = "\n".join(context_parts)
                            print(f"Retrieved {len(results)} relevant chunks from PDF")
                            
                        except Exception as e:
                            print(f"Error retrieving RAG context: {e}")
                            use_rag = False
                
                # Format messages for OpenAI with optional RAG context
                if use_rag and rag_context:
                    # Augment user message with RAG context
                    augmented_message = f"""Based on the following relevant excerpts from the uploaded PDF document, please answer the user's question.

Relevant Document Context:
{rag_context}

User Question: {user_message}

"""
                    if is_document_summary_query(user_message):
                        augmented_message += PDF_SUMMARY_FORMAT
                    augmented_message += """Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate that."""
                    messages = format_messages(augmented_message, chat_history)
                else:
                    messages = format_messages(user_message, chat_history)
                
                # Get streaming response from OpenAI
                print(f"Processing message: {user_message}")
                response_generator = get_openai_streaming_response(messages)
                
                # Stream each chunk to the client with proper async handling
                async for chunk in response_generator:
                    if chunk is not None and chunk != '':
                        print(f"Sending chunk: '{chunk}'")
                        await websocket.send_text(chunk)
                        # Small delay to ensure chunks are sent separately
                        await asyncio.sleep(0.02)
                
                # Send end marker
                await websocket.send_text("<<<END>>>")
                
            except json.JSONDecodeError:
                await websocket.send_text("<<<ERROR>>>")
            except Exception as e:
                print(f"Chat error: {e}")
                await websocket.send_text("<<<ERROR>>>")
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")
        # Clean up session on disconnect
        if session_id:
            cleanup_session(session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if session_id:
            cleanup_session(session_id)

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """HTTP endpoint for streaming chat (alternative to WebSocket)"""
    try:
        messages = format_messages(request.message, [msg.dict() for msg in request.history])
        
        # Collect all chunks for HTTP response
        response_chunks = []
        async for chunk in get_openai_streaming_response(messages):
            if chunk:
                response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        return {"response": full_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    """Execute Python code using Jupyter kernel"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Received execute request for cell {request.cell_id}")
        km = get_kernel_manager()
        result = km.execute_code(request.code, request.cell_id)
        logger.info(f"Execution completed, sending response back")
        return result
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return {
            "outputs": [{
                "type": "error",
                "ename": "KernelError",
                "evalue": str(e),
                "traceback": [str(e)]
            }],
            "status": "error"
        }


## Streaming endpoint removed per user request

@app.post("/api/restart_kernel")
async def restart_kernel():
    """Restart the Jupyter kernel"""
    try:
        km = get_kernel_manager()
        result = km.restart_kernel()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/kernel_status")
async def kernel_status():
    """Get kernel status"""
    try:
        km = get_kernel_manager()
        return km.get_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Notebook helper endpoints
# -----------------------------

def _resolve_notebook_path(path: Optional[str]) -> str:
    """Resolve the absolute path to the notebook file.

    Defaults to the project's root-level 'colab.ipynb' if no path provided.
    This backend typically runs from the 'backend' directory, so we go one level up.
    """
    if path and path.strip():
        return os.path.abspath(path)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.abspath(os.path.join(backend_dir, "..", "colab.ipynb"))
    return default_path


@app.get("/api/notebook/cells")
async def list_notebook_cells(path: Optional[str] = None):
    """List code cells from a Jupyter notebook as step candidates."""
    nb_path = _resolve_notebook_path(path)
    if not os.path.exists(nb_path):
        raise HTTPException(status_code=404, detail=f"Notebook not found: {nb_path}")

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read notebook: {e}")

    cells = nb_data.get("cells", [])
    result = []
    step_index = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            first_line = next((line for line in source if str(line).strip() != ""), "")
        else:
            # string
            lines = str(source).splitlines()
            first_line = next((line for line in lines if line.strip() != ""), "")

        title = first_line.strip()
        # Strip leading comment markers for a cleaner title
        if title.startswith("#"):
            title = title.lstrip("#").strip()

        step_index += 1
        result.append({
            "index": idx,               # actual notebook cell index
            "stepNumber": step_index,   # 1-based step sequence among code cells
            "title": title or f"Cell {idx}",
            "description": title or "Notebook code cell",
        })

    return {"notebook": nb_path, "steps": result}


@app.get("/api/notebook/cell/{index}")
async def get_notebook_cell(index: int, path: Optional[str] = None):
    """Fetch the full source of a specific notebook cell by its absolute cell index."""
    nb_path = _resolve_notebook_path(path)
    if not os.path.exists(nb_path):
        raise HTTPException(status_code=404, detail=f"Notebook not found: {nb_path}")

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read notebook: {e}")

    cells = nb_data.get("cells", [])
    if index < 0 or index >= len(cells):
        raise HTTPException(status_code=404, detail=f"Cell index out of range: {index}")

    cell = cells[index]
    if cell.get("cell_type") != "code":
        raise HTTPException(status_code=400, detail=f"Cell {index} is not a code cell")

    source = cell.get("source", [])
    if isinstance(source, list):
        code = "".join(source)
    else:
        code = str(source)

    return {"index": index, "source": code}

if __name__ == "__main__":
    import uvicorn
    # Increase timeout for long-running code execution in low-memory environments
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep-alive timeout
        workers=1  # Use single worker to save memory
    )