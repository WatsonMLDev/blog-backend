import threading
import schedule
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from src.ingestion import GitRepositoryIngester
from src.pipeline import PortfolioRagPipeline
from src.session_manager import SessionManager
from src.stats_tracker import StatsTracker
# Load environment variables from .env
from dotenv import load_dotenv
import os
import secrets
import google.genai.errors

load_dotenv()

# Configure logging to display pipeline messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

rag_pipeline = PortfolioRagPipeline()
ingester = GitRepositoryIngester()
session_manager = SessionManager(session_ttl_minutes=60)  # 1 hour TTL
stats_tracker = StatsTracker()  # Track chat statistics


API_KEY = os.environ.get("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Dependency function to validate the API key from the request header."""
    if not API_KEY:
        logger.critical("API_KEY environment variable not set! Server is insecure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured for authentication."
        )
    
    if secrets.compare_digest(api_key_header, API_KEY):
        return True
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )


# --- Data Models ---
class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000, description="User question, max 2000 chars")
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    documents: list

# --- Ingestion Logic ---
ingestion_lock = threading.Lock()

def _run_ingestion_internal():
    """Internal function to run ingestion. Assumes lock is held if necessary."""
    try:
        result = ingester.run()
        logger.info(f"Ingestion result: {result}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

def run_ingestion():
    """Scheduled task wrapper with locking."""
    if not ingestion_lock.acquire(blocking=False):
        logger.warning("Scheduled ingestion skipped: Ingestion already in progress.")
        return

    try:
        _run_ingestion_internal()
    finally:
        ingestion_lock.release()

def cleanup_sessions():
    """Clean up expired chat sessions."""
    try:
        count = session_manager.cleanup_expired_sessions()
        if count > 0:
            logger.info(f"Cleaned up {count} expired session(s)")
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

def schedule_tasks():
    """Schedule background tasks."""
    schedule.every().day.at("03:00").do(run_ingestion)
    schedule.every(15).minutes.do(cleanup_sessions)  # Clean up sessions every 15 minutes
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=schedule_tasks, daemon=True).start()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to handle pre-flight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - configure this for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, api_key: bool = Depends(get_api_key)):
    question = request.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question is required."})
    
    # Get or create session
    session_id = request.session_id
    if session_id:
        # Validate existing session
        session = session_manager.get_session(session_id)
        if not session:
            # Session expired or invalid, create new one
            session_id = session_manager.create_session()
            stats_tracker.log_event("session_created", session_id)
            logger.info(f"Session {request.session_id} not found, created new session: {session_id}")
    else:
        # Create new session
        session_id = session_manager.create_session()
        stats_tracker.log_event("session_created", session_id)
    
    # Get existing chat history (before adding current message)
    chat_history = session_manager.get_history(session_id) or []
    
    # Add user question to history
    session_manager.add_message(session_id, "user", question)
    stats_tracker.log_event("message_sent", session_id, {"role": "user"})
    
    try:
        # Pass chat history to pipeline for context-aware query expansion
        result = await rag_pipeline.run(question=question, chat_history=chat_history)
        answer = result.get("answer", "")
        documents = [doc.meta for doc in result.get("documents", [])]
        
        # Add assistant response to history
        session_manager.add_message(session_id, "assistant", answer)
        
        # Log inference stats
        stats_tracker.log_event("inference_completed", session_id, {
            "latency": result.get("latency"),
            "intent": result.get("intent"),
            "model": result.get("model"),
            "doc_count": len(documents)
        })

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            documents=documents
        )
    except Exception as e:
        # Check if it's a wrapped Google GenAI error (usually wrapped in RuntimeError)
        if isinstance(e, RuntimeError) and isinstance(e.__cause__, google.genai.errors.APIError):
            genai_error = e.__cause__
            status_code = genai_error.code

            # Simplified error mapping
            error_mapping = {
                429: "Rate limit exceeded. Please try again later.",
                400: "Bad request. Please check your input.",
                403: "Permission denied.",
                404: "Resource not found.",
                500: "Internal server error from AI provider."
            }

            error_message = error_mapping.get(status_code, "An error occurred with the AI provider.")

            # Concise logging for expected API errors (avoids dumping full JSON)
            logger.error(f"Google GenAI API Error: Code={status_code}, Status={genai_error.status}, Message={error_message}, Endpoint=/chat")

            return JSONResponse(status_code=status_code, content={"error": error_message})

        # Standard logging for other unexpected errors
        logger.error(f"Chat endpoint error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate answer."})

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, api_key: bool = Depends(get_api_key)):
    """Retrieve chat history for a given session."""
    history = session_manager.get_history(session_id)
    
    if history is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired"
        )
    
    return {"session_id": session_id, "history": history}

@app.get("/stats")
async def get_stats(days: int = None, api_key: bool = Depends(get_api_key)):
    """
    Get chat statistics.
    
    Args:
        days: Optional - filter to only include events from the last N days
    
    Returns:
        Aggregated statistics including:
        - Total sessions and messages
        - Unique session count
        - Activity by day
        - Recent activity (24h, 7d, 30d)
    """
    stats = stats_tracker.get_stats(days=days)
    
    # Also include current in-memory session info
    current_sessions = session_manager.get_stats()
    stats["current_active_sessions"] = current_sessions
    
    return stats

@app.post("/run-ingestion")
def run_ingestion_endpoint(api_key: bool = Depends(get_api_key)):
    # Sentinel Fix: Prevent concurrent ingestion (DoS protection)
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion is already in progress."
        )

    def protected_ingestion():
        try:
            _run_ingestion_internal()
        finally:
            ingestion_lock.release()

    threading.Thread(target=protected_ingestion, daemon=True).start()
    return {"status": "Ingestion started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
