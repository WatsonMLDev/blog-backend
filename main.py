import threading
import schedule
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.ingestion import GitRepositoryIngester
from src.pipeline import PortfolioRagPipeline
from dataclasses import dataclass
# Load environment variables from .env
from dotenv import load_dotenv
import os
import secrets

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
    question: str

# --- Ingestion Logic ---
def run_ingestion():
    try:
        result = ingester.run()
        logger.info(f"Ingestion result: {result}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

def schedule_ingestion():
    schedule.every().day.at("03:00").do(run_ingestion)
    while True:
        schedule.run_pending()
        time.sleep(60)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=schedule_ingestion, daemon=True).start()
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
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, api_key: bool = Depends(get_api_key)):
    question = request.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question is required."})
    try:
        result = await rag_pipeline.run(question=question)
        return {"answer": result.get("answer", ""), "documents": [doc.meta for doc in result.get("documents", [])]}
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate answer."})

@app.post("/run-ingestion")
def run_ingestion_endpoint(api_key: bool = Depends(get_api_key)):
    threading.Thread(target=run_ingestion, daemon=True).start()
    return {"status": "Ingestion started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
