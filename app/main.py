from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
from pathlib import Path

from app.schemas import SessionRequest, SessionResponse, QAPair
from app.agents import run_qa_session

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dual-LLM Q&A Agents API",
    description="API for generating Q&A pairs using two LLM agents with OpenAI",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory for JSON files
OUTPUT_DIR = Path("qa_sessions")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dual-LLM Q&A Agents API (OpenAI)",
        "endpoints": {
            "/run-session": "POST - Generate Q&A pairs"
        }
    }


@app.post("/run-session", response_model=SessionResponse)
async def run_session(request: SessionRequest):
    """
    Generate Q&A pairs using two LLM agents with OpenAI API
    
    Args:
        request: SessionRequest containing subject and num_pairs
        
    Returns:
        SessionResponse with generated Q&A pairs
    """
    logger.info(f"Received request for subject: {request.subject}, num_pairs: {request.num_pairs}")
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Please set it in .env file"
        )
    
    try:
        # Run the Q&A session
        pairs_data = run_qa_session(
            subject=request.subject,
            num_pairs=request.num_pairs,
            api_key=api_key
        )
        
        # Convert to Pydantic models
        pairs = [QAPair(**pair) for pair in pairs_data]
        
        # Create response
        response = SessionResponse(
            subject=request.subject,
            num_pairs=len(pairs),
            pairs=pairs
        )
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = "".join(c if c.isalnum() else "_" for c in request.subject)
        filename = f"qa_session_{safe_subject}_{timestamp}.json"
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Q&A session saved to {filepath}")
        
        return response
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)