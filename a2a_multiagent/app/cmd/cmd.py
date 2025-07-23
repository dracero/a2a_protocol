# =============================================================================
# Purpose:
# This file is a FastAPI backend that exposes the TellDateTimeTimezoneAgent
# functionality as HTTP endpoints instead of a CLI interface.
#
# It provides REST API endpoints to:
# - Send messages to the agent
# - Retrieve conversation history
# - Manage sessions
#
# This version supports:
# - FastAPI web framework
# - Session management via HTTP
# - JSON request/response format
# - Optional task history retrieval
# =============================================================================

import asyncio
import argparse
from uuid import uuid4
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the A2AClient from your client module
from client.client import A2AClient

# Import the Task model
from models.task import Task


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class SendMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_history: bool = False

class MessageResponse(BaseModel):
    task_id: str
    session_id: str
    agent_response: str
    history: Optional[List[dict]] = None

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[dict]

class HealthResponse(BaseModel):
    status: str
    agent_url: str


# =============================================================================
# Global Variables
# =============================================================================

# Global client instance - will be initialized when the app starts
client: Optional[A2AClient] = None
agent_url: str = "http://localhost:10002"


# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle - initialize client on startup
    """
    global client
    
    # Startup: Initialize the A2A client
    client = A2AClient(url=agent_url)
    print(f"✅ FastAPI server started, connected to agent at: {agent_url}")
    
    yield
    
    # Shutdown: Clean up if needed
    print("🔄 FastAPI server shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="A2A Agent API",
    description="REST API interface for TellDateTimeTimezoneAgent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API and agent connection
    """
    return HealthResponse(
        status="healthy",
        agent_url=agent_url
    )


@app.post("/send-message", response_model=MessageResponse)
async def send_message(request: SendMessageRequest):
    """
    Send a message to the A2A agent and get a response
    
    Args:
        request: SendMessageRequest containing message, optional session_id, and history flag
        
    Returns:
        MessageResponse with agent reply and optional conversation history
    """
    if not client:
        raise HTTPException(status_code=500, detail="A2A client not initialized")
    
    # Generate session ID if not provided
    session_id = request.session_id or uuid4().hex
    
    # Construir el payload SOLO con los campos de params, para TaskSendParams
    payload = {
        "id": uuid4().hex,
        "sessionId": session_id,
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": request.message}]
        },
        "historyLength": None,
        "metadata": None
    }
    
    try:
        # Send the task to the agent and get a structured Task response
        task: Task = await client.send_task(payload)
        
        # Extract the agent's response
        agent_response = "No response received."
        if task.history and len(task.history) > 1:
            reply = task.history[-1]  # Last message is usually from the agent
            agent_response = reply.parts[0].text
        
        # Prepare response
        response = MessageResponse(
            task_id=task.id,
            session_id=session_id,
            agent_response=agent_response
        )
        
        # Include history if requested
        if request.include_history and task.history:
            response.history = [
                {
                    "role": msg.role,
                    "message": msg.parts[0].text,
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in task.history
            ]
        
        return response
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] send-message failed: {e}\nTraceback:\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Error while sending task to agent: {str(e)}"
        )


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """
    Retrieve the conversation history for a specific session
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        SessionHistoryResponse with the conversation history
    """
    if not client:
        raise HTTPException(status_code=500, detail="A2A client not initialized")
    
    try:
        # Send a dummy message to get the session history
        # This is a limitation of the current A2A design - we need to send a message to get history
        payload = {
            "id": uuid4().hex,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": ""}]  # Empty message to just get history
            }
        }
        
        task: Task = await client.send_task(payload)
        
        history = []
        if task.history:
            history = [
                {
                    "role": msg.role,
                    "message": msg.parts[0].text,
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in task.history
            ]
        
        return SessionHistoryResponse(
            session_id=session_id,
            history=history
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while retrieving session history: {str(e)}"
        )


@app.post("/session/new")
async def create_new_session():
    """
    Create a new session ID
    
    Returns:
        A new session ID that can be used for subsequent requests
    """
    return {"session_id": uuid4().hex}


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="FastAPI backend for A2A Agent")
    parser.add_argument(
        "--agent", 
        default="http://localhost:10002",
        help="Base URL of the A2A agent server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the FastAPI server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the FastAPI server to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for the FastAPI application
    """
    global agent_url
    
    args = parse_args()
    agent_url = args.agent
    
    print(f"🚀 Starting FastAPI server...")
    print(f"📡 Agent URL: {agent_url}")
    print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API documentation at: http://{args.host}:{args.port}/docs")
    
    # Run the FastAPI server
    uvicorn.run(
        "app.cmd.cmd:app",  # Module path to the FastAPI app
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()