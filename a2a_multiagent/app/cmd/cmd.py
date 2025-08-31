# =============================================================================
# Purpose:
# This file is a FastAPI backend that exposes the ImageGeneratorAgent
# functionality as HTTP endpoints with image viewing capabilities.
# FIXED: Properly serves images using file paths instead of artifacts
# =============================================================================
import asyncio
import argparse
from uuid import uuid4
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import base64
import json
import os
import logging

# Import the A2AClient from your client module
from client.client import A2AClient
# Import the Task model
from models.task import Task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
agent_url: str = "http://localhost:10004"  # Puerto del agente de imágenes
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Image Cache Integration (debe ser la misma que usa el agente)
# =============================================================================
class ImageData:
    def __init__(self, id=None, name=None, mime_type=None, file_path=None, error=None):
        self.id = id
        self.name = name
        self.mime_type = mime_type
        self.file_path = file_path
        self.error = error

class InMemoryImageCache:
    def __init__(self):
        self._cache = {}
    
    def get(self, session_id: str):
        return self._cache.get(session_id)
    
    def add_image(self, session_id: str, image_data: ImageData):
        if session_id not in self._cache:
            self._cache[session_id] = {}
        self._cache[session_id][image_data.id] = image_data

# Global cache instance (debe ser compartida con el agente)
image_cache = InMemoryImageCache()

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
    print(f"FastAPI server started, connected to agent at: {agent_url}")
    print(f"Image storage directory: {IMAGE_STORAGE_DIR}")
    yield
    # Shutdown: Clean up if needed
    print("FastAPI server shutting down...")

# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="A2A Image Generator API",
    description="REST API interface for ImageGeneratorAgent with CrewAI and Gemini",
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
# Helper Functions
# =============================================================================
def find_image_file(session_id: str, image_id: str) -> Optional[str]:
    """
    Find image file by session_id and image_id.
    First tries the cache, then searches the filesystem.
    """
    # Try cache first
    session_data = image_cache.get(session_id)
    if session_data and image_id in session_data:
        image_data = session_data[image_id]
        if image_data.file_path and os.path.exists(image_data.file_path):
            return image_data.file_path
    
    # Search filesystem if not in cache
    for ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
        possible_paths = [
            os.path.join(IMAGE_STORAGE_DIR, f"{image_id}.{ext}"),
            os.path.join(IMAGE_STORAGE_DIR, session_id, f"{image_id}.{ext}"),
            os.path.join(IMAGE_STORAGE_DIR, f"image_{image_id}.{ext}"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found image file: {path}")
                return path
    
    return None

def get_mime_type_from_file(file_path: str) -> str:
    """Get MIME type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return mime_types.get(ext, 'image/png')

# =============================================================================
# Main API Endpoints
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
    """
    if not client:
        raise HTTPException(status_code=500, detail="A2A client not initialized")

    session_id = request.session_id or uuid4().hex

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
        task: Task = await client.send_task(payload)
        agent_response = "No response received."

        if task.history and len(task.history) > 1:
            reply = task.history[-1]
            agent_response = reply.parts[0].text

        response = MessageResponse(
            task_id=task.id,
            session_id=session_id,
            agent_response=agent_response
        )

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
        raise HTTPException(
            status_code=500,
            detail=f"Error while sending task to agent: {str(e)}"
        )

@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    if not client:
        raise HTTPException(status_code=500, detail="A2A client not initialized")

    try:
        payload = {
            "id": uuid4().hex,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": ""}]
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
    return {"session_id": uuid4().hex}

# =============================================================================
# Image Management Endpoints (CORREGIDO PARA USAR ARCHIVOS DIRECTOS)
# =============================================================================
@app.get("/image/{session_id}/{image_id}")
async def serve_image_direct(session_id: str, image_id: str):
    """
    Serve image file directly from disk by session_id and image_id.
    This matches the URL format returned by the ImageGenerationAgent.
    """
    try:
        # Find the image file
        image_path = find_image_file(session_id, image_id)
        
        if not image_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Image {image_id} not found for session {session_id}"
            )
        
        # Get MIME type
        mime_type = get_mime_type_from_file(image_path)
        
        # Return the file directly
        return FileResponse(
            image_path,
            media_type=mime_type,
            filename=f"image_{image_id}{os.path.splitext(image_path)[1]}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error serving image: {str(e)}"
        )
    
@app.get("/image/{image_id}")
async def get_image(image_id: str):
    # Construct the full path to the image file
    image_path = "/media/dracero/08c67654-6ed7-4725-b74e-50f29ea60cb2/pythonAI-Others/a2a_protocol/a2a_multiagent/agents/image_generation/{image_id}.png".format(image_id=image_id)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)   

@app.get("/images/{session_id}/{image_id}")
async def get_image_raw(session_id: str, image_id: str):
    """
    Alternative endpoint for backward compatibility.
    Redirects to the main image serving endpoint.
    """
    return await serve_image_direct(session_id, image_id)

@app.get("/view-image/{session_id}/{image_id}", response_class=HTMLResponse)
async def view_image(session_id: str, image_id: str):
    """
    Display image in a beautiful HTML page.
    """
    try:
        # Check if image exists
        image_path = find_image_file(session_id, image_id)
        
        if not image_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Image {image_id} not found"
            )
        
        # Get image info
        mime_type = get_mime_type_from_file(image_path)
        file_size = os.path.getsize(image_path)
        
        # Create the image URL for embedding
        image_url = f"/image/{session_id}/{image_id}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Imagen {image_id}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    backdrop-filter: blur(10px);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #f0f0f0;
                }}
                .header h1 {{
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .image-container {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 12px;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 12px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    transition: transform 0.3s ease;
                }}
                .image-container img:hover {{
                    transform: scale(1.02);
                }}
                .info {{
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                    border-left: 5px solid #667eea;
                    padding: 20px;
                    margin: 25px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .info-item {{
                    background: rgba(255,255,255,0.7);
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 25px;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    margin: 10px 8px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                }}
                .btn-secondary {{
                    background: linear-gradient(45deg, #28a745, #20c997);
                }}
                .btn-tertiary {{
                    background: linear-gradient(45deg, #ffc107, #fd7e14);
                }}
                .actions {{
                    text-align: center;
                    margin-top: 30px;
                }}
                .download-link {{
                    background: linear-gradient(45deg, #dc3545, #e83e8c);
                }}
                @media (max-width: 768px) {{
                    body {{
                        padding: 10px;
                    }}
                    .container {{
                        padding: 20px;
                    }}
                    .btn {{
                        display: block;
                        margin: 10px 0;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Imagen Generada</h1>
                </div>
                <div class="info">
                    <h3>Información de la Imagen</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>ID</strong><br>
                            <code>{image_id}</code>
                        </div>
                        <div class="info-item">
                            <strong>Sesión</strong><br>
                            <code>{session_id}</code>
                        </div>
                        <div class="info-item">
                            <strong>Tipo</strong><br>
                            <code>{mime_type}</code>
                        </div>
                        <div class="info-item">
                            <strong>Tamaño</strong><br>
                            <code>{file_size:,} bytes</code>
                        </div>
                        <div class="info-item">
                            <strong>Estado</strong><br>
                            <span style="color: #28a745;">Disponible</span>
                        </div>
                    </div>
                </div>
                <div class="image-container">
                    <img src="{image_url}" alt="Imagen generada {image_id}" loading="lazy" 
                         onerror="this.style.display='none'; document.getElementById('error-msg').style.display='block';">
                    <div id="error-msg" style="display:none; color: #dc3545; padding: 20px;">
                        Error cargando la imagen
                    </div>
                </div>
                <div class="actions">
                    <a href="{image_url}" class="btn download-link" download="image_{image_id}.png">
                        Descargar Imagen
                    </a>
                    <a href="/" class="btn">
                        Inicio
                    </a>
                    <a href="javascript:history.back()" class="btn btn-secondary">
                        Volver
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error displaying image {image_id}: {e}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Imagen {image_id}</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .error-container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .error {{
                    color: #dc3545;
                    font-size: 1.2em;
                    margin: 20px 0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin: 10px;
                }}
                .btn:hover {{
                    background: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Error</h1>
                <div class="error">
                    Error obteniendo imagen {image_id}: {str(e)}
                </div>
                <div>
                    <a href="/" class="btn">Inicio</a>
                    <a href="javascript:history.back()" class="btn">Volver</a>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)

# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A2A Image Generator API</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
            }
            .header h1 {
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0 0 20px 0;
                font-size: 3em;
            }
            .description {
                font-size: 1.2em;
                color: #666;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .feature-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
            }
            .feature-icon {
                font-size: 2.5em;
                margin-bottom: 15px;
            }
            .btn {
                display: inline-block;
                padding: 15px 30px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                margin: 10px;
                font-weight: 600;
                font-size: 1.1em;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .btn-secondary {
                background: linear-gradient(45deg, #28a745, #20c997);
            }
            .api-endpoints {
                text-align: left;
                background: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                margin: 30px 0;
            }
            .endpoint {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
            }
            .method {
                display: inline-block;
                padding: 4px 8px;
                background: #007bff;
                color: white;
                border-radius: 4px;
                font-size: 0.9em;
                font-weight: bold;
                margin-right: 10px;
            }
            .method.post { background: #28a745; }
            .method.get { background: #17a2b8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>A2A Image Generator</h1>
            </div>
            <div class="description">
                API REST para generar y gestionar imágenes usando CrewAI y Gemini.<br>
                Interfaz web para interactuar con el agente de generación de imágenes.
            </div>
            <div class="features">
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <h3>Chat con IA</h3>
                    <p>Envía mensajes al agente y recibe respuestas inteligentes</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🖼️</div>
                    <h3>Gestión de Imágenes</h3>
                    <p>Visualiza, descarga y administra imágenes generadas</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3>Interfaz Web</h3>
                    <p>Interfaz moderna y responsive para todos los dispositivos</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔗</div>
                    <h3>API REST</h3>
                    <p>Endpoints completos para integración con otras aplicaciones</p>
                </div>
            </div>
            <div class="api-endpoints">
                <h3>Endpoints Principales</h3>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <strong>/send-message</strong> - Enviar mensaje al agente
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/image/{session_id}/{image_id}</strong> - Servir imagen directa
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/view-image/{session_id}/{image_id}</strong> - Ver imagen con interfaz
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/session/{session_id}/history</strong> - Historial de conversación
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/health</strong> - Estado del sistema
                </div>
            </div>
            <div>
                <a href="/docs" class="btn" target="_blank">
                    Documentación API
                </a>
                <a href="/session/new" class="btn btn-secondary" onclick="handleNewSession()">
                    Nueva Sesión
                </a>
            </div>
        </div>
        <script>
            async function handleNewSession() {
                try {
                    const response = await fetch('/session/new', { method: 'POST' });
                    const data = await response.json();
                    alert(`Nueva sesión creada: ${data.session_id}`);
                } catch (error) {
                    console.error('Error creating session:', error);
                    alert('Error creando nueva sesión');
                }
                return false;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# =============================================================================
# Main Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FastAPI A2A Image Generator Server")
    parser.add_argument(
        "--agent-url", 
        default="http://localhost:10003",
        help="URL del agente A2A (default: http://localhost:10003)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    global agent_url
    agent_url = args.agent_url
    
    # Ensure image storage directory exists
    os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
    
    print(f"Starting FastAPI A2A Image Generator Server...")
    print(f"Agent URL: {agent_url}")
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"Image Storage: {IMAGE_STORAGE_DIR}")
    print(f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()