# =============================================================================
# Purpose:
# This file is a FastAPI backend that exposes the ImageGeneratorAgent
# functionality as HTTP endpoints with image viewing capabilities.
#
# It provides REST API endpoints to:
# - Send messages to the agent
# - Retrieve conversation history
# - Manage sessions
# - View and manage generated images
#
# This version supports:
# - FastAPI web framework
# - Session management via HTTP
# - JSON request/response format
# - Image viewing and management
# - HTML interfaces for image display
# =============================================================================
import asyncio
import argparse
from uuid import uuid4
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
import uvicorn
import base64
import json
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
class ImageListResponse(BaseModel):
    session_id: str
    total_images: int
    images: List[dict]
class ImageDataResponse(BaseModel):
    id: str
    name: str
    mime_type: str
    has_data: bool
    session_id: str
# =============================================================================
# Global Variables
# =============================================================================
# Global client instance - will be initialized when the app starts
client: Optional[A2AClient] = None
agent_url: str = "http://localhost:10003"  # Cambiado al puerto del agente de imágenes
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
# Image Management Endpoints
# =============================================================================
@app.get("/images/{session_id}/{image_id}")
async def get_image_raw(session_id: str, image_id: str):
    """
    Servir imagen por ID y session ID como archivo de imagen crudo
    """
    if not client:
        raise HTTPException(status_code=500, detail="A2A client not initialized")
    try:
        # Get image data from the agent
        payload = {
            "id": uuid4().hex,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": f"GET_IMAGE_DATA {image_id}"}]
            }
        }
        task: Task = await client.send_task(payload)
        if task.history and len(task.history) > 1:
            agent_response = task.history[-1].parts[0].text
            # The agent should return a JSON response with the image data
            if "JSON_DATA:" in agent_response:
                try:
                    # Extract the JSON part
                    json_str = agent_response.split("JSON_DATA:")[1].strip()
                    image_info = json.loads(json_str)
                    
                    # Check if the image data is available
                    if image_info.get("found") and image_info.get("bytes"):
                        # Decode base64
                        image_bytes = base64.b64decode(image_info["bytes"])
                        # Determine the mime type from the image info
                        mime_type = image_info.get("mime_type", "image/png")
                        return Response(
                            content=image_bytes,
                            media_type=mime_type,
                            headers={
                                "Content-Disposition": f"inline; filename={image_info.get('name', 'image.png')}"
                            }
                        )
                    else:
                        raise HTTPException(status_code=404, detail="Image data not found in response")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing image data: {str(e)}")
            else:
                raise HTTPException(status_code=404, detail="No JSON_DATA found in response")
        else:
            raise HTTPException(status_code=404, detail="No response from agent")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obtaining image: {str(e)}")

@app.get("/view-image/{session_id}/{image_id}", response_class=HTMLResponse)
async def view_image(session_id: str, image_id: str):
    """
    Mostrar imagen en el navegador con una interfaz HTML simple
    """
    try:
        # Get the image data from the server
        response = await get_image_raw(session_id, image_id)
        # Extract the image data from the response
        image_data = response.body
        # Determine the mime type from the response headers
        mime_type = response.media_type
        # Convert to base64 for HTML display
        base64_data = base64.b64encode(image_data).decode("utf-8")
        
        # Create HTML response
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
                    <h1>🎨 Imagen Generada</h1>
                </div>
                <div class="info">
                    <h3>📋 Información de la Imagen</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>🆔 ID</strong><br>
                            <code>{image_id}</code>
                        </div>
                        <div class="info-item">
                            <strong>🔗 Sesión</strong><br>
                            <code>{session_id}</code>
                        </div>
                        <div class="info-item">
                            <strong>🕒 Estado</strong><br>
                            <span style="color: #28a745;">✅ Disponible</span>
                        </div>
                    </div>
                </div>
                <div class="image-container">
                    <img src="data:{mime_type};base64,{base64_data}" alt="Imagen generada {image_id}" loading="lazy">
                </div>
                <div class="actions">
                    <a href="/images/{session_id}/{image_id}" class="btn download-link" download="image_{image_id}.png">
                        💾 Descargar Imagen
                    </a>
                    <a href="/list-images/{session_id}" class="btn btn-secondary">
                        📋 Ver Todas las Imágenes
                    </a>
                    <a href="/session/new" class="btn btn-tertiary" onclick="window.open(this.href); return false;">
                        🆕 Nueva Sesión
                    </a>
                    <a href="/" class="btn">
                        🏠 Inicio
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        # Fallback error page
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
                <h1>❌ Error</h1>
                <div class="error">
                    Error obteniendo imagen {image_id}: {str(e)}
                </div>
                <div>
                    <a href="/list-images/{session_id}" class="btn">📋 Ver Lista</a>
                    <a href="/" class="btn">🏠 Inicio</a>
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
    """
    Página de inicio con interfaz HTML simple
    """
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
                <h1>🎨 A2A Image Generator</h1>
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
                <h3>🚀 Endpoints Principales</h3>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <strong>/send-message</strong> - Enviar mensaje al agente
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/view-image/{session_id}/{image_id}</strong> - Ver imagen
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/list-images/{session_id}</strong> - Listar imágenes de sesión
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
                    📚 Documentación API
                </a>
                <a href="/session/new" class="btn btn-secondary" onclick="handleNewSession()">
                    🆕 Nueva Sesión
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
    """
    Main function to run the FastAPI server
    """
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
    # Update global agent URL
    global agent_url
    agent_url = args.agent_url
    print(f"🚀 Starting FastAPI A2A Image Generator Server...")
    print(f"📡 Agent URL: {agent_url}")
    print(f"🌐 Server URL: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    # Run the server using direct app reference instead of module string
    uvicorn.run(
        app,  # Direct reference to the FastAPI app object
        host=args.host,
        port=args.port,
        reload=args.reload
    )
if __name__ == "__main__":
    main()