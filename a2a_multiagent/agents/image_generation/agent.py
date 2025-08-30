# =============================================================================
# 🎯 Purpose:
# Image Generation Agent using CrewAI framework integrated with A2A protocol.
# - Uses Google Gemini API for image generation
# - Stores images on disk (not in memory as base64)
# - Manages image cache for session persistence
# - Supports both new image generation and image editing
# - Integrates CrewAI agents and tasks for structured AI workflows
# FIXED: Properly returns image URLs to orchestrator via A2A protocol
# =============================================================================

import asyncio
import base64
import logging
import os
import os.path
import re
from io import BytesIO
from typing import Dict
from uuid import uuid4
import json
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Crew, LLM, Task
from crewai.process import Process
from crewai.tools import tool

# Google Gemini imports
from google import genai
from google.genai import types

# PIL for image handling
from PIL import Image
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure image storage directory - FIXED TO USE SCRIPT'S DIRECTORY
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


class ImageData(BaseModel):
    """Represents image data with metadata."""
    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    file_path: str | None = None  # Stores disk path instead of base64
    error: str | None = None


class InMemoryImageCache:
    """Cache that stores image file paths instead of base64 data."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, ImageData]] = {}
    
    def get(self, session_id: str) -> Dict[str, ImageData] | None:
        """Get all images for a session."""
        return self._cache.get(session_id)
    
    def set(self, session_id: str, data: Dict[str, ImageData]) -> None:
        """Set session data."""
        self._cache[session_id] = data
    
    def add_image(self, session_id: str, image_data: ImageData) -> None:
        """Add an image to a session (stores file path)."""
        if session_id not in self._cache:
            self._cache[session_id] = {}
        self._cache[session_id][image_data.id] = image_data


# Global cache instance
image_cache = InMemoryImageCache()


def get_api_key() -> str:
    """Get Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return api_key


def build_image_url(session_id: str, image_id: str) -> str:
    """
    Build the complete URL for accessing an image.
    
    Args:
        session_id: Session identifier
        image_id: Image identifier
        
    Returns:
        str: Complete URL for the image
    """
    # Get host and port from environment, with fallback defaults
    host = os.getenv('SERVER_HOST', 'localhost')
    port = os.getenv('SERVER_PORT', '8000')
    base_url = os.getenv('BASE_URL', f'http://{host}:{port}')
    
    # Ensure base_url doesn't end with slash
    base_url = base_url.rstrip('/')
    
    return f"{base_url}/image/{session_id}/{image_id}"


@tool("ImageGenerationTool")
def generate_image_tool(prompt: str, session_id: str, artifact_file_id: str = None) -> str:
    """
    Generate or edit images based on text prompts.
    
    Args:
        prompt: Text description for image generation/editing
        session_id: Session identifier for cache management
        artifact_file_id: Optional ID of existing image to edit
        
    Returns:
        str: JSON string with image data (ID and URL) or error code
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    try:
        client = genai.Client(api_key=get_api_key())
        
        # Prepare the text input
        text_input = (
            prompt,
            "Generate a high-quality, detailed image based on the description. "
            "If modifying an existing image, make the requested changes while "
            "maintaining overall quality and coherence."
        )

        # Check for reference image
        ref_image = None
        session_data = image_cache.get(session_id)
        
        if session_data:
            ref_image_data = None
            
            # Try to get specific image if artifact_file_id provided
            if artifact_file_id and artifact_file_id in session_data:
                ref_image_data = session_data[artifact_file_id]
                logger.info(f"Using reference image {artifact_file_id} for modification")
            elif not artifact_file_id and session_data:
                # Use the most recent image
                latest_image_key = list(session_data.keys())[-1]
                ref_image_data = session_data[latest_image_key]
                logger.info("Using latest image for modification")
            
            if ref_image_data and ref_image_data.file_path:
                try:
                    # Load reference image from disk
                    ref_image = Image.open(ref_image_data.file_path)
                except Exception as e:
                    logger.warning(f"Failed to load reference image: {e}")
                    ref_image = None

        # Prepare content for the API call
        contents = [text_input, ref_image] if ref_image else text_input

        # Generate image using Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )

        # Process the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    # Create image data object
                    image_id = uuid4().hex
                    extension = part.inline_data.mime_type.split('/')[1]
                    file_path = os.path.join(IMAGE_STORAGE_DIR, f"{image_id}.{extension}")
                    
                    # Save image to disk
                    with open(file_path, "wb") as f:
                        f.write(part.inline_data.data)
                    
                    # Create image data object
                    image_data = ImageData(
                        id=image_id,
                        name=f"image_{image_id}.{extension}",
                        mime_type=part.inline_data.mime_type,
                        file_path=file_path
                    )
                    
                    # Store in cache
                    image_cache.add_image(session_id, image_data)
                    
                    # Build URL for the generated image
                    image_url = build_image_url(session_id, image_id)
                    
                    logger.info(f"Generated image saved to: {file_path}")
                    logger.info(f"Generated image URL: {image_url}")
                    
                    # Return JSON with both ID and URL for the agent to process
                    result = {
                        "success": True,
                        "image_id": image_id,
                        "image_url": image_url,
                        "mime_type": part.inline_data.mime_type,
                        "name": image_data.name
                    }
                    return json.dumps(result)
                    
                except Exception as e:
                    logger.error(f"Error processing generated image: {e}")
                    return json.dumps({"success": False, "error": "ERROR_PROCESSING_IMAGE", "details": str(e)})

        logger.warning("No image data found in response")
        return json.dumps({"success": False, "error": "ERROR_NO_IMAGE_GENERATED"})

    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        return json.dumps({"success": False, "error": "ERROR_GENERATION_FAILED", "details": str(e)})


class ImageGenerationAgent:
    """
    CrewAI-based agent for image generation and editing.
    Integrates with A2A protocol for standardized communication.
    """
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "image/png", "image/jpeg"]

    def __init__(self):
        """Initialize the Image Generation Agent with CrewAI components."""
        try:
            # Initialize the LLM
            self.model = LLM(model="gemini/gemini-2.0-flash", api_key=get_api_key())
            
            # Create the image generation agent
            self.image_creator_agent = Agent(
                role="AI Image Generation Specialist",
                goal=(
                    "Create stunning, high-quality images based on user descriptions "
                    "and edit existing images with precision and creativity. "
                    "Always return the complete image information including URL."
                ),
                backstory=(
                    "You are an advanced AI artist with expertise in visual creation "
                    "and image manipulation. You understand artistic styles, composition, "
                    "lighting, and can transform text descriptions into vivid visual "
                    "representations. You can also skillfully modify existing images "
                    "based on user requests. You always ensure the image URL is properly "
                    "returned for accessibility."
                ),
                verbose=False,
                allow_delegation=False,
                tools=[generate_image_tool],
                llm=self.model,
            )

            # Create the image generation task - FIXED TO PROPERLY HANDLE TOOL RESPONSE
            self.image_creation_task = Task(
                description=(
                    "Process the user's request: '{user_prompt}'\n\n"
                    "Analyze the request to determine if this is:\n"
                    "1. A new image generation request\n"
                    "2. An image editing/modification request\n\n"
                    "For new images: Create a detailed, high-quality image based on the description.\n"
                    "For edits: Look for references to 'this image', 'the previous image', "
                    "'that picture', etc. and use the artifact_file_id if provided.\n\n"
                    "Use the ImageGenerationTool with:\n"
                    "- prompt: The user's request (enhanced if needed for better results)\n"
                    "- session_id: {session_id}\n"
                    "- artifact_file_id: {artifact_file_id} (if editing an existing image)\n\n"
                    "IMPORTANT: The tool will return a JSON response with image data. "
                    "Parse this JSON response and extract the image_id and image_url. "
                    "Return ONLY the complete JSON response from the tool."
                ),
                expected_output=(
                    "The complete JSON response from the ImageGenerationTool containing: "
                    "success status, image_id, image_url, mime_type, and name"
                ),
                agent=self.image_creator_agent,
            )

            # Create the crew
            self.image_crew = Crew(
                agents=[self.image_creator_agent],
                tasks=[self.image_creation_task],
                process=Process.sequential,
                verbose=True,  # Enable verbose for debugging
            )
            
            logger.info("ImageGenerationAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ImageGenerationAgent: {e}")
            raise

    def extract_artifact_file_id(self, query: str) -> str | None:
        """
        Extract artifact file ID from user query if present.
        
        Args:
            query: User's input text
            
        Returns:
            str | None: Extracted artifact ID or None if not found
        """
        try:
            # Look for various patterns that might indicate an artifact ID
            patterns = [
                r'(?:id|artifact[_-]?file[_-]?id)\s*[:\s]\s*([0-9a-f]{32})',
                r'artifact[_-]?id\s*[:\s]\s*([0-9a-f]{32})',
                r'image[_-]?id\s*[:\s]\s*([0-9a-f]{32})',
                r'([0-9a-f]{32})',  # Any 32-char hex string
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    return match.group(1)
                    
            return None
        except Exception as e:
            logger.warning(f"Error extracting artifact ID: {e}")
            return None

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Process user request and generate/edit images following A2A protocol standards.
        
        Args:
            query: User's text input
            session_id: Session identifier
            
        Returns:
            str: JSON-RPC 2.0 formatted response per A2A protocol specifications
        """
        try:
            # Generate a unique request ID for JSON-RPC tracking
            request_id = uuid4().hex
            
            # Handle image retrieval requests according to A2A standards
            if query.upper().startswith(("GET_IMAGE_DATA", "RETRIEVE_IMAGE", "SHOW_IMAGE")):
                parts = query.split()
                if len(parts) > 1:
                    image_id = parts[-1]
                    image_data = self.get_image_data(session_id, image_id)
                    
                    if image_data.error:
                        # Return proper JSON-RPC error response
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": 404,
                                "message": "Image not found",
                                "data": image_data.error
                            },
                            "id": request_id
                        }
                        return json.dumps(error_response)
                    else:
                        # Generate URL for image
                        url = build_image_url(session_id, image_id)
                        
                        # Return proper JSON-RPC response with image URL
                        success_response = {
                            "jsonrpc": "2.0",
                            "result": {
                                "found": True,
                                "url": url,
                                "mime_type": image_data.mime_type,
                                "name": image_data.name,
                                "id": image_data.id
                            },
                            "id": request_id
                        }
                        return json.dumps(success_response)
                else:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 400,
                            "message": "Bad Request",
                            "data": "Image ID missing in request"
                        },
                        "id": request_id
                    }
                    return json.dumps(error_response)
            
            # Extract artifact file ID if present
            artifact_file_id = self.extract_artifact_file_id(query)
            
            # Prepare inputs for CrewAI
            inputs = {
                "user_prompt": query,
                "session_id": session_id,
                "artifact_file_id": artifact_file_id or ""
            }
            
            logger.info(f"Processing request with inputs: {inputs}")
            
            # Execute the crew
            crew_response = self.image_crew.kickoff(inputs)
            response_text = str(crew_response).strip() if crew_response else ""
            
            logger.info(f"Raw crew response: {response_text}")
            
            if not response_text:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": 500,
                        "message": "No response from image generation crew",
                        "data": "ERROR_NO_RESPONSE"
                    },
                    "id": request_id
                }
                return json.dumps(error_response)
            
            # Try to parse the crew response as JSON
            try:
                # The crew should return the JSON from the tool
                tool_result = json.loads(response_text)
                
                if not tool_result.get("success"):
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 500,
                            "message": "Image generation failed",
                            "data": tool_result.get("error", "Unknown error")
                        },
                        "id": request_id
                    }
                    return json.dumps(error_response)
                
                # Extract image data from tool result
                image_id = tool_result.get("image_id")
                image_url = tool_result.get("image_url")
                mime_type = tool_result.get("mime_type")
                name = tool_result.get("name")
                
                # Return proper JSON-RPC response with image URL
                success_response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "id": image_id,
                        "name": name,
                        "mime_type": mime_type,
                        "url": image_url,
                        "status": "generated"
                    },
                    "id": request_id
                }
                
                logger.info(f"Returning success response with URL: {image_url}")
                return json.dumps(success_response)
                
            except json.JSONDecodeError:
                # If response is not JSON, treat it as an error or plain text
                logger.warning(f"Could not parse crew response as JSON: {response_text}")
                
                # Check if it's an error message
                if response_text.startswith("ERROR"):
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 500,
                            "message": "Image generation failed",
                            "data": response_text
                        },
                        "id": request_id
                    }
                    return json.dumps(error_response)
                else:
                    # Treat as unexpected response format
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 500,
                            "message": "Unexpected response format",
                            "data": response_text
                        },
                        "id": request_id
                    }
                    return json.dumps(error_response)
                
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            request_id = uuid4().hex
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "data": f"ERROR_INVOKE_FAILED: {str(e)}"
                },
                "id": request_id
            }
            return json.dumps(error_response)

    def get_image_data(self, session_id: str, image_key: str) -> ImageData:
        """
        Retrieve image data from cache.
        
        Args:
            session_id: Session identifier
            image_key: Image identifier
            
        Returns:
            ImageData: Image data object with file path
        """
        try:
            session_data = image_cache.get(session_id)
            if session_data and image_key in session_data:
                return session_data[image_key]
            else:
                logger.warning(f"Image {image_key} not found in session {session_id}")
                return ImageData(error="Image not found. Please try generating a new image.")
        except Exception as e:
            logger.error(f"Error retrieving image data: {e}")
            return ImageData(error="Error retrieving image. Please try again.")