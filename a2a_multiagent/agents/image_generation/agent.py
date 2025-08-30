# =============================================================================
# 🎯 Purpose:
# FIXED Image Generation Agent using CrewAI framework integrated with A2A protocol.
# 
# KEY FIXES:
# 1. Simplified task description to ensure proper JSON response parsing
# 2. Better error handling and response formatting
# 3. Clearer instructions for the CrewAI agent
# 4. Improved URL building and response structure
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

# Configure image storage directory
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


class ImageData(BaseModel):
    """Represents image data with metadata."""
    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    file_path: str | None = None
    error: str | None = None


class InMemoryImageCache:
    """Cache that stores image file paths instead of base64 data."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, ImageData]] = {}
    
    def get(self, session_id: str) -> Dict[str, ImageData] | None:
        return self._cache.get(session_id)
    
    def set(self, session_id: str, data: Dict[str, ImageData]) -> None:
        self._cache[session_id] = data
    
    def add_image(self, session_id: str, image_data: ImageData) -> None:
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
    """Build the complete URL for accessing an image."""
    host = os.getenv('SERVER_HOST', 'localhost')
    port = os.getenv('SERVER_PORT', '8000')
    base_url = os.getenv('BASE_URL', f'http://{host}:{port}')
    base_url = base_url.rstrip('/')
    return f"{base_url}/image/{session_id}/{image_id}"


@tool("ImageGenerationTool")
def generate_image_tool(prompt: str, session_id: str, artifact_file_id: str = None) -> str:
    """
    Generate or edit images based on text prompts.
    
    Returns JSON string with image data (ID and URL) or error information.
    """
    if not prompt:
        return json.dumps({"success": False, "error": "Prompt cannot be empty"})

    try:
        client = genai.Client(api_key=get_api_key())
        
        # Prepare the text input
        text_input = (
            prompt,
            "Generate a high-quality, detailed image based on the description."
        )

        # Check for reference image
        ref_image = None
        session_data = image_cache.get(session_id)
        
        if session_data and artifact_file_id and artifact_file_id in session_data:
            ref_image_data = session_data[artifact_file_id]
            if ref_image_data.file_path:
                try:
                    ref_image = Image.open(ref_image_data.file_path)
                    logger.info(f"Using reference image {artifact_file_id} for modification")
                except Exception as e:
                    logger.warning(f"Failed to load reference image: {e}")

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
                    
                    # Return JSON with image information
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
                    return json.dumps({
                        "success": False, 
                        "error": "Error processing generated image", 
                        "details": str(e)
                    })

        logger.warning("No image data found in response")
        return json.dumps({"success": False, "error": "No image data found in response"})

    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        return json.dumps({
            "success": False, 
            "error": "Image generation failed", 
            "details": str(e)
        })


class ImageGenerationAgent:
    """
    FIXED CrewAI-based agent for image generation and editing.
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
                    "and return the complete JSON response from the image generation tool."
                ),
                backstory=(
                    "You are an advanced AI artist with expertise in visual creation. "
                    "When using the ImageGenerationTool, you must return the EXACT JSON "
                    "response from the tool without modification or interpretation."
                ),
                verbose=False,
                allow_delegation=False,
                tools=[generate_image_tool],
                llm=self.model,
            )

            # FIXED: Simplified task description for better JSON handling
            self.image_creation_task = Task(
                description=(
                    "Process the user's image request: '{user_prompt}'\n\n"
                    "Use the ImageGenerationTool with these parameters:\n"
                    "- prompt: {user_prompt}\n"
                    "- session_id: {session_id}\n"
                    "- artifact_file_id: {artifact_file_id}\n\n"
                    "CRITICAL: Return ONLY the JSON response from the tool. "
                    "Do not add any additional text or formatting."
                ),
                expected_output="The exact JSON response from the ImageGenerationTool",
                agent=self.image_creator_agent,
            )

            # Create the crew
            self.image_crew = Crew(
                agents=[self.image_creator_agent],
                tasks=[self.image_creation_task],
                process=Process.sequential,
                verbose=False,  # Reduced verbosity for cleaner output
            )
            
            logger.info("ImageGenerationAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ImageGenerationAgent: {e}")
            raise

    def extract_artifact_file_id(self, query: str) -> str | None:
        """Extract artifact file ID from user query if present."""
        try:
            patterns = [
                r'(?:id|artifact[_-]?file[_-]?id)\s*[:\s]\s*([0-9a-f]{32})',
                r'artifact[_-]?id\s*[:\s]\s*([0-9a-f]{32})',
                r'image[_-]?id\s*[:\s]\s*([0-9a-f]{32})',
                r'([0-9a-f]{32})',
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
        FIXED: Process user request and return properly formatted JSON response.
        """
        try:
            request_id = uuid4().hex
            
            # Handle image retrieval requests
            if query.upper().startswith(("GET_IMAGE_DATA", "RETRIEVE_IMAGE", "SHOW_IMAGE")):
                parts = query.split()
                if len(parts) > 1:
                    image_id = parts[-1]
                    image_data = self.get_image_data(session_id, image_id)
                    
                    if image_data.error:
                        return json.dumps({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": 404,
                                "message": "Image not found",
                                "data": image_data.error
                            },
                            "id": request_id
                        })
                    else:
                        url = build_image_url(session_id, image_id)
                        return json.dumps({
                            "jsonrpc": "2.0",
                            "result": {
                                "found": True,
                                "url": url,
                                "mime_type": image_data.mime_type,
                                "name": image_data.name,
                                "id": image_data.id
                            },
                            "id": request_id
                        })
                else:
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 400,
                            "message": "Bad Request",
                            "data": "Image ID missing in request"
                        },
                        "id": request_id
                    })
            
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
                return json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": 500,
                        "message": "No response from image generation crew"
                    },
                    "id": request_id
                })
            
            # FIXED: Try to parse the crew response as JSON
            try:
                tool_result = json.loads(response_text)
                
                if not tool_result.get("success"):
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 500,
                            "message": "Image generation failed",
                            "data": tool_result.get("error", "Unknown error")
                        },
                        "id": request_id
                    })
                
                # Return proper JSON-RPC response with image information
                return json.dumps({
                    "jsonrpc": "2.0",
                    "result": {
                        "id": tool_result.get("image_id"),
                        "name": tool_result.get("name"),
                        "mime_type": tool_result.get("mime_type"),
                        "url": tool_result.get("image_url"),
                        "status": "generated"
                    },
                    "id": request_id
                })
                
            except json.JSONDecodeError:
                logger.warning(f"Could not parse crew response as JSON: {response_text}")
                
                # If response contains error indicators
                if any(error_word in response_text.lower() for error_word in ["error", "failed", "unable"]):
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": 500,
                            "message": "Image generation failed",
                            "data": response_text
                        },
                        "id": request_id
                    })
                else:
                    # Return the raw response if it's not an error
                    return response_text
                
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "data": f"Error: {str(e)}"
                },
                "id": uuid4().hex
            })

    def get_image_data(self, session_id: str, image_key: str) -> ImageData:
        """Retrieve image data from cache."""
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