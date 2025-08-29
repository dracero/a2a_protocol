# =============================================================================
# 🎯 Purpose:
# Image Generation Agent using CrewAI framework integrated with A2A protocol.
# - Uses Google Gemini API for image generation
# - Stores images on disk (not in memory as base64)
# - Manages image cache for session persistence
# - Supports both new image generation and image editing
# - Integrates CrewAI agents and tasks for structured AI workflows
# =============================================================================

import asyncio
import base64
import logging
import os
import re
from io import BytesIO
from typing import Dict
from uuid import uuid4
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
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", "generated_images")
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


@tool("ImageGenerationTool")
def generate_image_tool(prompt: str, session_id: str, artifact_file_id: str = None) -> str:
    """
    Generate or edit images based on text prompts.
    
    Args:
        prompt: Text description for image generation/editing
        session_id: Session identifier for cache management
        artifact_file_id: Optional ID of existing image to edit
        
    Returns:
        str: Generated image ID or error code
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
                    
                    logger.info(f"Generated image saved to: {file_path}")
                    return image_data.id
                    
                except Exception as e:
                    logger.error(f"Error processing generated image: {e}")
                    return "ERROR_PROCESSING_IMAGE"

        logger.warning("No image data found in response")
        return "ERROR_NO_IMAGE_GENERATED"

    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        return "ERROR_GENERATION_FAILED"


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
                    "and edit existing images with precision and creativity."
                ),
                backstory=(
                    "You are an advanced AI artist with expertise in visual creation "
                    "and image manipulation. You understand artistic styles, composition, "
                    "lighting, and can transform text descriptions into vivid visual "
                    "representations. You can also skillfully modify existing images "
                    "based on user requests."
                ),
                verbose=False,
                allow_delegation=False,
                tools=[generate_image_tool],
                llm=self.model,
            )

            # Create the image generation task
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
                    "Return the generated image ID."
                ),
                expected_output="The unique ID of the generated or modified image",
                agent=self.image_creator_agent,
            )

            # Create the crew
            self.image_crew = Crew(
                agents=[self.image_creator_agent],
                tasks=[self.image_creation_task],
                process=Process.sequential,
                verbose=False,
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
        Process user request and generate/edit images.
        
        Args:
            query: User's text input
            session_id: Session identifier
            
        Returns:
            str: Generated image ID or error message
        """
        try:
            # ✅ CORRECTED: Return JSON_DATA with image file path
            if query.upper().startswith(("GET_IMAGE_DATA", "RETRIEVE_IMAGE", "SHOW_IMAGE")):
                parts = query.split()
                if len(parts) > 1:
                    image_id = parts[-1]  # Last part should be the image ID
                    image_data = self.get_image_data(session_id, image_id)
                    
                    if image_data.error:
                        return f"JSON_DATA: {{\"found\": false, \"error\": \"{image_data.error}\"}}"
                    else:
                        # Return JSON with file path instead of base64
                        return f"JSON_DATA: {{\"found\": true, \"file_path\": \"{image_data.file_path}\", \"mime_type\": \"{image_data.mime_type}\", \"name\": \"{image_data.name}\"}}"
                return "JSON_DATA: {\"found\": false, \"error\": \"Image ID missing\"}"
            
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
            response = self.image_crew.kickoff(inputs)
            
            # Return the response (should be the image ID)
            return str(response).strip() if response else "ERROR_NO_RESPONSE"
            
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            return f"ERROR_INVOKE_FAILED: {str(e)}"

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