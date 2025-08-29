# =============================================================================
# 🎯 Purpose:
# Connects the ImageGenerationAgent class to the Agent-to-Agent (A2A) protocol by
# handling incoming JSON-RPC "tasks/send" requests. It:
# 1. Receives a SendTaskRequest model
# 2. Stores the user message in memory
# 3. Calls ImageGenerationAgent.invoke() to generate/edit images
# 4. Returns the image ID (not base64 data)
# 5. Returns a SendTaskResponse containing the completed Task
# =============================================================================

import logging
import json
from typing import AsyncIterable

# A2A Framework imports
from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Task, TaskStatus, TaskState, Message, TextPart

# Our custom agent
from agents.image_generation.agent import ImageGenerationAgent, ImageData

logger = logging.getLogger(__name__)


class ImageGenerationTaskManager(InMemoryTaskManager):
    """
    🧩 TaskManager for ImageGenerationAgent:

    - Handles image generation/editing requests
    - Returns ONLY image IDs (not base64 data)
    - Properly handles all response types
    """

    def __init__(self, agent: ImageGenerationAgent):
        super().__init__()
        self.agent = agent

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """Extract the raw user text from the incoming request"""
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Handle image generation requests without base64 token overflow.
        
        Flow:
        1. Store incoming message
        2. Call agent with user text
        3. If response is error -> return error
        4. If response is image ID -> return success with ID
        """
        try:
            logger.info(f"ImageGenerationTaskManager received task {request.params.id}")
            task = await self.upsert_task(request.params)
            user_text = self._get_user_text(request)
            logger.info(f"Processing user request: {user_text[:100]}...")

            # Get response from agent (now returns ONLY image ID)
            image_id = await self.agent.invoke(user_text, request.params.sessionId)

            # Handle errors
            if isinstance(image_id, str) and image_id.startswith("ERROR"):
                error_msg = f"Image generation failed: {image_id}"
                logger.error(error_msg)
                return self._create_error_response(request, error_msg)

            # Handle missing image ID (for GET_IMAGE_DATA requests)
            if image_id == "IMAGE_ID_MISSING":
                error_msg = "Image ID missing in request"
                logger.error(error_msg)
                return self._create_error_response(request, error_msg)

            # ✅ CORRECTED: Only return image ID as response
            # Client will use this ID to fetch the actual image via /image/{session_id}/{image_id}
            response_text = f"✅ Image ready! Use ID: {image_id} to fetch the image."

            # Create reply message
            reply_message = Message(
                role="agent",
                parts=[TextPart(text=response_text)]
            )

            # Update task status
            async with self.lock:
                task.status = TaskStatus(state=TaskState.COMPLETED)
                task.history.append(reply_message)

            logger.info(f"Successfully completed task {task.id}")
            return SendTaskResponse(id=request.id, result=task)

        except Exception as e:
            logger.error(f"Error in on_send_task: {e}", exc_info=True)
            return self._create_error_response(request, f"Internal error: {str(e)}")

    async def on_send_task_streaming(self, request) -> AsyncIterable[SendTaskResponse]:
        """Streaming not supported by this agent"""
        logger.warning("Streaming not supported by CrewAI agent")
        yield self._create_error_response(
            request, 
            "Streaming is not supported by this agent. Please use regular task mode."
        )
    
    def _create_error_response(self, request: SendTaskRequest, error_message: str) -> SendTaskResponse:
        """Create a standardized error response"""
        task = Task(
            id=request.params.id,
            sessionId=request.params.sessionId,
            status=TaskStatus(state=TaskState.FAILED),
            history=[
                request.params.message,
                Message(
                    role="agent",
                    parts=[TextPart(text=f"❌ Error: {error_message}")]
                )
            ]
        )
        return SendTaskResponse(id=request.id, result=task)