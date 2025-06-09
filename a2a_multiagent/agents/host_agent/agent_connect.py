# =============================================================================
# 🎯 Purpose:
# Provides a simple wrapper (`AgentConnector`) around the A2AClient to send tasks
# to any remote agent identified by a base URL. This decouples the Orchestrator
# from low-level HTTP details and HTTP client setup.
# =============================================================================

import uuid
import logging

from client.client import A2AClient
from models.task import Task

logger = logging.getLogger(__name__)

class AgentConnector:
    """
    🔗 Connects to a remote A2A agent and provides a uniform method to delegate tasks.

    Attributes:
        name (str): Human-readable identifier of the remote agent.
        client (A2AClient): HTTP client pointing at the agent's URL.
    """

    def __init__(self, name: str, base_url: str):
        """
        Initialize the connector for a specific remote agent.

        Args:
            name (str): Identifier for the agent (e.g., "TellDateTimeTimezoneAgent").
            base_url (str): The HTTP endpoint (e.g., "http://localhost:10000").
        """
        self.name = name
        self.client = A2AClient(url=base_url)
        logger.info(f"AgentConnector: initialized for {self.name} at {base_url}")

    async def send_task(self, message: str, session_id: str) -> Task:
        """
        Send a text task to the remote agent and return its completed Task.

        Args:
            message (str): What you want the agent to do (e.g., "What is the current date and time?").
            session_id (str): Session identifier to group related calls.

        Returns:
            Task: The full Task object (including history) from the remote agent.
        """
        task_id = uuid.uuid4().hex
        payload = {
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": message}
                ]
            }
        }

        task_result = await self.client.send_task(payload)
        logger.info(f"AgentConnector: received response from {self.name} for task {task_id}")
        return task_result
