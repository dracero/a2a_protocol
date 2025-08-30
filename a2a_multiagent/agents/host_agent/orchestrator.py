# =============================================================================
# 🎯 Purpose:
# FIXED VERSION: Defines the OrchestratorAgent that uses a Gemini-based LLM to interpret user
# queries and delegate them to any child A2A agent discovered at startup.
# 
# KEY FIXES:
# 1. Properly handles JSON responses from image generation agents
# 2. Parses and formats image URLs for user presentation
# 3. Better error handling for different response types
# =============================================================================

import os                           # Standard library for interacting with the operating system
import uuid                         # For generating unique identifiers (e.g., session IDs)
import logging                      # Standard library for configurable logging
import json                         # For parsing JSON responses
from dotenv import load_dotenv      # Utility to load environment variables from a .env file

# Print current working directory and .env location for debug
env_path = os.path.join(os.getcwd(), '.env')
print(f"[DEBUG] CWD: {os.getcwd()} | Looking for .env at: {env_path}")

# Load the .env file so that environment variables like GOOGLE_API_KEY
# are available to the ADK client when creating LLMs
load_dotenv(dotenv_path=env_path)

# Check if GOOGLE_API_KEY is loaded
if not os.environ.get("GOOGLE_API_KEY"):
    print("[WARNING] GOOGLE_API_KEY not found in environment! Check your .env file and location.")
else:
    print("[DEBUG] GOOGLE_API_KEY loaded from environment.")

# -----------------------------------------------------------------------------
# Google ADK / Gemini imports
# -----------------------------------------------------------------------------
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types           

# -----------------------------------------------------------------------------
# A2A server-side infrastructure
# -----------------------------------------------------------------------------
from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart

# -----------------------------------------------------------------------------
# Connector to child A2A agents
# -----------------------------------------------------------------------------
from agents.host_agent.agent_connect import AgentConnector
from models.agent import AgentCard

# Set up module-level logger for debug/info messages
logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    🤖 Uses a Gemini LLM to route incoming user queries,
    calling out to any discovered child A2A agents via tools.
    
    FIXED: Now properly handles JSON responses from image generation agents.
    """

    # Define supported MIME types for input/output
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, agent_cards: list[AgentCard]):
        # Build one AgentConnector per discovered AgentCard
        self.connectors = {
            card.name: AgentConnector(card.name, card.url)
            for card in agent_cards
        }

        # Build the internal LLM agent with our custom tools and instructions
        self._agent = self._build_agent()

        # Static user ID for session tracking across calls
        self._user_id = "orchestrator_user"

        # Runner wires up sessions, memory, artifacts, and handles agent.run()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        """
        Construct the Gemini-based LlmAgent with:
        - Model name
        - Agent name/description
        - System instruction callback
        - Available tool functions
        """
        return LlmAgent(
            model="gemini-2.0-flash",
            name="orchestrator_agent",
            description="Delegates user queries to child A2A agents based on intent.",
            instruction=self._root_instruction,
            tools=[
                self._list_agents,
                self.delegate_task
            ],
        )

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """
        System prompt function: returns instruction text for the LLM,
        including which tools it can use and a list of child agents.
        """
        # Build a bullet-list of agent names
        agent_list = "\n".join(f"- {name}" for name in self.connectors)
        return (
            "You are an orchestrator with two tools:\n"
            "1) list_agents() -> list available child agents\n"
            "2) delegate_task(agent_name, message) -> call that agent\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Use these tools to satisfy the user's requests\n"
            "- For image generation requests, delegate to ImageGenerationAgent\n"
            "- When you receive an image URL from an agent, present it clearly to the user\n"
            "- If an agent returns JSON with an image URL, extract and display the URL\n"
            "- Always be helpful and provide clear responses\n\n"
            "Available agents:\n" + agent_list
        )

    def _list_agents(self) -> list[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        """
        return list(self.connectors.keys())

    async def delegate_task(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext
    ) -> str:
        """
        FIXED Tool function: forwards the message to the specified child agent,
        properly handles JSON responses, and extracts relevant information.
        """
        # Validate agent_name exists
        if agent_name not in self.connectors:
            raise ValueError(f"Unknown agent: {agent_name}")
        connector = self.connectors[agent_name]

        # Ensure session_id persists across tool calls via tool_context.state
        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        try:
            # Delegate task asynchronously and await Task result
            child_task = await connector.send_task(message, session_id)

            # Extract response from the last history entry
            if not child_task.history or len(child_task.history) < 2:
                return f"No response received from {agent_name}"
            
            response_text = child_task.history[-1].parts[0].text
            
            # FIXED: Handle different response types from child agents
            
            # Try to parse as JSON first (for structured responses like image generation)
            try:
                response_data = json.loads(response_text)
                
                # Handle JSON-RPC 2.0 responses
                if isinstance(response_data, dict):
                    # Check for JSON-RPC error
                    if "error" in response_data:
                        error_info = response_data["error"]
                        return f"Error from {agent_name}: {error_info.get('message', 'Unknown error')}"
                    
                    # Check for JSON-RPC result
                    if "result" in response_data:
                        result = response_data["result"]
                        
                        # Handle image generation results
                        if isinstance(result, dict) and "url" in result:
                            image_url = result["url"]
                            image_id = result.get("id", "unknown")
                            return (
                                f"✅ Image generated successfully!\n\n"
                                f"🖼️ **Image URL**: {image_url}\n"
                                f"🆔 **Image ID**: {image_id}\n\n"
                                f"You can view your image by clicking the URL above or using the image ID."
                            )
                        
                        # Handle other structured results
                        if isinstance(result, dict):
                            # Format the result nicely
                            formatted_result = []
                            for key, value in result.items():
                                formatted_result.append(f"**{key.title()}**: {value}")
                            return f"Result from {agent_name}:\n" + "\n".join(formatted_result)
                        
                        # Simple result
                        return f"Result from {agent_name}: {result}"
                    
                    # Handle direct JSON responses (legacy format)
                    if "success" in response_data:
                        if response_data["success"]:
                            image_url = response_data.get("image_url")
                            image_id = response_data.get("image_id")
                            if image_url:
                                return (
                                    f"✅ Image generated successfully!\n\n"
                                    f"🖼️ **Image URL**: {image_url}\n"
                                    f"🆔 **Image ID**: {image_id}\n\n"
                                    f"You can view your image by clicking the URL above."
                                )
                        else:
                            error_msg = response_data.get("error", "Unknown error")
                            return f"❌ Error from {agent_name}: {error_msg}"
                    
                    # Generic JSON response - format it nicely
                    formatted_json = json.dumps(response_data, indent=2)
                    return f"Response from {agent_name}:\n```json\n{formatted_json}\n```"
                        
            except json.JSONDecodeError:
                # Not JSON - treat as plain text response
                pass
            
            # Handle plain text responses
            if response_text.strip():
                # Check if it contains an image URL pattern
                if "http" in response_text and ("image" in response_text.lower() or "png" in response_text.lower() or "jpg" in response_text.lower()):
                    # Try to extract URLs from the text
                    import re
                    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    urls = re.findall(url_pattern, response_text)
                    if urls:
                        formatted_response = response_text + "\n\n🔗 **Direct link**: " + urls[0]
                        return formatted_response
                
                return f"Response from {agent_name}: {response_text}"
            
            return f"Empty response received from {agent_name}"
            
        except Exception as e:
            logger.error(f"Error delegating task to {agent_name}: {e}")
            return f"❌ Error communicating with {agent_name}: {str(e)}"

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Main entry: receives a user query + session_id,
        sets up or retrieves a session, wraps the query for the LLM,
        runs the Runner (with tools enabled), and returns the final text.
        """
        # Attempt to reuse an existing session
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id
        )
        # Create new if not found
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={}
            )

        # Wrap the user query in a types.Content message
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        # 🚀 Run the agent using the Runner and collect the last event
        last_event = None
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ):
            last_event = event

        # 🧹 Fallback: return empty string if something went wrong
        if not last_event or not last_event.content or not last_event.content.parts:
            return "I apologize, but I wasn't able to process your request properly. Please try again."

        # 📤 Extract and join all text responses into one string
        return "\n".join([p.text for p in last_event.content.parts if p.text])


class OrchestratorTaskManager(InMemoryTaskManager):
    """
    🪄 TaskManager wrapper: exposes OrchestratorAgent.invoke() over the
    A2A JSON-RPC `tasks/send` endpoint.
    """
    def __init__(self, agent: OrchestratorAgent):
        super().__init__()
        self.agent = agent

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Helper: extract the user's raw input text from the request object.
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Called by the A2A server when a new task arrives.
        """
        logger.info(f"OrchestratorTaskManager received task {request.params.id}")

        # Step 1: save the initial message
        task = await self.upsert_task(request.params)

        # Step 2: run orchestration logic
        user_text = self._get_user_text(request)
        response_text = await self.agent.invoke(user_text, request.params.sessionId)

        # Step 3: wrap the LLM output into a Message
        reply = Message(role="agent", parts=[TextPart(text=response_text)])
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply)

        # Step 4: return structured response
        return SendTaskResponse(id=request.id, result=task)