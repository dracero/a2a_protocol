# =============================================================================
# 🎯 Purpose:
# This file defines a simple AI agent called TellDateTimeZoneAgent.
# It uses Google's ADK (Agent Development Kit) and Gemini model to respond with
# the current date, time, and time zone.
#
# The agent supports:
# - Plain text input/output
# - Session handling (conversation memory)
# - Single response or simulated "streaming" response
# =============================================================================

# -----------------------------------------------------------------------------
# 📦 Built-in & External Library Imports
# -----------------------------------------------------------------------------

from datetime import datetime                    # Used to get current date and time
import pytz                                      # Used for timezone formatting
import tzlocal                                   # Used to get system timezone name

# 🧠 Gemini-based AI agent provided by Google's ADK
from google.adk.agents.llm_agent import LlmAgent

# 📚 ADK services for session, memory, and file-like "artifacts"
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService

# 🏃 The "Runner" connects the agent, session, memory, and files into a complete system
from google.adk.runners import Runner

# 🧾 Gemini-compatible types for formatting input/output messages
from google.genai import types

# 🔐 Load environment variables (like API keys) from a `.env` file
from dotenv import load_dotenv
load_dotenv()  # Load variables like GOOGLE_API_KEY into the system


# -----------------------------------------------------------------------------
# ⏳ Helper Function: Get System Timezone Name
# -----------------------------------------------------------------------------

def get_system_timezone_name():
    """
    Returns the system's timezone name as a string.
    Example: 'Europe/London', 'Asia/Kolkata', 'America/New_York'

    Returns:
        str: Timezone name or 'Unknown Timezone' if it cannot be determined.
    """
    try:
        # Get local timezone name (e.g. 'Europe/London' or 'Asia/Kolkata')
        tz = tzlocal.get_localzone()
        return str(tz)
    except Exception as e:
        print(f"Error getting timezone: {e}")
        return "Unknown Timezone"


# -----------------------------------------------------------------------------
# 🕒 TellDateTimeZoneAgent: AI agent that tells date, time, and time zone
# -----------------------------------------------------------------------------

class TellDateTimeZoneAgent:
    # This agent only supports plain text input/output
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        👷 Initialize the TellDateTimeZoneAgent:
        - Creates the LLM agent (powered by Gemini)
        - Sets up session handling, memory, and a runner to execute tasks
        """
        self._agent = self._build_agent()  # Set up the Gemini agent
        self._user_id = "datetime_agent_user"  # Use a fixed user ID for simplicity

        # 🧠 The Runner is what actually manages the agent and its environment
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),  # For files (not used here)
            session_service=InMemorySessionService(),    # Keeps track of conversations
            memory_service=InMemoryMemoryService(),      # Optional: remembers past messages
        )

    def _build_agent(self) -> LlmAgent:
        """
        ⚙️ Creates and returns a Gemini agent with basic settings.

        Returns:
            LlmAgent: An agent object from Google's ADK
        """
        return LlmAgent(
            model="gemini-2.5-flash",                # Gemini model version
            name="tell_date_time_zone_agent",               # Name of the agent
            description="Tells the current date, time, and time zone",  # Description for metadata
            instruction="Reply with the current date, time, and time zone in the format YYYY-MM-DD HH:MM:SS TZ."  # System prompt
        )

    async def invoke(self, query: str, session_id: str) -> str:
        """
        📥 Handle a user query and return a response string.

        Args:
            query (str): What the user said (e.g., "What time is it?")
            session_id (str): Helps group messages into a session

        Returns:
            str: Agent's reply (usually current date, time, and time zone)
        """

        # 🔁 Try to reuse an existing session (or create one if needed)
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id
        )

        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={}  # Optional dictionary to hold session state
            )

        # 📨 Format the user message in a way the Gemini model expects
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
            return ""

        # 📤 Extract and join all text responses into one string
        return "\n".join([p.text for p in last_event.content.parts if p.text])

    async def stream(self, query: str, session_id: str):
        """
        🌀 Simulates a "streaming" agent that returns a single reply.

        Yields:
            dict: Response payload that says the task is complete and gives
            the current date, time, and time zone.
        """
        # Get system timezone name (example: 'Asia/Kolkata', 'Europe/London', etc.)
        timezone_name = get_system_timezone_name()

        try:
            # Load timezone using pytz
            tz = pytz.timezone(timezone_name) if timezone_name != "Unknown Timezone" else pytz.utc
        except Exception as e:
            print(f"Error using timezone '{timezone_name}': {e}. Falling back to UTC.")
            tz = pytz.utc

        # Get current date and time in selected timezone
        now = datetime.now(tz)

        # Format as YYYY-MM-DD HH:MM:SS TZ
        formatted_datetime = now.strftime('%Y-%m-%d %H:%M:%S %Z')

        # Return as streaming response
        yield {
            "is_task_complete": True,
            "content": f"The current date, time, and timezone is: {formatted_datetime}"
        }
