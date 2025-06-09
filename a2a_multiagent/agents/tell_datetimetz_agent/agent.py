# =============================================================================
# 🎯 Purpose:
# This file defines a simple AI agent called TellDateTimeTimezoneAgent.
# It uses Google's ADK (Agent Development Kit) and Gemini model to respond with:
#   - the current date 🗓️
#   - the current time 🕒
#   - the timezone 🕰️
# This is useful for multi-agent workflows where an agent provides time-related information.
# =============================================================================


# -----------------------------------------------------------------------------
# 📦 Built-in & External Library Imports
# -----------------------------------------------------------------------------

from datetime import datetime
from tzlocal import get_localzone  # Get system timezone

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
load_dotenv()


# -----------------------------------------------------------------------------
# 🕒 TellDateTimeTimezoneAgent: AI agent that tells date, time, and timezone
# -----------------------------------------------------------------------------

class TellDateTimeTimezoneAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        👷 Initialize the TellDateTimeTimezoneAgent:
        - Creates the LLM agent (powered by Gemini)
        - Sets up session handling, memory, and a runner to execute tasks
        """
        self._agent = self._build_agent()
        self._user_id = "datetime_agent_user"

        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        """
        ⚙️ Creates and returns a Gemini agent configured to tell:
            - Current date
            - Current time
            - Timezone
        """
        return LlmAgent(
            model="gemini-1.5-flash-latest",
            name="tell_datetime_timezone_agent",
            description="Tells the current date, time, and timezone",
            instruction=(
                "Reply with the current date, time, and timezone "
                "in the format: 'YYYY-MM-DD HH:MM:SS Timezone_Name'."
            )
        )

    async def invoke(self, query: str, session_id: str) -> str:
        """
        📥 Handle a user query and return a response string with date, time, and timezone.
        
        Args:
            query (str): User input (e.g., "what is the current time?")
            session_id (str): Helps group messages into a session
        
        Returns:
            str: Agent's reply with current date, time, and timezone.
        """

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
                state={}
            )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        last_event = None
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ):
            last_event = event

        if not last_event or not last_event.content or not last_event.content.parts:
            return ""

        return "\n".join([p.text for p in last_event.content.parts if p.text])

    async def stream(self, query: str, session_id: str):
        """
        🌀 Simulates a "streaming" response with current date, time, and timezone.
        
        Yields:
            dict: Response payload with date, time, and timezone.
        """
        local_tz = get_localzone()
        now = datetime.now(local_tz)

        yield {
            "is_task_complete": True,
            "content": now.strftime(f"%Y-%m-%d %H:%M:%S {local_tz}")
        }

# import asyncio

# async def main():
#     agent = TellDateTimeTimezoneAgent()

#     print("=== Testing invoke() ===")
#     result = await agent.invoke("what is current date and time and time zone", "123456asdfgt")
#     print(result)

#     print("\n=== Testing stream() ===")
#     async for response in agent.stream("what is current date and time and time zone", "123456asdfgt"):
#         print(response["content"])

# if __name__ == "__main__":
#     asyncio.run(main())
