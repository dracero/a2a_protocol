import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class AgenteFisica:
    SUPPORTED_CONTENT_TYPES = ["text/plain", "text"]

    def __init__(self):
        self.name = "asistente_fisica"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        self.runner = Runner(
            app_name=self.name,
            agent=self._build_agent(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
            artifact_service=InMemoryArtifactService()
        )

    def _build_agent(self):
        return LlmAgent(
            model="gemini-2.5-flash",
            name=self.name,
            description="Asistente que responde preguntas de Física usando RAG",
            instruction="Responde como un experto en Física utilizando el contexto proporcionado."
        )

    async def invoke(self, query: str, session_id: str) -> str:
        session = await self.runner.session_service.get_session(
            app_name=self.name,
            user_id="fisica_user",
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.name,
                user_id="fisica_user",
                session_id=session_id,
                state={},
            )
        message = types.Content(role="user", parts=[types.Part(text=query)])
        async for event in self.runner.run_async(
            user_id="fisica_user",
            session_id=session_id,
            new_message=message
        ):
            last_event = event
        if last_event and last_event.content and last_event.content.parts:
            return "\n".join([p.text for p in last_event.content.parts if p.text])
        return ""
