import logging
import click
from server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from agents.asistente_fisica.agent import AsistenteFisica
from agents.asistente_fisica.task_manager import FisicaTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost", help="Host para Asistente Fisica")
@click.option("--port", default=10002, help="Puerto del servidor del agente")
def main(host: str, port: int):
    print(f"🚀 Iniciando Agente de Física en http://{host}:{port}/\n")

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="physics_rag",
        name="Asistente de Física I",
        description="Responde consultas de Física utilizando recuperación aumentada de información",
        tags=["fisica", "rag", "pdf"],
        examples=["Explicá el principio de conservación de la energía", "¿Qué es el centro de masa?"]
    )
    agent_card = AgentCard(
        name="AsistenteFisica",
        description="Asistente para consultas de Física I con RAG",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=capabilities,
        skills=[skill]
    )

    agente = AsistenteFisica()
    task_manager = FisicaTaskManager(agent=agente)
    server = A2AServer(host=host, port=port, agent_card=agent_card, task_manager=task_manager)
    server.start()

if __name__ == "__main__":
    main()
