import logging
import asyncio
import click
from server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from agents.image_generator.agent import ImageGeneratorAgent
from agents.image_generator.task_manager import ImageGeneratorTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost", help="Host para Image Generator")
@click.option("--port", default=10003, help="Puerto del servidor del agente")
def main(host: str, port: int):
    print(f"🚀 Iniciando Agente de Generación de Imágenes en http://{host}:{port}/\n")

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="image_generation",
        name="Generador de Imágenes con CrewAI",
        description="Genera y modifica imágenes usando CrewAI y Gemini Image Generation",
        tags=["imagen", "generacion", "crewai", "gemini"],
        examples=[
            "Genera una imagen de un gato espacial explorando una ciudad futurista",
            "Crea una imagen de un dragón volando sobre montañas",
            "Genera una imagen artística de un atardecer en la playa"
        ]
    )
    agent_card = AgentCard(
        name="ImageGeneratorAgent",
        description="Agente especializado en generación de imágenes usando CrewAI y Gemini",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text", "image"],
        capabilities=capabilities,
        skills=[skill]
    )

    # Inicializar el agente
    print("🔧 Inicializando agente de generación de imágenes...")
    agente = ImageGeneratorAgent()
    
    # Inicializar componentes del agente
    try:
        agente.inicializar_componentes()
        print("✅ Componentes del agente inicializados")
        
        # Inicializar el agent_card
        agente.inicializar_agent_card(host=host, port=port)
        print("✅ Agent card inicializado")
            
    except Exception as e:
        logger.error(f"Error inicializando agente: {e}")
        print(f"❌ Error inicializando agente: {e}")
        # Continuar pero con funcionalidad limitada
    
    # Crear task manager y servidor
    task_manager = ImageGeneratorTaskManager(agent=agente)
    server = A2AServer(host=host, port=port, agent_card=agente.agent_card, task_manager=task_manager)
    
    print(f"✅ Servidor listo en http://{host}:{port}/")
    server.start()

if __name__ == "__main__":
    main()