import logging
import asyncio
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

    # IMPORTANTE: Inicializar el agente correctamente
    print("🔧 Inicializando agente de física...")
    agente = AsistenteFisica()
    
    # Inicializar componentes del agente
    try:
        agente.inicializar_componentes()
        print("✅ Componentes del agente inicializados")
        
        # Inicializar el agent_card
        agente.inicializar_agent_card(host=host, port=port)
        print("✅ Agent card inicializado")
        
        # Procesar PDFs si están disponibles
        archivos_pdf = agente.get_pdf_files()
        if archivos_pdf:
            print(f"📚 Procesando {len(archivos_pdf)} archivos PDF...")
            # Ejecutar procesamiento de PDFs de forma síncrona
            asyncio.run(setup_agent_data(agente, archivos_pdf))
            print("✅ PDFs procesados correctamente")
        else:
            print("⚠️ No se encontraron archivos PDF en PDF_DIR")
            
    except Exception as e:
        logger.error(f"Error inicializando agente: {e}")
        print(f"❌ Error inicializando agente: {e}")
        # Continuar pero con funcionalidad limitada
    
    # Crear task manager y servidor
    task_manager = FisicaTaskManager(agent=agente)
    server = A2AServer(host=host, port=port, agent_card=agente.agent_card, task_manager=task_manager)
    
    print(f"✅ Servidor listo en http://{host}:{port}/")
    server.start()

async def setup_agent_data(agente, archivos_pdf):
    """Configurar datos del agente de forma asíncrona"""
    try:
        # Extraer temario
        temario = agente.procesar_pdfs_temario(archivos_pdf)
        print("✅ Temario extraído")
        
        # Procesar y almacenar PDFs en Qdrant
        await agente.procesar_y_almacenar_pdfs(archivos_pdf)
        print("✅ PDFs almacenados en Qdrant")
        
    except Exception as e:
        logger.error(f"Error configurando datos del agente: {e}")
        raise

if __name__ == "__main__":
    main()