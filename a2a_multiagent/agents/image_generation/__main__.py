# =============================================================================
# 🎯 Purpose:
# Starts the Image Generation Agent as an Agent-to-Agent (A2A) server.
# - Defines the agent's metadata (AgentCard)
# - Wraps the ImageGenerationAgent logic in a ImageGenerationTaskManager
# - Listens for incoming tasks on a configurable host and port
# =============================================================================

import logging
import click
from dotenv import load_dotenv
import os

# Import A2A framework components
from server.server import A2AServer
from models.agent import (
    AgentCard,
    AgentCapabilities,
    AgentSkill
)

# Import our custom components - adjust path based on your structure
from agents.image_generation.task_manager import ImageGenerationTaskManager
from agents.image_generation.agent import ImageGenerationAgent

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# ⚙️ Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingAPIKeyError(Exception):
    """Raised when required API key is missing."""
    pass

# -----------------------------------------------------------------------------
# ✨ CLI Entrypoint
# -----------------------------------------------------------------------------
@click.command()
@click.option(
    "--host",
    default="localhost",
    help="Host to bind Image Generation Agent server to"
)
@click.option(
    "--port",
    default=10002,  # Different port to avoid conflicts
    help="Port for Image Generation Agent server"
)
def main(host: str, port: int):
    """
    Launches the Image Generation Agent A2A server.

    Args:
        host (str): Hostname or IP to bind to (default: localhost)
        port (int): TCP port to listen on (default: 10002)
    """
    try:
        # Check for required API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

        print(f"\n🚀 Starting Image Generation Agent on http://{host}:{port}/\n")

        # -------------------------------------------------------------------------
        # 1) Define the agent's capabilities
        # -------------------------------------------------------------------------
        capabilities = AgentCapabilities(streaming=False)

        # -------------------------------------------------------------------------
        # 2) Define the agent's skill metadata
        # -------------------------------------------------------------------------
        skill = AgentSkill(
            id="image_generator",
            name="Image Generator",
            description=(
                "Generate stunning, high-quality images on demand and leverage "
                "powerful editing capabilities to modify, enhance, or completely "
                "transform visuals using AI-powered image generation."
            ),
            tags=["image", "generation", "art", "visual", "create", "edit"],
            examples=[
                "Generate a photorealistic image of a sunset over mountains",
                "Create an image of a futuristic city",
                "Generate a portrait of a person in watercolor style",
                "Edit the previous image to make it more colorful"
            ]
        )

        # -------------------------------------------------------------------------
        # 3) Compose the AgentCard for discovery
        # -------------------------------------------------------------------------
        agent_card = AgentCard(
            name="ImageGenerationAgent",
            description=(
                "AI-powered agent that generates stunning, high-quality images "
                "from text descriptions and can edit existing images using "
                "advanced machine learning models."
            ),
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["image/png", "image/jpeg"],
            capabilities=capabilities,
            skills=[skill]
        )

        # -------------------------------------------------------------------------
        # 4) Instantiate the core logic and its TaskManager
        # -------------------------------------------------------------------------
        image_agent = ImageGenerationAgent()
        task_manager = ImageGenerationTaskManager(agent=image_agent)

        # -------------------------------------------------------------------------
        # 5) Create and start the A2A server
        # -------------------------------------------------------------------------
        server = A2AServer(
            host=host,
            port=port,
            agent_card=agent_card,
            task_manager=task_manager
        )
        logger.info(f"Starting Image Generation Agent server on {host}:{port}")
        server.start()

    except MissingAPIKeyError as e:
        logger.error(f"Configuration Error: {e}")
        logger.error("Please set your GOOGLE_API_KEY environment variable.")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)


# -----------------------------------------------------------------------------
# Entrypoint guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()