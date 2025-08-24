"""
Agente de Generación de Imágenes usando CrewAI - Adaptado para protocolo A2A
Optimizado para integración con el sistema multiagente
"""
import asyncio
import base64
import os
import re
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Imports para A2A
from models.agent import AgentCard, AgentCapabilities, AgentSkill

# Imports para CrewAI
from crewai import Agent, Crew, LLM, Task
from crewai.process import Process
from crewai.tools import tool

# Imports para Gemini
from google import genai
from google.genai import types

# Imports adicionales
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


class ImageData(BaseModel):
    """Representa los datos de una imagen.

    Attributes:
        id: Identificador único de la imagen.
        name: Nombre de la imagen.
        mime_type: Tipo MIME de la imagen.
        bytes: Datos de imagen codificados en base64.
        error: Mensaje de error si hubo algún problema.
    """
    id: Optional[str] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
    bytes: Optional[str] = None
    error: Optional[str] = None


class InMemoryCache:
    """Cache simple en memoria para las sesiones de imágenes."""

    def __init__(self):
        self._cache = {}

    def get(self, session_id: str) -> Optional[Dict]:
        """Obtener datos de sesión."""
        return self._cache.get(session_id)

    def set(self, session_id: str, data: Dict):
        """Establecer datos de sesión."""
        self._cache[session_id] = data

    def add_image(self, session_id: str, image_data: ImageData):
        """Añadir imagen a la sesión."""
        session_data = self.get(session_id)
        if session_data is None:
            self.set(session_id, {image_data.id: image_data})
        else:
            session_data[image_data.id] = image_data

    def list_sessions(self) -> List[str]:
        """Listar todas las sesiones."""
        return list(self._cache.keys())

    def get_session_stats(self, session_id: str) -> Dict:
        """Obtener estadísticas de una sesión."""
        session_data = self.get(session_id)
        if not session_data:
            return {"total_images": 0, "image_ids": []}

        return {
            "total_images": len(session_data),
            "image_ids": list(session_data.keys()),
            "session_id": session_id
        }


class ImageGeneratorAgent:
    """Agente de generación de imágenes integrado con protocolo A2A"""
    
    SUPPORTED_CONTENT_TYPES = ["text/plain", "text"]

    def __init__(self):
        # Configuración básica
        self.name = "image_generator"
        self._setup_apis()
        
        # Estado de inicialización
        self._components_initialized = False
        
        # Cache global para imágenes
        self.image_cache = InMemoryCache()
        
        # Componentes CrewAI
        self.model = None
        self.image_creator_agent = None
        self.prompt_enhancer_agent = None
        
        # Configuración A2A
        self.agent_card = None
        self.skills = []
        
        print("✅ ImageGeneratorAgent inicializado correctamente")

    def _setup_apis(self):
        """Configurar las APIs necesarias"""
        #Obtener API key desde variables de entorno
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY no encontrada")
            raise ValueError(
                "GOOGLE_API_KEY no está configurada. "
                "Por favor verifica que esté en tu archivo .env o como variable de entorno del sistema."
            )
        
        # Establecer la variable de entorno (por si acaso)
        os.environ["GOOGLE_API_KEY"] = api_key
        logger.info("✅ API key cargada correctamente desde .env")
        print("✅ APIs configuradas")

    def inicializar_componentes(self):
        """Inicializar todos los componentes del agente"""
        if self._components_initialized:
            logger.info("⚠️ Componentes ya inicializados, saltando...")
            return
            
        try:
            self._inicializar_crewai()
            self._components_initialized = True
            print("✅ Todos los componentes inicializados")
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            raise

    def _inicializar_crewai(self):
        """Inicializar componentes de CrewAI"""
        try:
            # Configurar LLM para CrewAI
            api_key = os.getenv("GOOGLE_API_KEY")
            self.model = LLM(model="gemini/gemini-2.0-flash", api_key=api_key)

            # Agente especializado en creación de imágenes
            self.image_creator_agent = Agent(
                role="Experto en Creación de Imágenes",
                goal=(
                    "Generar imágenes de alta calidad basadas en los prompts del usuario. "
                    "Si el prompt es vago, interpretar creativamente y generar la mejor "
                    "imagen posible. Enfocarme en la precisión y creatividad."
                ),
                backstory=(
                    "Eres un artista digital especializado en transformar descripciones "
                    "textuales en representaciones visuales impresionantes. Tienes "
                    "experiencia en múltiples estilos artísticos y entiendes cómo "
                    "optimizar prompts para obtener los mejores resultados."
                ),
                verbose=True,
                allow_delegation=False,
                tools=[self._create_image_generation_tool()],
                llm=self.model,
            )

            # Agente de mejora de prompts
            self.prompt_enhancer_agent = Agent(
                role="Especialista en Prompts",
                goal=(
                    "Mejorar y optimizar prompts de usuario para generar imágenes "
                    "más detalladas y de mayor calidad. Añadir detalles artísticos "
                    "y técnicos relevantes."
                ),
                backstory=(
                    "Eres un experto en ingeniería de prompts para IA generativa. "
                    "Conoces las mejores prácticas para describir imágenes y sabes "
                    "qué palabras clave y frases producen los mejores resultados "
                    "en modelos de generación de imágenes."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.model,
            )
            
            print("✅ Componentes CrewAI inicializados")
        except Exception as e:
            logger.error(f"❌ Error inicializando CrewAI: {e}")
            raise

    def _create_image_generation_tool(self):
        """Crear herramienta de generación de imágenes"""
        @tool("ImageGenerationTool")
        def generate_image_tool(prompt: str, session_id: str = "default", artifact_file_id: Optional[str] = None) -> str:
            """Herramienta de generación de imágenes que genera o modifica imágenes basadas en un prompt."""
            return self._generate_image_sync(prompt, session_id, artifact_file_id)
        
        return generate_image_tool

    def _generate_image_sync(self, prompt: str, session_id: str = "default", artifact_file_id: Optional[str] = None) -> str:
        """Versión síncrona de generación de imágenes para CrewAI tools"""
        if not prompt:
            raise ValueError("El prompt no puede estar vacío")

        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)

            # Preparar entrada de texto
            text_input = (
                prompt,
                "Ignore any input images if they do not match the request.",
            )

            ref_image = None
            logger.info(f"Session ID: {session_id}")
            print(f"🎨 Generando imagen para sesión: {session_id}")

            # Obtener imagen de referencia del cache si existe
            try:
                ref_image_data = None
                session_image_data = self.image_cache.get(session_id)

                if session_image_data and artifact_file_id:
                    try:
                        ref_image_data = session_image_data[artifact_file_id]
                        logger.info(f"Encontrada imagen de referencia con ID: {artifact_file_id}")
                        print(f"🔄 Usando imagen de referencia: {artifact_file_id}")
                    except Exception as e:
                        ref_image_data = None

                elif session_image_data:
                    # Usar la última imagen generada como referencia
                    latest_image_key = list(session_image_data.keys())[-1]
                    ref_image_data = session_image_data[latest_image_key]
                    print(f"🔄 Usando última imagen como referencia: {latest_image_key}")

                if ref_image_data:
                    ref_bytes = base64.b64decode(ref_image_data.bytes)
                    ref_image = Image.open(BytesIO(ref_bytes))

            except Exception as e:
                ref_image = None
                print(f"⚠️ No se pudo cargar imagen de referencia: {e}")

            # Preparar contenido para la API
            if ref_image:
                contents = [text_input, ref_image]
                print("🖼️ Generando con imagen de referencia")
            else:
                contents = text_input
                print("🆕 Generando nueva imagen")

            # Llamada a la API con el modelo correcto
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
            )

            print("✅ Respuesta recibida de Gemini")

            # Procesar respuesta
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    try:
                        # Crear objeto de datos de imagen
                        data = ImageData(
                            bytes=base64.b64encode(part.inline_data.data).decode("utf-8"),
                            mime_type=part.inline_data.mime_type,
                            name=f"generated_image_{uuid4().hex[:8]}.png",
                            id=uuid4().hex,
                        )

                        # Guardar en cache
                        self.image_cache.add_image(session_id, data)

                        print(f"✅ Imagen generada con ID: {data.id}")
                        return data.id

                    except Exception as e:
                        logger.error(f"Error procesando imagen: {e}")
                        print(f"❌ Error procesando imagen: {e}")

            return "ERROR: No se encontró imagen en la respuesta"

        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            print(f"❌ Error en generación: {e}")
            return "ERROR: No se pudo generar la imagen"

    def create_tasks(self, user_prompt: str, session_id: str = "default",
                    artifact_file_id: Optional[str] = None, enhance_prompt: bool = True):
        """Crear tareas para la generación de imágenes."""
        tasks = []

        if enhance_prompt:
            # Tarea de mejora de prompt
            prompt_enhancement_task = Task(
                description=(
                    f"Mejora este prompt para generación de imágenes: '{user_prompt}'\n\n"
                    "Añade detalles artísticos como:\n"
                    "- Estilo visual específico\n"
                    "- Calidad y detalles técnicos\n"
                    "- Composición y encuadre\n"
                    "- Iluminación y colores\n"
                    "- Atmósfera y mood\n\n"
                    "Devuelve solo el prompt mejorado, sin explicaciones adicionales."
                ),
                expected_output="Un prompt detallado y optimizado para generación de imágenes",
                agent=self.prompt_enhancer_agent,
            )
            tasks.append(prompt_enhancement_task)

        # Tarea de generación de imagen
        image_creation_task = Task(
            description=(
                f"Usa el prompt {'mejorado de la tarea anterior' if enhance_prompt else f'proporcionado: {user_prompt}'} "
                f"para generar una imagen de alta calidad.\n\n"
                f"Detalles de la sesión:\n"
                f"- Session ID: {session_id}\n"
                f"- Artifact File ID: {artifact_file_id or 'Ninguno'}\n\n"
                f"Usa la herramienta ImageGenerationTool con estos parámetros.\n"
                f"Si hay un artifact_file_id, la herramienta usará esa imagen como referencia."
            ),
            expected_output="El ID de la imagen generada",
            agent=self.image_creator_agent,
            context=tasks if enhance_prompt else None,
        )
        tasks.append(image_creation_task)

        return tasks

    def generate_image(self, user_prompt: str, session_id: str = "default",
                      artifact_file_id: Optional[str] = None, enhance_prompt: bool = True) -> str:
        """Generar imagen usando el crew de agentes."""
        print(f"🚀 Iniciando generación con CrewAI...")
        print(f"📝 Prompt: {user_prompt}")
        print(f"🔧 Mejora de prompt: {'Activada' if enhance_prompt else 'Desactivada'}")

        # Crear tareas
        tasks = self.create_tasks(user_prompt, session_id, artifact_file_id, enhance_prompt)

        # Crear y ejecutar crew
        image_crew = Crew(
            agents=[self.prompt_enhancer_agent, self.image_creator_agent] if enhance_prompt else [self.image_creator_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = image_crew.kickoff()
            print(f"✅ Crew completado: {result}")
            return str(result)
        except Exception as e:
            print(f"❌ Error en crew: {e}")
            return f"ERROR: {e}"

    def extract_artifact_file_id(self, query: str) -> Optional[str]:
        """Extraer ID de archivo artifact de una consulta."""
        try:
            pattern = r'(?:id|artifact-file-id)\s+([0-9a-f]{32})'
            match = re.search(pattern, query, re.IGNORECASE)
            return match.group(1) if match else None
        except Exception:
            return None

    def extract_image_id_from_result(self, result: str) -> Optional[str]:
        """Extraer ID de imagen del resultado del crew."""
        # Buscar patrones comunes de ID
        patterns = [
            r'ID:\s*([a-f0-9]{32})',
            r'id:\s*([a-f0-9]{32})',
            r'imagen generada con ID:\s*([a-f0-9]{32})',
            r'([a-f0-9]{32})'
        ]

        for pattern in patterns:
            match = re.search(pattern, str(result), re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    # Método de comunicación principal - Compatible con protocolo A2A
    async def invoke(self, query: str, session_id: str = "default_session") -> str:
        """Método de comunicación principal para protocolo A2A"""
        logger.info(f"🔍 ImageGenerator invoke llamado con query: {query[:100]}... session_id: {session_id}")
        
        try:
            # Verificar inicialización
            if not self._components_initialized:
                logger.warning("⚠️ Componentes no inicializados, inicializando...")
                self.inicializar_componentes()
            
            # Extraer artifact file ID si existe
            artifact_file_id = self.extract_artifact_file_id(query)
            
            # Determinar si se debe mejorar el prompt
            enhance_prompt = not query.lower().startswith("no mejores")
            if not enhance_prompt:
                # Remover la instrucción del prompt
                query = query.replace("no mejores", "").strip()
            
            # Generar imagen
            logger.info(f"🎨 Generando imagen para: {query[:100]}...")
            result = self.generate_image(
                user_prompt=query,
                session_id=session_id,
                artifact_file_id=artifact_file_id,
                enhance_prompt=enhance_prompt
            )
            
            # Extraer ID de imagen del resultado
            image_id = self.extract_image_id_from_result(result)
            
            if image_id and not result.startswith("ERROR"):
                # Obtener estadísticas de la sesión
                stats = self.image_cache.get_session_stats(session_id)
                
                response = (
                    f"✅ Imagen generada exitosamente!\n\n"
                    f"🆔 ID de la imagen: {image_id}\n"
                    f"📊 Imágenes en la sesión '{session_id}': {stats['total_images']}\n\n"
                    f"💡 Para usar esta imagen como referencia en futuras generaciones, "
                    f"incluye 'artifact-file-id {image_id}' en tu próximo prompt.\n\n"
                    f"🎨 Prompt procesado: {query}"
                )
                
                # Agregar información de mejora de prompt
                if enhance_prompt:
                    response += "\n\n🔧 Se aplicó mejora automática al prompt para obtener mejores resultados."
                
                return response
            else:
                return f"❌ Error generando la imagen: {result}"
                
        except Exception as e:
            logger.error(f"❌ Error en invoke: {e}")
            return f"❌ Error procesando la solicitud: {str(e)}"

    def inicializar_agent_card(self, host="localhost", port=10004):
        """Inicializa el AgentCard y los skills del agente"""
        try:
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
            self.skills = [skill]
            self.agent_card = AgentCard(
                name="ImageGeneratorAgent",
                description="Agente especializado en generación de imágenes usando CrewAI y Gemini",
                url=f"http://{host}:{port}/",
                version="1.0.0",
                defaultInputModes=["text"],
                defaultOutputModes=["text", "image"],
                capabilities=capabilities,
                skills=self.skills
            )
            logger.info("✅ AgentCard inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando AgentCard: {e}")
            raise

    # Métodos de utilidad para manejo de imágenes
    def get_image_data(self, image_id: str, session_id: str = "default") -> Optional[ImageData]:
        """Obtener datos de una imagen por ID"""
        try:
            session_data = self.image_cache.get(session_id)
            if session_data and image_id in session_data:
                return session_data[image_id]
            return None
        except Exception as e:
            logger.error(f"Error obteniendo imagen {image_id}: {e}")
            return None

    def list_images(self, session_id: str = "default") -> Dict:
        """Listar todas las imágenes de una sesión."""
        stats = self.image_cache.get_session_stats(session_id)
        session_data = self.image_cache.get(session_id)

        if not session_data:
            return {"total_images": 0, "images": []}

        images = []
        for img_id, img_data in session_data.items():
            images.append({
                "id": img_id,
                "name": img_data.name,
                "mime_type": img_data.mime_type
            })

        return {"total_images": len(images), "images": images}