# Desactivar OpenTelemetry y tracing globalmente
import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"

# Asistente de Física Unificado - Versión PC con ADK y RAG
import json
import time
import logging
import torch
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
from models.agent import AgentCard, AgentCapabilities, AgentSkill

# Imports ADK
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types

# Cargar variables de entorno
load_dotenv()
logger = logging.getLogger(__name__)

class AsistenteFisica:
    """Asistente de Física unificado con RAG, memoria semántica y comunicación ADK"""
    
    SUPPORTED_CONTENT_TYPES = ["text/plain", "text"]

    def __init__(self):
        # Configuración básica
        self.name = "asistente_fisica"
        self._setup_apis()
        
        # Estado de inicialización
        self._components_initialized = False
        self._is_running = False
        
        # Inicializar componentes
        self.llm = None
        self.memoria_semantica = None
        self.agents = {}
        self.runner = None
        self.temario = ""
        self.contenido_completo = ""

        # Configuración de embedding
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None

        # Configuración de Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_KEY")
        self.collection_name = "documentos_pdf"

        # Inicializar agent_card y skills
        self.agent_card = None
        self.skills = []

        print("✅ AsistenteFisica inicializado correctamente")

    def _setup_apis(self):
        """Configurar las APIs necesarias"""
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        print("✅ APIs configuradas")

    def inicializar_componentes(self):
        """Inicializar todos los componentes del asistente"""
        if self._components_initialized:
            logger.info("⚠️ Componentes ya inicializados, saltando...")
            return
            
        try:
            self._inicializar_modelos()
            self._inicializar_memoria()
            self._inicializar_adk()
            self._inicializar_modelo_embedding()
            self._components_initialized = True
            print("✅ Todos los componentes inicializados")
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            raise

    def _inicializar_modelos(self):
        """Inicializar los modelos de lenguaje"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
                max_output_tokens=None,
            )
            print("✅ Modelos inicializados")
        except Exception as e:
            logger.error(f"❌ Error inicializando modelos: {e}")
            raise

    def _inicializar_memoria(self):
        """Inicializar la memoria semántica"""
        try:
            self.memoria_semantica = self.SemanticMemory(llm=self.llm)
            print("✅ Memoria semántica inicializada")
        except Exception as e:
            logger.error(f"❌ Error inicializando memoria: {e}")
            raise

    def _inicializar_adk(self):
        """Inicializar componentes ADK"""
        try:
            # Crear agentes ANTES del Runner
            self._crear_agentes()

            # Crear runner principal usando el patrón del segundo script
            self.runner = Runner(
                app_name=self.name,
                agent=self._build_main_agent(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
                artifact_service=InMemoryArtifactService()
            )

            print("✅ Componentes ADK inicializados")
        except Exception as e:
            logger.error(f"❌ Error inicializando ADK: {e}")
            raise

    def _build_main_agent(self):
        """Construir agente principal para comunicación externa"""
        return LlmAgent(
            model="gemini-2.5-flash",
            name=self.name,
            description="Asistente experto en Física que utiliza RAG y memoria semántica",
            instruction=f"""Eres un asistente experto en Física que utiliza un sistema RAG avanzado.
            
IMPORTANTE: Cuando recibas una consulta, debes procesarla através del sistema completo que incluye:
1. Clasificación temática
2. Búsqueda en base de datos vectorial
3. Respuesta contextualizada

TEMARIO DISPONIBLE:
{self.temario}

Tu respuesta debe ser clara, didáctica y basada en la información de los documentos cuando esté disponible.
"""
        )

    def _crear_agentes(self):
        """Crear los agentes ADK especializados"""
        try:
            # Agente Clasificador
            self.classifier_agent = LlmAgent(
                name="clasificador",
                model="gemini-2.5-flash",
                description="Clasifica consultas de física según el temario",
                instruction="""Eres un agente especializado en clasificar consultas de física.
Debes proporcionar tu respuesta en el siguiente formato:
TEMA: [número y título]
SUBTEMAS: [lista]
KEYWORDS: [palabras clave]
"""
            )

            # Agente Buscador
            self.search_agent = LlmAgent(
                name="buscador",
                model="gemini-2.5-flash",
                description="Genera consultas de búsqueda optimizadas",
                instruction="""Eres un agente de búsqueda especializado en física.
Genera la mejor consulta de búsqueda posible para la información solicitada.
Responde SOLAMENTE con la consulta de búsqueda optimizada."""
            )

            # Agente de Respuesta
            self.response_agent = LlmAgent(
                name="respondedor",
                model="gemini-2.5-flash",
                description="Profesor experto que da explicaciones claras",
                instruction="""Eres un profesor experto en física que proporciona explicaciones claras y didácticas.
Usa principalmente los fragmentos de documentos relevantes para construir tu respuesta.
Estructura tu respuesta de manera clara, usando ecuaciones cuando sea apropiado.
IMPORTANTE: Actúa como un profesor experto con pleno conocimiento."""
            )

            self.agents = {
                'classifier': self.classifier_agent,
                'search': self.search_agent,
                'response': self.response_agent
            }
            print("✅ Agentes ADK creados correctamente")
        except Exception as e:
            logger.error(f"❌ Error creando agentes ADK: {e}")
            raise

    def _inicializar_modelo_embedding(self):
        """Inicializar el modelo de embeddings"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(device)
            print("✅ Modelo de embeddings inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando modelo de embeddings: {e}")
            raise

    # Método de comunicación principal - Compatible con el segundo script
    async def invoke(self, query: str, session_id: str = "default_session") -> str:
        """Método de comunicación principal simplificado y robusto"""
        logger.info(f"🔍 Invoke llamado con query: {query[:100]}... session_id: {session_id}")
        
        try:
            # Verificar inicialización
            if not self._components_initialized:
                logger.warning("⚠️ Componentes no inicializados, inicializando...")
                self.inicializar_componentes()
            
            # Si hay problemas con ADK/Runner, usar flujo RAG directo
            if self.runner is None or not self._is_running:
                logger.info("📋 Usando flujo RAG directo...")
                return await self.iniciar_flujo_rag(query, session_id)
            
            # Intentar usar runner ADK
            try:
                logger.info("🚀 Intentando usar runner ADK...")
                
                # Obtener o crear sesión
                session = await self.runner.session_service.get_session(
                    app_name=self.name,
                    user_id="fisica_user",
                    session_id=session_id,
                )
                
                if session is None:
                    logger.info("📝 Creando nueva sesión...")
                    session = await self.runner.session_service.create_session(
                        app_name=self.name,
                        user_id="fisica_user",
                        session_id=session_id,
                        state={},
                    )
                
                # Procesar con RAG primero
                respuesta_rag = await self.iniciar_flujo_rag(query, user_id="fisica_user")
                
                # Crear mensaje para el runner
                message = types.Content(
                    role="user", 
                    parts=[types.Part(text=f"Consulta: {query}\n\nRespuesta procesada: {respuesta_rag}")]
                )
                
                # Ejecutar runner con timeout
                respuesta = None
                event_count = 0
                max_events = 5  # Limitar eventos para evitar bucles infinitos
                
                async for event in self.runner.run_async(
                    user_id="fisica_user",
                    session_id=session_id,
                    new_message=message
                ):
                    event_count += 1
                    if event_count > max_events:
                        logger.warning("⚠️ Máximo de eventos alcanzado, usando respuesta RAG")
                        break
                        
                    if event and event.content and event.content.parts:
                        texto = "\n".join([p.text for p in event.content.parts if p.text])
                        if texto:
                            respuesta = texto
                            break
                
                return respuesta if respuesta else respuesta_rag
                
            except Exception as runner_error:
                logger.warning(f"⚠️ Error con runner ADK, usando RAG directo: {runner_error}")
                return await self.iniciar_flujo_rag(query, session_id)
                
        except Exception as e:
            logger.error(f"❌ Error en invoke: {e}")
            
            # Fallback final: respuesta simple con LLM
            try:
                if self.llm:
                    messages = [
                        SystemMessage(content="Eres un experto en física. Responde de manera clara y didáctica."),
                        HumanMessage(content=query)
                    ]
                    response = self.llm(messages)
                    return response.content
                else:
                    return f"Error: No se pudo procesar la consulta. {str(e)}"
            except Exception as fallback_error:
                logger.error(f"❌ Error en fallback final: {fallback_error}")
                return f"Error crítico procesando la consulta: {str(e)}"

    # Métodos RAG simplificados
    def leer_pdf(self, nombre_archivo):
        """Leer contenido de un archivo PDF"""
        try:
            reader = PdfReader(nombre_archivo)
            return "".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            logger.error(f"Error al leer {nombre_archivo}: {e}")
            return ""

    def get_pdf_files(self):
        """Obtiene la lista de archivos PDF desde el directorio especificado en la variable de entorno PDF_DIR"""
        pdf_dir = os.getenv("PDF_DIR")
        if not pdf_dir or not os.path.isdir(pdf_dir):
            logger.warning(f"⚠️ El directorio PDF_DIR no está definido o no existe: {pdf_dir}")
            return []
        return [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    def procesar_pdfs_temario(self, archivos_pdf=None):
        """Procesar PDFs para extraer el temario"""
        try:
            if archivos_pdf is None:
                archivos_pdf = self.get_pdf_files()
            
            contenido_completo = ""
            for archivo in archivos_pdf:
                if os.path.exists(archivo):
                    contenido_completo += f"\n--- Contenido de {archivo} ---\n"
                    contenido_completo += self.leer_pdf(archivo)

            self.contenido_completo = contenido_completo

            if not contenido_completo.strip():
                logger.warning("⚠️ No se pudo extraer contenido de los PDFs")
                self.temario = "Temario no disponible - PDFs no procesados"
                return self.temario

            # Extraer temario usando LLM
            system_message = f"""
Eres un experto profesor Física I de la Universidad de Buenos Aires.
Utiliza el siguiente contenido como referencia:
---
{self.contenido_completo}
---
"""
            user_question = "Sobre que contenidos podes contestarme"
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_question),
            ]
            
            ai_msg = self.llm.invoke(messages)
            self.temario = ai_msg.content

            # Actualizar el agente principal con el temario
            if hasattr(self, 'runner') and self.runner and self.runner.agent:
                self.runner.agent.instruction = f"""Eres un asistente experto en Física que utiliza un sistema RAG avanzado.

TEMARIO DISPONIBLE:
{self.temario}

Responde consultas de física de manera clara y didáctica, utilizando la información de los documentos.
"""

            print("✅ Temario extraído correctamente")
            return self.temario
            
        except Exception as e:
            logger.error(f"❌ Error procesando PDFs para temario: {e}")
            self.temario = f"Error procesando temario: {str(e)}"
            return self.temario

    def split_into_chunks(self, text, chunk_size=2000):
        """Dividir texto en chunks"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_embeddings(self, chunks, batch_size=32):
        """Generar embeddings para los chunks"""
        try:
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error generando embeddings: {e}")
            return []

    async def store_in_qdrant(self, points):
        """Almacenar puntos en Qdrant"""
        try:
            client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

            try:
                await client.get_collection(self.collection_name)
            except Exception:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
                )
                print(f"Colección '{self.collection_name}' creada")

            await client.upsert(collection_name=self.collection_name, points=points, wait=True)
            print(f"{len(points)} chunks almacenados en Qdrant")
        except Exception as e:
            logger.error(f"❌ Error almacenando en Qdrant: {e}")
            raise

    async def procesar_y_almacenar_pdfs(self, pdf_files):
        """Procesar PDFs y almacenar en Qdrant"""
        try:
            all_chunks = []
            pdf_metadata = []
            global_id_counter = 0

            for pdf_file in pdf_files:
                if not os.path.exists(pdf_file):
                    logger.warning(f"⚠️ {pdf_file} no encontrado")
                    continue

                text = self.leer_pdf(pdf_file)
                if text:
                    chunks = self.split_into_chunks(text)

                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        pdf_metadata.append({
                            "pdf_name": pdf_file,
                            "chunk_id": i,
                            "global_id": global_id_counter
                        })
                        global_id_counter += 1

            if not all_chunks:
                logger.warning("⚠️ No se encontraron chunks para procesar")
                return

            embeddings = self.generate_embeddings(all_chunks)
            if not embeddings:
                logger.error("❌ No se pudieron generar embeddings")
                return

            points = [
                PointStruct(
                    id=meta["global_id"],
                    vector=embedding.tolist(),
                    payload={
                        "pdf_name": meta["pdf_name"],
                        "chunk_id": meta["chunk_id"],
                        "text": all_chunks[idx]
                    }
                )
                for idx, (meta, embedding) in enumerate(zip(pdf_metadata, embeddings))
            ]

            await self.store_in_qdrant(points)

            metadata_dict = {
                p.id: {
                    "pdf": p.payload["pdf_name"],
                    "chunk": p.payload["text"]
                } for p in points
            }

            with open("pdf_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
            print("✅ Metadatos guardados en 'pdf_metadata.json'")
            
        except Exception as e:
            logger.error(f"❌ Error procesando y almacenando PDFs: {e}")
            raise

    async def search_documents(self, query, top_k=5):
        """Realizar búsqueda en Qdrant"""
        try:
            client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

            try:
                await client.get_collection(self.collection_name)
                logger.info("✅ Conexión a Qdrant exitosa")
            except Exception as e:
                logger.warning(f"⚠️ Error al conectar con Qdrant: {str(e)}")
                return []

            inputs = self.tokenizer(
                [query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            query_embedding = query_embedding.flatten()

            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )

            formatted_results = []
            metadata = {}

            if os.path.exists("pdf_metadata.json"):
                with open("pdf_metadata.json", "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            for result in results:
                meta = metadata.get(str(result.id), {})
                payload = result.payload or {}

                formatted_results.append({
                    "pdf": meta.get("pdf", payload.get("pdf_name", "N/A")),
                    "texto": meta.get("chunk", payload.get("text", "Texto no disponible")),
                    "similitud": round(result.score, 4)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error en la búsqueda: {str(e)}")
            return []

    async def _get_agent_response_simple(self, instruction: str, query: str) -> str:
        """Método simplificado para obtener respuestas de agentes usando LLM directo"""
        try:
            messages = [
                SystemMessage(content=instruction),
                HumanMessage(content=query)
            ]
            response = self.llm(messages)
            return response.content
        except Exception as e:
            logger.error(f"❌ Error obteniendo respuesta simple: {e}")
            return f"Error: {str(e)}"

    async def iniciar_flujo_rag(self, consulta_usuario: str, user_id: str = "default_user"):
        """Flujo simplificado de procesamiento RAG"""
        logger.info(f"📝 Iniciando flujo RAG para: {consulta_usuario[:100]}...")
        
        try:
            # Obtener contexto de memoria
            contexto_memoria = ""
            if self.memoria_semantica:
                contexto_memoria = self.memoria_semantica.get_context()

            # Paso 1: Clasificación simplificada
            clasificacion_instruction = f"""Eres un agente especializado en clasificar consultas de física.
TEMARIO DISPONIBLE: {self.temario}
Clasifica la siguiente consulta según el temario."""
            
            clasificacion = await self._get_agent_response_simple(
                clasificacion_instruction, 
                f"Consulta: {consulta_usuario}\nContexto: {contexto_memoria}"
            )

            # Paso 2: Generar consulta de búsqueda
            search_instruction = """Eres un agente de búsqueda especializado en física.
Genera la mejor consulta de búsqueda posible para la información solicitada.
Responde SOLAMENTE con la consulta de búsqueda optimizada."""
            
            consulta_busqueda = await self._get_agent_response_simple(
                search_instruction,
                f"Clasificación: {clasificacion}\nConsulta original: {consulta_usuario}"
            )

            # Paso 3: Búsqueda en Qdrant
            resultados_busqueda = await self.search_documents(consulta_busqueda.strip())

            # Paso 4: Generar respuesta final
            contexto_busqueda = ""
            if resultados_busqueda:
                contexto_busqueda = "\n".join([
                    f"--- Fragmento {i} (PDF: {res['pdf']}) ---\n{res['texto']}"
                    for i, res in enumerate(resultados_busqueda, 1)
                ])

            response_instruction = """Eres un profesor experto en física que proporciona explicaciones claras y didácticas.
Usa principalmente los fragmentos de documentos relevantes para construir tu respuesta.
Estructura tu respuesta de manera clara, usando ecuaciones cuando sea apropiado."""

            response_query = f"""CONSULTA ORIGINAL: {consulta_usuario}

CONTEXTO DE CONVERSACIÓN: {contexto_memoria}

CLASIFICACIÓN: {clasificacion}

FRAGMENTOS DE DOCUMENTOS RELEVANTES:
{contexto_busqueda}

Proporciona una respuesta completa y didáctica."""

            respuesta_final = await self._get_agent_response_simple(response_instruction, response_query)
            
            # Actualizar memoria si está disponible
            if self.memoria_semantica:
                try:
                    self.memoria_semantica.add_interaction(consulta_usuario, respuesta_final)
                except Exception as e:
                    logger.warning(f"⚠️ Error actualizando memoria: {e}")
            
            logger.info("✅ Flujo RAG completado exitosamente")
            return respuesta_final

        except Exception as e:
            logger.error(f"❌ Error en flujo RAG: {e}")
            # Fallback: respuesta simple sin RAG
            try:
                fallback_instruction = f"""Eres un experto en física. 
TEMARIO DISPONIBLE: {self.temario}
Responde de manera clara y didáctica."""
                
                return await self._get_agent_response_simple(fallback_instruction, consulta_usuario)
            except Exception as fallback_error:
                logger.error(f"❌ Error en fallback RAG: {fallback_error}")
                return f"Error procesando la consulta: {str(e)}"

    def inicializar_agent_card(self, host="localhost", port=10002):
        """Inicializa el AgentCard y los skills del asistente"""
        try:
            capabilities = AgentCapabilities(streaming=False)
            skill = AgentSkill(
                id="physics_rag",
                name="Asistente de Física I",
                description="Responde consultas de Física utilizando recuperación aumentada de información",
                tags=["fisica", "rag", "pdf"],
                examples=["Explicá el principio de conservación de la energía", "¿Qué es el centro de masa?"]
            )
            self.skills = [skill]
            self.agent_card = AgentCard(
                name="AsistenteFisica",
                description="Agente que responde preguntas de física usando RAG",
                url=f"http://{host}:{port}/",
                version="1.0.0",
                defaultInputModes=["text"],
                defaultOutputModes=["text"],
                capabilities=capabilities,
                skills=self.skills
            )
            logger.info("✅ AgentCard inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando AgentCard: {e}")
            raise

    # Clase interna para memoria semántica
    class SemanticMemory:
        def __init__(self, llm, max_entries=10):
            self.conversations = []
            self.max_entries = max_entries
            self.summary = ""
            self.direct_history = ""
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=2000,
                return_messages=True
            )

        def add_interaction(self, query, response):
            """Añadir interacción a la memoria"""
            try:
                self.memory.save_context({"input": query}, {"output": response})
                self.conversations.append({"query": query, "response": response})
                if len(self.conversations) > self.max_entries:
                    self.conversations.pop(0)

                self.direct_history += f"\nUsuario: {query}\nAsistente: {response}\n"
                if len(self.conversations) > 3:
                    recent = self.conversations[-3:]
                    self.direct_history = ""
                    for conv in recent:
                        self.direct_history += f"\nUsuario: {conv['query']}\nAsistente: {conv['response']}\n"

                self.update_summary()
            except Exception as e:
                logger.error(f"Error añadiendo interacción a memoria: {e}")

        def update_summary(self):
            """Actualizar resumen de la conversación"""
            try:
                memory_variables = self.memory.load_memory_variables({})
                history_messages = memory_variables.get("history", [])
                langchain_summary = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in history_messages])

                if langchain_summary:
                    self.summary = f"Resumen de conversación previa: {langchain_summary}\n\nInteracciones recientes:{self.direct_history}"
                else:
                    self.summary = f"Interacciones recientes:{self.direct_history}"
            except Exception as e:
                logger.error(f"Error actualizando resumen: {e}")
                self.summary = f"Interacciones recientes:{self.direct_history}"

        def get_context(self):
            """Obtener contexto actual de la conversación"""
            return self.summary if self.summary.strip() else "No hay conversación previa."