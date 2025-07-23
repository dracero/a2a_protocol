# Asistente de Física Unificado - Versión PC con ADK y RAG
import os
import json
import time
import asyncio
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

        print("✅ AsistenteFisica inicializado correctamente")

    def _setup_apis(self):
        """Configurar las APIs necesarias"""
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        print("✅ APIs configuradas")

    def inicializar_componentes(self):
        """Inicializar todos los componentes del asistente"""
        self._inicializar_modelos()
        self._inicializar_memoria()
        self._inicializar_adk()
        self._inicializar_modelo_embedding()
        print("✅ Todos los componentes inicializados")

    def _inicializar_modelos(self):
        """Inicializar los modelos de lenguaje"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=None,
        )
        print("✅ Modelos inicializados")

    def _inicializar_memoria(self):
        """Inicializar la memoria semántica"""
        self.memoria_semantica = self.SemanticMemory(llm=self.llm)
        print("✅ Memoria semántica inicializada")

    def _inicializar_adk(self):
        """Inicializar componentes ADK"""
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

    def _inicializar_modelo_embedding(self):
        """Inicializar el modelo de embeddings"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        print("✅ Modelo de embeddings inicializado")

    # Método de comunicación principal - Compatible con el segundo script
    async def invoke(self, query: str, session_id: str = "default_session") -> str:
        """Método principal de comunicación compatible con el patrón del segundo script"""
        try:
            # Obtener o crear sesión
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

            # Procesar consulta através del sistema RAG completo
            respuesta_rag = await self.iniciar_flujo_rag(query, user_id="fisica_user")
            
            # Enviar respuesta através del sistema ADK
            message = types.Content(role="user", parts=[types.Part(text=f"Consulta: {query}\n\nRespuesta procesada: {respuesta_rag}")])
            
            last_event = None
            async for event in self.runner.run_async(
                user_id="fisica_user",
                session_id=session_id,
                new_message=message
            ):
                last_event = event

            if last_event and last_event.content and last_event.content.parts:
                return "\n".join([p.text for p in last_event.content.parts if p.text])
            
            return respuesta_rag

        except Exception as e:
            logger.error(f"Error en invoke: {e}")
            return f"Error al procesar la consulta: {str(e)}"

    # Métodos RAG del primer script
    def leer_pdf(self, nombre_archivo):
        """Leer contenido de un archivo PDF"""
        try:
            reader = PdfReader(nombre_archivo)
            return "".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error al leer {nombre_archivo}: {e}")
            return ""

    def get_pdf_files(self):
        """Obtiene la lista de archivos PDF desde el directorio especificado en la variable de entorno PDF_DIR"""
        pdf_dir = os.getenv("PDF_DIR")
        if not pdf_dir or not os.path.isdir(pdf_dir):
            print(f"⚠️ El directorio PDF_DIR no está definido o no existe: {pdf_dir}")
            return []
        return [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    def procesar_pdfs_temario(self, archivos_pdf=None):
        """Procesar PDFs para extraer el temario. Si archivos_pdf es None, usa los PDFs del directorio PDF_DIR"""
        if archivos_pdf is None:
            archivos_pdf = self.get_pdf_files()
        contenido_completo = ""

        for archivo in archivos_pdf:
            if os.path.exists(archivo):
                contenido_completo += f"\n--- Contenido de {archivo} ---\n"
                contenido_completo += self.leer_pdf(archivo)

        self.contenido_completo = contenido_completo

        # Extraer temario
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
        if hasattr(self, 'runner') and self.runner.agent:
            self.runner.agent.instruction = f"""Eres un asistente experto en Física que utiliza un sistema RAG avanzado.

TEMARIO DISPONIBLE:
{self.temario}

Responde consultas de física de manera clara y didáctica, utilizando la información de los documentos.
"""

        print("✅ Temario extraído correctamente")
        return self.temario

    def split_into_chunks(self, text, chunk_size=2000):
        """Dividir texto en chunks"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_embeddings(self, chunks, batch_size=32):
        """Generar embeddings para los chunks"""
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

    async def store_in_qdrant(self, points):
        """Almacenar puntos en Qdrant"""
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

    async def procesar_y_almacenar_pdfs(self, pdf_files):
        """Procesar PDFs y almacenar en Qdrant"""
        all_chunks = []
        pdf_metadata = []
        global_id_counter = 0

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
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
            print("⚠️ No se encontraron chunks para procesar")
            return

        embeddings = self.generate_embeddings(all_chunks)

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

    async def search_documents(self, query, top_k=5):
        """Realizar búsqueda en Qdrant"""
        try:
            client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

            try:
                await client.get_collection(self.collection_name)
                print("✅ Conexión a Qdrant exitosa")
            except Exception as e:
                print(f"❌ Error al conectar con Qdrant: {str(e)}")
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
            print(f"❌ Error en la búsqueda: {str(e)}")
            return []

    async def _get_agent_response(self, agent, input_data, session_id="default_session"):
        """Función auxiliar para obtener respuesta de un agente ADK"""
        try:
            if isinstance(input_data, dict):
                prompt = self._format_prompt_for_agent(agent.name, input_data)
            else:
                prompt = str(input_data)

            # Crear runner temporal para este agente
            temp_runner = Runner(
                app_name=f"temp_{agent.name}",
                agent=agent,
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
                artifact_service=InMemoryArtifactService()
            )

            session = await temp_runner.session_service.get_session(
                app_name=f"temp_{agent.name}",
                user_id="fisica_user",
                session_id=session_id,
            )
            if session is None:
                session = await temp_runner.session_service.create_session(
                    app_name=f"temp_{agent.name}",
                    user_id="fisica_user",
                    session_id=session_id,
                    state={},
                )

            message = types.Content(role="user", parts=[types.Part(text=prompt)])
            
            last_event = None
            async for event in temp_runner.run_async(
                user_id="fisica_user",
                session_id=session_id,
                new_message=message
            ):
                last_event = event

            if last_event and last_event.content and last_event.content.parts:
                return "\n".join([p.text for p in last_event.content.parts if p.text])
            
            return "No se pudo obtener respuesta del agente"

        except Exception as e:
            print(f"Error ejecutando agente {agent.name}: {e}")
            # Fallback usando LLM directo
            try:
                messages = [
                    SystemMessage(content=getattr(agent, 'instruction', 'You are a helpful AI assistant.')),
                    HumanMessage(content=prompt)
                ]
                response = self.llm(messages)
                return response.content
            except Exception as fallback_error:
                print(f"Error en fallback para agente {agent.name}: {fallback_error}")
                return "Error al procesar la consulta"

    def _format_prompt_for_agent(self, agent_name, data):
        """Formatear el prompt según el agente específico"""
        if agent_name == "clasificador":
            return f"""
TEMARIO DE FÍSICA:
{data.get('temario', '')}

CONTEXTO DE CONVERSACIÓN PREVIA:
{data.get('contexto_memoria', '')}

CONSULTA DEL USUARIO:
{data.get('consulta_usuario', '')}

Clasifica esta consulta según el temario proporcionado.
"""
        elif agent_name == "buscador":
            return f"""
CLASIFICACIÓN:
{data.get('clasificacion', '')}

CONSULTA ORIGINAL:
{data.get('consulta_original', '')}

Genera la mejor consulta de búsqueda para esta información.
"""
        elif agent_name == "respondedor":
            return f"""
**CONSULTA ORIGINAL DEL USUARIO:**
{data.get('consulta_usuario', '')}

**CONTEXTO DE CONVERSACIÓN ANTERIOR:**
{data.get('contexto_memoria', '')}

**CLASIFICACIÓN TEMÁTICA:**
{data.get('clasificacion', '')}

**FRAGMENTOS DE DOCUMENTOS RELEVANTES:**
{data.get('contexto_documentos', '')}

Proporciona una respuesta completa y didáctica.
"""
        return str(data)

    async def iniciar_flujo_rag(self, consulta_usuario: str, user_id: str = "default_user"):
        """Flujo completo de procesamiento RAG"""
        print(f"📝 Consulta recibida: {consulta_usuario}")
        contexto_memoria = self.memoria_semantica.get_context()

        try:
            # Paso 1: Clasificación
            clasificacion_data = {
                "consulta_usuario": consulta_usuario,
                "contexto_memoria": contexto_memoria,
                "temario": self.temario
            }
            clasificacion = await self._get_agent_response(self.classifier_agent, clasificacion_data)

            # Paso 2: Generar consulta de búsqueda
            search_data = {
                "clasificacion": clasificacion,
                "consulta_original": consulta_usuario,
                "contexto_conversacion": contexto_memoria
            }
            consulta_busqueda = await self._get_agent_response(self.search_agent, search_data)

            # Paso 3: Búsqueda en Qdrant
            resultados_busqueda = await self.search_documents(consulta_busqueda)

            # Paso 4: Generar respuesta final
            contexto_busqueda = "\n".join([
                f"--- Fragmento {i} (PDF: {res['pdf']}) ---\n{res['texto']}"
                for i, res in enumerate(resultados_busqueda, 1)
            ])

            response_data = {
                "consulta_usuario": consulta_usuario,
                "contexto_memoria": contexto_memoria,
                "clasificacion": clasificacion,
                "contexto_documentos": contexto_busqueda
            }

            respuesta_final = await self._get_agent_response(self.response_agent, response_data)
            
            # Actualizar memoria
            self.memoria_semantica.add_interaction(consulta_usuario, respuesta_final)
            
            return respuesta_final

        except Exception as e:
            print(f"❌ Error en flujo RAG: {e}")
            return f"Error al procesar la consulta: {str(e)}"

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
                print(f"Error al actualizar resumen: {e}")
                self.summary = f"Interacciones recientes:{self.direct_history}"

        def get_context(self):
            """Obtener contexto actual de la conversación"""
            return self.summary if self.summary.strip() else "No hay conversación previa."


# Función principal para inicialización
async def main_pc():
    """Función principal robusta para inicialización y consulta"""
    asistente = AsistenteFisica()
    asistente.inicializar_componentes()

    # Leer PDFs desde PDF_DIR si no se especifican
    archivos_pdf = asistente.get_pdf_files()
    if not archivos_pdf:
        print("⚠️ No se encontraron archivos PDF en el directorio PDF_DIR")
        return None

    print(f"Procesando {len(archivos_pdf)} archivos PDF...")
    inicio_temario = time.time()
    temario = asistente.procesar_pdfs_temario(archivos_pdf)
    tiempo_temario = time.time() - inicio_temario
    print("\nTemario extraído:")
    print("-" * 80)
    print(temario)
    print("-" * 80)
    print(f"✅ Temario extraído en {tiempo_temario:.2f}s\n")

    inicio_chunks = time.time()
    await asistente.procesar_y_almacenar_pdfs(archivos_pdf)
    tiempo_chunks = time.time() - inicio_chunks
    print(f"✅ PDFs procesados y almacenados en Qdrant en {tiempo_chunks:.2f}s\n")

    # Consulta de ejemplo
    consulta = input("\nIngresa tu consulta de física: ")
    inicio_consulta = time.time()
    respuesta = await asistente.invoke(consulta, session_id="sesion_1")
    tiempo_consulta = time.time() - inicio_consulta

    print("\n📣 RESPUESTA FINAL:")
    print("-" * 80)
    print(respuesta)
    print("-" * 80)
    print(f"✅ Respuesta generada en {tiempo_consulta:.2f}s\n")

    # Guardar trayectoria de la consulta
    trayectoria = {
        "consulta": consulta,
        "temario": temario,
        "respuesta": respuesta,
        "tiempos": {
            "temario": tiempo_temario,
            "chunks": tiempo_chunks,
            "consulta": tiempo_consulta
        }
    }
    with open("trayectoria_adk.json", "w", encoding="utf-8") as f:
        json.dump(trayectoria, f, indent=4, ensure_ascii=False)
    print("✅ Trayectoria guardada en 'trayectoria_adk.json'")

    return asistente

# Ejemplo de uso
"""
import asyncio

# Inicializar el asistente
asistente = await main_pc()

# Usar el método invoke para comunicación (compatible con el segundo script)
if asistente:
    respuesta = await asistente.invoke("¿Qué es el efecto Doppler?", session_id="sesion_1")
    print(f"Respuesta: {respuesta}")
    
    # También puedes usar múltiples sesiones
    respuesta2 = await asistente.invoke("Explícame las ondas electromagnéticas", session_id="sesion_2")
    print(f"Respuesta 2: {respuesta2}")
"""