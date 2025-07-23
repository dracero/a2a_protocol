import logging
from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart

logger = logging.getLogger(__name__)

class FisicaTaskManager(InMemoryTaskManager):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        logger.info("✅ FisicaTaskManager inicializado")

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """Extraer texto del usuario del request"""
        try:
            user_text = request.params.message.parts[0].text
            logger.info(f"📝 Texto extraído: {user_text[:100]}...")
            return user_text
        except (AttributeError, IndexError) as e:
            logger.error(f"❌ Error extrayendo texto del usuario: {e}")
            return "Error al procesar la consulta"

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Procesar tarea enviada al agente"""
        task_id = request.params.id
        logger.info(f"🚀 FisicaTaskManager recibió task {task_id}")
        
        try:
            # Crear/actualizar tarea
            logger.info(f"📦 Creando tarea para {task_id}")
            task = await self.upsert_task(request.params)
            logger.info(f"✅ Tarea creada: {task}")
            
            # Extraer texto del usuario
            user_text = self._get_user_text(request)
            logger.info(f"👤 Consulta del usuario: {user_text}")
            
            # Verificar que el agente esté inicializado
            if not hasattr(self.agent, 'llm') or self.agent.llm is None:
                logger.warning("⚠️ Agente no completamente inicializado, inicializando componentes...")
                try:
                    self.agent.inicializar_componentes()
                    logger.info("✅ Componentes inicializados correctamente")
                except Exception as init_error:
                    logger.error(f"❌ Error inicializando componentes: {init_error}")
                    response_text = f"Error: El agente no está correctamente inicializado. {str(init_error)}"
                else:
                    # Procesar consulta después de inicializar
                    response_text = await self._process_query(user_text, request.params.sessionId)
            else:
                logger.info("✅ Agente ya inicializado, procesando consulta...")
                # Agente ya inicializado, procesar consulta
                response_text = await self._process_query(user_text, request.params.sessionId)
            
            logger.info(f"✅ Respuesta obtenida: {response_text[:200]}...")
            
            # Crear mensaje de respuesta
            reply_message = Message(
                role="agent",
                parts=[TextPart(text=response_text)]
            )
            logger.info(f"📨 Mensaje de respuesta creado: {reply_message}")
            
            # Actualizar tarea con respuesta
            async with self.lock:
                task.status = TaskStatus(state=TaskState.COMPLETED)
                task.history.append(reply_message)
                logger.info(f"✅ Tarea actualizada: status={task.status}, history_length={len(task.history)}")
            
            # Crear respuesta
            response = SendTaskResponse(id=request.id, result=task)
            logger.info(f"📤 SendTaskResponse creado con ID: {request.id}")
            logger.info(f"📤 Response result type: {type(response.result)}")
            logger.info(f"📤 Task status: {response.result.status}")
            logger.info(f"📤 Task history: {len(response.result.history)} messages")
            
            logger.info(f"✅ Tarea {task_id} completada exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error procesando tarea {task_id}: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Crear mensaje de error
            error_message = Message(
                role="agent",
                parts=[TextPart(text=f"Error procesando consulta: {str(e)}")]
            )
            
            try:
                # Intentar actualizar tarea con error
                async with self.lock:
                    if 'task' in locals():
                        task.status = TaskStatus(state=TaskState.FAILED)
                        task.history.append(error_message)
                        logger.info(f"❌ Tarea marcada como fallida: {task_id}")
                    else:
                        # Si no se pudo crear la tarea, crear una básica
                        task = await self.upsert_task(request.params)
                        task.status = TaskStatus(state=TaskState.FAILED)
                        task.history.append(error_message)
                        logger.info(f"❌ Tarea creada y marcada como fallida: {task_id}")
                
                error_response = SendTaskResponse(id=request.id, result=task)
                logger.info(f"❌ Retornando respuesta de error para {task_id}")
                return error_response
                    
            except Exception as task_error:
                logger.error(f"❌ Error crítico manejando tarea: {task_error}")
                # Re-raise el error original para que el servidor lo maneje
                raise e

    async def _process_query(self, user_text: str, session_id: str) -> str:
        """Procesar consulta del usuario con manejo de errores mejorado"""
        try:
            logger.info(f"🔄 Procesando consulta: {user_text[:100]}... con session_id: {session_id}")
            
            # Llamar al método invoke del agente
            response_text = await self.agent.invoke(user_text, session_id)
            
            # Verificar que la respuesta no esté vacía
            if not response_text or response_text.strip() == "":
                logger.warning("⚠️ Respuesta vacía del agente")
                response_text = "No pude generar una respuesta. Intenta reformular tu pregunta."
            
            logger.info(f"✅ Respuesta generada: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error en _process_query: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return f"Error procesando tu consulta: {str(e)}. Por favor intenta de nuevo."