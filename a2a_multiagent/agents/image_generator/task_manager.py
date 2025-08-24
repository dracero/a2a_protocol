import logging
from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart

logger = logging.getLogger(__name__)

class ImageGeneratorTaskManager(InMemoryTaskManager):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        logger.info("✅ ImageGeneratorTaskManager inicializado")

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """Extraer texto del usuario del request"""
        try:
            user_text = request.params.message.parts[0].text
            logger.info(f"📝 Texto extraído: {user_text[:100]}...")
            return user_text
        except (AttributeError, IndexError) as e:
            logger.error(f"❌ Error extrayendo texto del usuario: {e}")
            return "Error al procesar la consulta"

    async def _process_query(self, user_text: str, session_id: str) -> str:
        """Procesar la consulta del usuario a través del agente"""
        try:
            logger.info(f"🔄 Procesando consulta para sesión: {session_id}")
            
            # El ImageGeneratorAgent usa el método 'invoke' como método principal
            if hasattr(self.agent, 'invoke'):
                response = await self.agent.invoke(user_text, session_id)
            elif hasattr(self.agent, 'generate_image'):
                # Método alternativo usando generate_image directamente
                response = self.agent.generate_image(user_text, session_id)
            elif hasattr(self.agent, 'process_query'):
                response = await self.agent.process_query(user_text, session_id)
            else:
                # Último recurso - intentar llamar algún método disponible
                logger.warning("⚠️ Métodos estándar no encontrados, usando método genérico")
                response = f"Error: El agente no tiene métodos de procesamiento disponibles"
            
            logger.info(f"✅ Consulta procesada exitosamente")
            return str(response)
            
        except Exception as e:
            logger.error(f"❌ Error procesando consulta: {e}")
            return f"Error procesando la consulta: {str(e)}"

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Procesar tarea enviada al agente de generación de imágenes"""
        task_id = request.params.id
        logger.info(f"🚀 ImageGeneratorTaskManager recibió task {task_id}")
        
        try:
            # Crear/actualizar tarea
            logger.info(f"📦 Creando tarea para {task_id}")
            task = await self.upsert_task(request.params)
            logger.info(f"✅ Tarea creada: {task}")
            
            # Extraer texto del usuario
            user_text = self._get_user_text(request)
            logger.info(f"👤 Prompt del usuario: {user_text}")
            
            # Verificar que el agente esté inicializado
            if not hasattr(self.agent, '_components_initialized') or not self.agent._components_initialized:
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
                parts=[TextPart(text=f"Error procesando solicitud de generación de imagen: {str(e)}")]
            )
            
            try:
                # Intentar actualizar tarea con error
                async with self.lock:
                    if 'task' in locals():
                        task.status = TaskStatus(state=TaskState.FAILED)
                        task.history.append(error_message)
                        logger.info(f"❌ Tarea {task_id} marcada como fallida")
                        
                        # Retornar respuesta con error
                        return SendTaskResponse(id=request.id, result=task)
                    else:
                        # Si no se pudo crear la tarea, crear una básica con error
                        logger.error(f"❌ No se pudo crear tarea, creando tarea de error básica")
                        from models.task import Task
                        error_task = Task(
                            id=task_id,
                            status=TaskStatus(state=TaskState.FAILED),
                            history=[error_message],
                            sessionId=request.params.sessionId if hasattr(request.params, 'sessionId') else "unknown"
                        )
                        return SendTaskResponse(id=request.id, result=error_task)
                        
            except Exception as error_handling_exception:
                logger.error(f"❌ Error crítico manejando error: {error_handling_exception}")
                
                # Último recurso: crear respuesta de error mínima
                from models.task import Task
                fallback_task = Task(
                    id=task_id,
                    status=TaskStatus(state=TaskState.FAILED),
                    history=[Message(
                        role="agent",
                        parts=[TextPart(text="Error crítico en el sistema. Por favor, inténtelo más tarde.")]
                    )],
                    sessionId="error"
                )
                return SendTaskResponse(id=request.id, result=fallback_task)

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Obtener el estado de una tarea específica"""
        try:
            async with self.lock:
                task = self.tasks.get(task_id)
                if task:
                    logger.info(f"📊 Estado de tarea {task_id}: {task.status}")
                    return task.status
                else:
                    logger.warning(f"⚠️ Tarea {task_id} no encontrada")
                    return TaskStatus(state=TaskState.FAILED)
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado de tarea {task_id}: {e}")
            return TaskStatus(state=TaskState.FAILED)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancelar una tarea en progreso"""
        try:
            async with self.lock:
                task = self.tasks.get(task_id)
                if task and task.status.state in [TaskState.PENDING, TaskState.RUNNING]:
                    task.status = TaskStatus(state=TaskState.CANCELLED)
                    logger.info(f"🚫 Tarea {task_id} cancelada")
                    return True
                else:
                    logger.warning(f"⚠️ No se pudo cancelar tarea {task_id}")
                    return False
        except Exception as e:
            logger.error(f"❌ Error cancelando tarea {task_id}: {e}")
            return False

    def get_agent_status(self) -> dict:
        """Obtener información del estado del agente"""
        try:
            status = {
                "initialized": hasattr(self.agent, '_components_initialized') and self.agent._components_initialized,
                "agent_type": type(self.agent).__name__,
                "active_tasks": len(self.tasks),
                "available_methods": [method for method in dir(self.agent) if not method.startswith('_')],
                "image_cache_sessions": len(self.agent.image_cache.list_sessions()) if hasattr(self.agent, 'image_cache') else 0,
                "crewai_ready": all([
                    hasattr(self.agent, 'model') and self.agent.model is not None,
                    hasattr(self.agent, 'image_creator_agent') and self.agent.image_creator_agent is not None,
                    hasattr(self.agent, 'prompt_enhancer_agent') and self.agent.prompt_enhancer_agent is not None
                ]) if hasattr(self.agent, '_components_initialized') and self.agent._components_initialized else False
            }
            logger.info(f"📈 Estado del agente: {status}")
            return status
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del agente: {e}")
            return {"error": str(e)}

    async def get_session_images(self, session_id: str) -> dict:
        """Obtener información de imágenes de una sesión específica"""
        try:
            if hasattr(self.agent, 'image_cache'):
                stats = self.agent.image_cache.get_session_stats(session_id)
                images = self.agent.list_images(session_id)
                return {
                    "session_id": session_id,
                    "stats": stats,
                    "images": images
                }
            else:
                return {"error": "Image cache no disponible"}
        except Exception as e:
            logger.error(f"❌ Error obteniendo imágenes de sesión {session_id}: {e}")
            return {"error": str(e)}

    async def get_image_data(self, image_id: str, session_id: str = "default") -> dict:
        """Obtener datos de una imagen específica"""
        try:
            if hasattr(self.agent, 'get_image_data'):
                image_data = self.agent.get_image_data(image_id, session_id)
                if image_data:
                    return {
                        "id": image_data.id,
                        "name": image_data.name,
                        "mime_type": image_data.mime_type,
                        "has_data": bool(image_data.bytes)
                    }
                else:
                    return {"error": f"Imagen {image_id} no encontrada en sesión {session_id}"}
            else:
                return {"error": "Método get_image_data no disponible"}
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos de imagen {image_id}: {e}")
            return {"error": str(e)}

    def cleanup_session(self, session_id: str) -> bool:
        """Limpiar datos de una sesión específica"""
        try:
            if hasattr(self.agent, 'image_cache'):
                # Remover datos de la sesión del cache
                self.agent.image_cache._cache.pop(session_id, None)
                logger.info(f"🧹 Sesión {session_id} limpiada del cache")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error limpiando sesión {session_id}: {e}")
            return False