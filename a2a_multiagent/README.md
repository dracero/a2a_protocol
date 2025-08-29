# a2a_multiagent

Este proyecto implementa un sistema multiagente A2A (Agent-to-Agent) con orquestador y agentes hijos usando Google Gemini y ADK.

## 1. Configuración del archivo `.env`

Crea un archivo llamado `.env` en el directorio raíz de `a2a_multiagent` con el siguiente contenido:

```
GOOGLE_API_KEY=tu_api_key_aqui
```

Reemplaza `tu_api_key_aqui` por tu clave real de Google Gemini.

---

## 2. Instalación de dependencias

Asegúrate de tener [uv](https://github.com/astral-sh/uv) instalado. Luego, desde el directorio `a2a_multiagent`:

```bash
uv pip install -r requirements.txt  # O usa pyproject.toml si corresponde
```

---

## 3. Iniciar los agentes hijos

Cada agente hijo tiene su propio entrypoint. Por ejemplo, para el agente de fecha y hora:

```bash
uv run -m agents.tell_datetimetz_agent.agent
```

Haz lo mismo para otros agentes, cambiando el módulo según corresponda.

---

## 4. Iniciar el orquestador

El orquestador puede iniciarse como API backend (FastAPI) o por línea de comando (CLI con Click).

### Como API (recomendado):

```bash
uvicorn agents.host_agent.entry:app --reload
```

O usando uv:

```bash
uv run -m uvicorn agents.host_agent.entry:app --reload
```

Esto levanta una API REST en el puerto 10002 por defecto.

### Como CLI (modo clásico con Click):

El archivo `entry.py` del orquestador soporta ejecución directa por CLI gracias a `@click.command`. Puedes pasarle opciones como host, puerto y path del registry:

```bash
uv run -m agents.host_agent.entry --host localhost --port 10002 --registry utilities/agent_registry.json
```

---

## 5. Uso de la API del orquestador

- Endpoint de salud: `GET /`
- Enviar tarea: `POST /tasks/send` (ver estructura en el código)

---

## 6. Notas

- Asegúrate de que todos los agentes estén corriendo antes de iniciar el orquestador.
- El orquestador descubre los agentes hijos usando el archivo `utilities/agent_registry.json`.
- Si tienes problemas con variables de entorno, revisa la consola para mensajes de debug.

---

## 7. Arranque de agentes y orquestador

Para iniciar los agentes y el orquestador, sigue estos pasos:

1. **Iniciar el agente de fecha y hora**:

   ```bash
   uv run -m agents.tell_datetimetz_agent \
     --host localhost --port 10000
   ```

2. **Iniciar el agente de saludo**:

   ```bash
    uv run -m agents.greeting_agent \
      --host localhost --port 10001
   ```

3. **Iniciar el agente de física**:

   ```bash
   uv run -m agents.asistente_fisica \
     --host localhost --port 10003
   ```

4. **Iniciar el agente generador de imagenes**:

   ```bash
    uv run -m agents.image_generation \
      --host localhost --port 10004


5. **Iniciar el orquestador (agente host)**:

   ```bash
      uv run -m agents.host_agent.entry \
        --host localhost --port 10002
   ```

6. **Lanzar la CLI (cmd.py)**:

   ```bash
   uv run -m app.cmd.cmd --agent http://localhost:10002
   ```

---

## Sobre uv (Python package manager)

[uv](https://github.com/astral-sh/uv) es un gestor de paquetes y proyectos para Python extremadamente rápido, escrito en Rust. Reemplaza herramientas como `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine` y `virtualenv` en un solo comando.

### Características principales
- 🚀 10-100x más rápido que pip.
- 🗂️ Manejo de proyectos, entornos virtuales y lockfiles universales.
- 🐍 Instalación y gestión de múltiples versiones de Python.
- 🛠️ Ejecución e instalación de herramientas CLI publicadas como paquetes Python.
- 💾 Cache global eficiente para deduplicación de dependencias.
- 🔩 Interfaz compatible con pip y comandos familiares.
- 🏢 Soporte para workspaces estilo Cargo.

### Instalación

**Con instalador (recomendado):**

```bash
# En macOS y Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# En Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Con pip:**

```bash
pip install uv
```

**Con pipx:**

```bash
pipx install uv
```

### Comandos útiles

- Crear un entorno virtual:
  ```bash
  uv venv
  ```
- Instalar dependencias:
  ```bash
  uv pip install -r requirements.txt
  ```
- Ejecutar scripts o módulos:
  ```bash
  uv run -m modulo
  ```
- Sincronizar dependencias con lockfile:
  ```bash
  uv pip sync requirements.txt
  ```
- Instalar herramientas CLI:
  ```bash
  uv tool install ruff
  ```
- Actualizar uv:
  ```bash
  uv self update
  ```

Más información y documentación en: https://docs.astral.sh/uv/

---

**¡Listo! Ahora puedes experimentar con tu sistema multiagente A2A.**