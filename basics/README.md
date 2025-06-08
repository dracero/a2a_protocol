# 📍 TellDateTimeTimezoneAgent – Google A2A Protocol Demo

This project demonstrates a minimal implementation of Google's **Agent-to-Agent (A2A) Protocol** using Python.

Full details: https://medium.com/@pankajchandravanshi/82676bc27144?source=friends_link&sk=3816931bf7f60690895e9132d98d5d98

It features:

- A simple A2A server agent (`TellDateTimeTimezoneAgent`) built with Flask  
- A client agent that **discovers** and **communicates** with it  
- Full compliance with the **A2A message structure** and **discovery flow**  

This is perfect for beginners who want to understand how agents discover each other and exchange messages using A2A.

---

## 🚀 Features

✅ Implements **A2A discovery** via `/.well-known/agent.json`  
✅ Exposes a `tasks/send` endpoint for receiving tasks  
✅ Replies to queries with the **current system date**, **time**, and **system timezone**  
✅ Client generates a task and parses the response using A2A conventions  

---

## 📂 Project Structure

```plaintext
a2a_samples/
├── server/
│   └── tell_datetime_timezone_server.py    # Flask-based A2A server agent
├── client/
│   └── datetime_timezone_client.py         # A2A client agent that queries the server
```

---

## 🧪 How to Run

### 1 Start the server

```bash
cd a2a_samples/server
uv init .
uv venv
source .venv/bin/activate
uv add flask tzlocal
uv run tell_datetime_timezone_server.py
```

---

### 2 Run the client

In a separate terminal:

```bash
cd a2a_samples/client
uv run datetime_timezone_client.py
```

---

## ✅ Example Output

```plaintext
Connected to: TellDateTimeTimezoneAgent – Tells the current date, time, and timezone when asked.
Agent says: The current date and time is: 2025-06-08 14:32:17. The system timezone is: Europe/Paris.
```

---

## Notes

- The **timezone** is automatically extracted from the system where the server is running (e.g. `"Europe/Paris"` or `"Asia/Kolkata"`).  
- No external API calls are used — fully local extraction.

---

## Summary

This simple demo shows how an **Agent-to-Agent (A2A)** conversation works:  
🔍 **Discovery → Task → Response → Artifacts → Completion**

Agents can now communicate across boundaries using a well-defined, structured protocol — one of the foundational ideas behind future **multi-agent collaboration**.

---

## 📌 Placeholder for A2A Protocol Diagram

<!-- You can add a diagram here illustrating:
     - A2A Client → Discovery → Agent Card
     - Client → Task Send → Server
     - Server → Response → Client
-->

---

## License

This project is released under the license terms specified in this repository by the owner.  
Please refer to the `LICENSE` file (if provided), or to the license selected on this GitHub repository page for details.
