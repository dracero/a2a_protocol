Start the TellDateTimeTimezoneAgent

```uv run -m agents.tell_time_agent \
  --host localhost --port 10000```

Start the GreetingAgent

```uv run -m agents.greeting_agent \
  --host localhost --port 10001```

Start the Orchestrator (Host) Agent

```uv run -m agents.host_agent.entry \
  --host localhost --port 10002```

Launch the CLI (cmd.py)

```uv run -m app.cmd.cmd --agent http://localhost:10002```