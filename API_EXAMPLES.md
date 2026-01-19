# Example API Usage

This document provides example requests for testing the deployed API.

## Prerequisites

Start the server:
```bash
python serving/app.py
```

The API will be available at `http://localhost:8000`.

## Example Requests

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Add a Task

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Add a task to buy groceries tomorrow"}'
```

**Response**:
```json
{
  "response": "Task added: 'Buy groceries' due on 2026-01-20",
  "tool_used": "add_task",
  "observation": "Task added: 'Buy groceries' due on 2026-01-20"
}
```

### 3. List Tasks

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What tasks do I have?"}'
```

**Response**:
```json
{
  "response": "Tasks:\n1. Buy groceries (due: 2026-01-20)",
  "tool_used": "list_tasks",
  "observation": "Tasks:\n1. Buy groceries (due: 2026-01-20)"
}
```

### 4. Summarize Tasks

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize my tasks"}'
```

**Response**:
```json
{
  "response": "You have 1 total task(s), 1 pending.",
  "tool_used": "summarize_tasks",
  "observation": "You have 1 total task(s), 1 pending."
}
```

### 5. Conversational Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

**Response**:
```json
{
  "response": "Hello! How can I help you manage your tasks today?",
  "tool_used": null,
  "observation": null
}
```

## Using Python Requests

```python
import requests

url = "http://localhost:8000/chat"

# Add a task
response = requests.post(
    url,
    json={"message": "Add a task to finish project report by Friday"}
)
print(response.json())

# List tasks
response = requests.post(
    url,
    json={"message": "Show me my tasks"}
)
print(response.json())
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Use these interfaces to test the API interactively in your browser.
