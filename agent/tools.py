"""Tool implementations for task management."""

from typing import List, Dict, Any
from datetime import datetime
from .schemas import Task

# In-memory task store
TASK_STORE: List[Task] = []


def add_task(title: str, due_date: str) -> Dict[str, Any]:
    """
    Add a new task to the task list.
    
    Args:
        title: Task description
        due_date: Due date in YYYY-MM-DD format
        
    Returns:
        Confirmation message with task details
    """
    try:
        # Validate date format
        datetime.strptime(due_date, "%Y-%m-%d")
        
        task = Task(title=title, due_date=due_date)
        TASK_STORE.append(task)
        
        return {
            "status": "success",
            "message": f"Task added: '{title}' due on {due_date}",
            "task_count": len(TASK_STORE)
        }
    except ValueError:
        return {
            "status": "error",
            "message": f"Invalid date format. Use YYYY-MM-DD format."
        }


def list_tasks() -> Dict[str, Any]:
    """
    List all pending tasks.
    
    Returns:
        Dictionary containing all tasks
    """
    if not TASK_STORE:
        return {
            "status": "success",
            "tasks": [],
            "message": "No tasks found"
        }
    
    tasks_data = [
        {
            "title": task.title,
            "due_date": task.due_date,
            "created_at": task.created_at,
            "completed": task.completed
        }
        for task in TASK_STORE
    ]
    
    return {
        "status": "success",
        "tasks": tasks_data,
        "count": len(tasks_data)
    }


def summarize_tasks() -> Dict[str, Any]:
    """
    Generate a natural language summary of all tasks.
    
    Returns:
        Summary of tasks
    """
    if not TASK_STORE:
        return {
            "status": "success",
            "summary": "You have no tasks at the moment."
        }
    
    total = len(TASK_STORE)
    pending = sum(1 for task in TASK_STORE if not task.completed)
    
    # Group by date
    today = datetime.now().strftime("%Y-%m-%d")
    due_today = [t for t in TASK_STORE if t.due_date == today]
    
    summary_parts = [f"You have {total} total task(s), {pending} pending."]
    
    if due_today:
        summary_parts.append(f"{len(due_today)} task(s) due today.")
    
    return {
        "status": "success",
        "summary": " ".join(summary_parts),
        "total": total,
        "pending": pending
    }


# Tool registry for dynamic execution
TOOL_REGISTRY = {
    "add_task": add_task,
    "list_tasks": list_tasks,
    "summarize_tasks": summarize_tasks,
}
