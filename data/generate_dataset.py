"""Generate synthetic dataset for fine-tuning the task agent."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any


def generate_date(days_offset: int = 0) -> str:
    """Generate a date string with optional offset from today."""
    date = datetime.now() + timedelta(days=days_offset)
    return date.strftime("%Y-%m-%d")


def generate_tool_invocation_samples() -> List[Dict[str, Any]]:
    """Generate samples for tool invocation scenarios (add_task)."""
    samples = []
    
    # Sample task titles
    tasks = [
        "buy groceries", "finish project report", "call dentist",
        "pay electricity bill", "book flight tickets", "gym workout",
        "read a book", "prepare presentation", "team meeting",
        "code review", "update resume", "water plants",
        "clean the house", "fix the bug", "write documentation",
        "schedule appointment", "send email", "backup files",
        "order supplies", "renew subscription", "plan vacation",
        "study for exam", "practice guitar", "walk the dog",
        "organize desk", "reply to messages", "attend webinar"
    ]
    
    # Variations of how users might request adding tasks
    templates = [
        "Add a task to {task}",
        "I need to {task}",
        "Remind me to {task}",
        "Create a task for {task}",
        "Add {task} to my list",
        "I have to {task}",
        "Don't forget to {task}",
        "Schedule {task}",
        "Put {task} on my todo list",
        "I should {task}",
    ]
    
    # Time specifications
    time_specs = [
        ("tomorrow", 1),
        ("today", 0),
        ("next week", 7),
        ("in 3 days", 3),
        ("by Friday", 5),
        ("next Monday", 7),
        ("this weekend", 6),
    ]
    
    # Generate samples
    for i in range(120):  # 120 add_task samples
        task = random.choice(tasks)
        template = random.choice(templates)
        time_spec, days = random.choice(time_specs)
        
        instruction = f"{template.format(task=task)} {time_spec}"
        due_date = generate_date(days)
        
        sample = {
            "instruction": instruction,
            "analysis": f"User wants to create a new task '{task}' with due date {time_spec}.",
            "action": f'add_task(title="{task.capitalize()}", due_date="{due_date}")'
        }
        samples.append(sample)
    
    return samples


def generate_list_tasks_samples() -> List[Dict[str, Any]]:
    """Generate samples for listing tasks."""
    samples = []
    
    # Variations of requests to list tasks
    queries = [
        "What tasks do I have?",
        "Show me my tasks",
        "List all my tasks",
        "What's on my todo list?",
        "Show my pending tasks",
        "What do I need to do?",
        "Display my tasks",
        "What are my tasks?",
        "Can you show me what I have to do?",
        "What's pending?",
        "Give me my task list",
        "Show all tasks",
        "What tasks are there?",
        "List tasks",
        "Show todo items",
    ]
    
    for i in range(60):  # 60 list_tasks samples
        instruction = random.choice(queries)
        
        sample = {
            "instruction": instruction,
            "analysis": "User is requesting a list of all tasks.",
            "action": "list_tasks()"
        }
        samples.append(sample)
    
    return samples


def generate_summarize_tasks_samples() -> List[Dict[str, Any]]:
    """Generate samples for task summary."""
    samples = []
    
    # Variations of requests for summaries
    queries = [
        "Summarize my tasks",
        "Give me an overview of my tasks",
        "How many tasks do I have?",
        "What's my task status?",
        "Task summary please",
        "Give me a summary",
        "Overview of tasks",
        "How's my todo list looking?",
        "What's the status of my tasks?",
        "Summarize what I need to do",
        "Quick summary of tasks",
        "Task overview",
    ]
    
    for i in range(60):  # 60 summarize_tasks samples
        instruction = random.choice(queries)
        
        sample = {
            "instruction": instruction,
            "analysis": "User wants a summary of their tasks.",
            "action": "summarize_tasks()"
        }
        samples.append(sample)
    
    return samples


def generate_direct_response_samples() -> List[Dict[str, Any]]:
    """Generate samples for direct conversational responses (no tools)."""
    samples = []
    
    # Greetings and casual conversation
    conversations = [
        ("Hello", "Hello! How can I help you manage your tasks today?"),
        ("Hi there", "Hi! I'm here to help you with your tasks. What would you like to do?"),
        ("Hey", "Hey! Ready to help you organize your tasks."),
        ("Good morning", "Good morning! What can I help you with today?"),
        ("How are you?", "I'm functioning well, thank you! How can I assist you with your tasks?"),
        ("Thanks", "You're welcome! Let me know if you need anything else."),
        ("Thank you", "Happy to help! Feel free to ask if you need anything."),
        ("Goodbye", "Goodbye! Have a productive day!"),
        ("Bye", "Bye! Come back anytime you need task management help."),
        ("What can you do?", "I can help you manage tasks: add new tasks, list your tasks, and provide summaries. Just let me know what you need!"),
        ("Help", "I can add tasks, list your tasks, and provide summaries. What would you like to do?"),
        ("Who are you?", "I'm a task management assistant. I can help you organize and track your tasks efficiently."),
    ]
    
    # Questions not requiring tools
    questions = [
        ("What time is it?", "I don't have access to the current time, but I can help you manage your tasks."),
        ("Tell me a joke", "I'm focused on helping with tasks, but I appreciate your humor! How can I assist you today?"),
        ("What's the weather?", "I don't have weather information, but I can help you manage your tasks."),
    ]
    
    # Combine all conversational samples
    all_convs = conversations + questions
    
    for instruction, response in all_convs:
        sample = {
            "instruction": instruction,
            "final": response
        }
        samples.append(sample)
    
    # Add variations by repeating with slight changes
    for i in range(60):  # Total ~120 direct response samples
        instruction, response = random.choice(conversations)
        sample = {
            "instruction": instruction,
            "final": response
        }
        samples.append(sample)
    
    return samples


def generate_edge_cases() -> List[Dict[str, Any]]:
    """Generate edge case samples."""
    samples = []
    
    # Ambiguous or complex requests
    edge_cases = [
        {
            "instruction": "Add buy milk and call mom",
            "analysis": "User mentions multiple tasks but in a single request. Adding the first task mentioned.",
            "action": 'add_task(title="Buy milk", due_date="' + generate_date(1) + '")'
        },
        {
            "instruction": "I need to do something tomorrow",
            "analysis": "User wants to add a task but didn't specify what. This is ambiguous.",
            "final": "Could you please specify what task you'd like to add for tomorrow?"
        },
        {
            "instruction": "Delete all tasks",
            "final": "I don't have a delete function available. I can add tasks, list them, or summarize them."
        },
        {
            "instruction": "Mark task as complete",
            "final": "I don't have the ability to mark tasks as complete yet. I can add, list, and summarize tasks."
        },
        {
            "instruction": "What's task number 3?",
            "analysis": "User wants to see a specific task. Listing all tasks is the closest available action.",
            "action": "list_tasks()"
        },
    ]
    
    samples.extend(edge_cases)
    
    # Repeat some edge cases with variations
    for i in range(35):  # Total ~40 edge case samples
        sample = random.choice(edge_cases).copy()
        samples.append(sample)
    
    return samples


def save_dataset(samples: List[Dict[str, Any]], output_dir: str = "data"):
    """Save dataset and create train/eval splits."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Save full dataset
    dataset_file = output_path / "dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(samples)} total samples")
    print(f"Saved to: {dataset_file}")
    
    # Create train/eval split (90/10)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    # Save splits
    train_file = output_path / "train.json"
    eval_file = output_path / "eval.json"
    
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Train samples: {len(train_samples)} -> {train_file}")
    print(f"Eval samples: {len(eval_samples)} -> {eval_file}")
    
    # Print distribution
    tool_samples = sum(1 for s in samples if "action" in s)
    direct_samples = sum(1 for s in samples if "final" in s)
    
    print(f"\nDataset distribution:")
    print(f"  Tool invocations: {tool_samples} ({tool_samples/len(samples)*100:.1f}%)")
    print(f"  Direct responses: {direct_samples} ({direct_samples/len(samples)*100:.1f}%)")


def main():
    """Generate complete dataset."""
    print("Generating synthetic dataset for task agent training...")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate all samples
    samples = []
    
    print("Generating tool invocation samples...")
    samples.extend(generate_tool_invocation_samples())
    
    print("Generating list tasks samples...")
    samples.extend(generate_list_tasks_samples())
    
    print("Generating summarize tasks samples...")
    samples.extend(generate_summarize_tasks_samples())
    
    print("Generating direct response samples...")
    samples.extend(generate_direct_response_samples())
    
    print("Generating edge case samples...")
    samples.extend(generate_edge_cases())
    
    print()
    
    # Save dataset
    save_dataset(samples)
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
