"""Test script to verify agent tools and executor locally."""

import json
from agent.tools import add_task, list_tasks, summarize_tasks, TASK_STORE
from agent.executor import parse_output, execute_tool, format_observation


def test_tools():
    """Test individual tool functions."""
    print("Testing Agent Tools")
    print("=" * 60)
    
    # Clear task store
    TASK_STORE.clear()
    
    # Test add_task
    print("\n1. Testing add_task...")
    result = add_task("Buy groceries", "2026-01-20")
    print(f"Result: {result}")
    assert result["status"] == "success"
    
    result = add_task("Finish report", "2026-01-22")
    print(f"Result: {result}")
    
    # Test list_tasks
    print("\n2. Testing list_tasks...")
    result = list_tasks()
    print(f"Result: {result}")
    assert result["count"] == 2
    
    # Test summarize_tasks
    print("\n3. Testing summarize_tasks...")
    result = summarize_tasks()
    print(f"Result: {result}")
    
    print("\nAll tool tests passed!")


def test_parser():
    """Test output parser."""
    print("\n\nTesting Output Parser")
    print("=" * 60)
    
    # Test tool invocation parsing
    print("\n1. Testing tool invocation parsing...")
    tool_output = """<analysis>
User wants to create a new task.
</analysis>

<action>
add_task(title="Test task", due_date="2026-01-20")
</action>"""
    
    response_type, analysis, action = parse_output(tool_output)
    print(f"Type: {response_type}")
    print(f"Analysis: {analysis}")
    print(f"Action: {action}")
    assert response_type == "tool"
    
    # Test direct response parsing
    print("\n2. Testing direct response parsing...")
    direct_output = """<final>
Hello! How can I help you?
</final>"""
    
    response_type, analysis, final = parse_output(direct_output)
    print(f"Type: {response_type}")
    print(f"Final: {final}")
    assert response_type == "direct"
    
    print("\nParser tests passed!")


def test_executor():
    """Test tool executor."""
    print("\n\nTesting Tool Executor")
    print("=" * 60)
    
    # Clear task store
    TASK_STORE.clear()
    
    # Test tool execution
    print("\n1. Testing tool execution...")
    action = 'add_task(title="Meeting with team", due_date="2026-01-25")'
    result = execute_tool(action)
    print(f"Result: {result}")
    assert result["status"] == "success"
    
    # Test list execution
    print("\n2. Testing list execution...")
    action = 'list_tasks()'
    result = execute_tool(action)
    print(f"Result: {result}")
    
    # Test observation formatting
    print("\n3. Testing observation formatting...")
    observation = format_observation(result)
    print(f"Observation:\n{observation}")
    
    print("\nExecutor tests passed!")


def test_end_to_end():
    """Test complete end-to-end flow."""
    print("\n\nTesting End-to-End Flow")
    print("=" * 60)
    
    # Clear task store
    TASK_STORE.clear()
    
    # Simulate model output
    model_output = """<analysis>
User wants to add a task to buy milk tomorrow.
</analysis>

<action>
add_task(title="Buy milk", due_date="2026-01-20")
</action>"""
    
    print("\n1. Parse model output...")
    response_type, analysis, action = parse_output(model_output)
    print(f"Parsed: type={response_type}, action={action}")
    
    print("\n2. Execute tool...")
    result = execute_tool(action)
    print(f"Tool result: {result}")
    
    print("\n3. Format observation...")
    observation = format_observation(result)
    print(f"Observation: {observation}")
    
    print("\nEnd-to-end test passed!")


def main():
    """Run all tests."""
    try:
        test_tools()
        test_parser()
        test_executor()
        test_end_to_end()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe agent core is working correctly.")
        print("Next steps:")
        print("1. Generate dataset: python data/generate_dataset.py")
        print("2. Train model on Colab: training_notebook.ipynb")
        print("3. Deploy API: python serving/app.py")
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
