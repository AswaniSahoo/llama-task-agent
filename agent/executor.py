"""Output parser and tool executor."""

import re
import json
from typing import Dict, Any, Tuple, Optional
from .tools import TOOL_REGISTRY


def parse_output(model_output: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse model output to extract analysis, action, or final response.
    
    Args:
        model_output: Raw output from the LLM
        
    Returns:
        Tuple of (response_type, analysis, action_or_final)
        - response_type: "tool" or "direct"
        - analysis: reasoning (if tool invocation)
        - action_or_final: tool call or final response
    """
    # Extract analysis
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', model_output, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else None
    
    # Extract action
    action_match = re.search(r'<action>(.*?)</action>', model_output, re.DOTALL)
    action = action_match.group(1).strip() if action_match else None
    
    # Extract final response
    final_match = re.search(r'<final>(.*?)</final>', model_output, re.DOTALL)
    final = final_match.group(1).strip() if final_match else None
    
    if action and analysis:
        return "tool", analysis, action
    elif final:
        return "direct", None, final
    else:
        # Fallback if model doesn't follow format
        return "direct", None, model_output.strip()


def execute_tool(action: str) -> Dict[str, Any]:
    """
    Execute a tool based on the action string.
    
    Args:
        action: Tool call string, e.g., 'add_task(title="Buy milk", due_date="2026-01-20")'
        
    Returns:
        Tool execution result
    """
    try:
        # Parse tool name and arguments
        tool_match = re.match(r'(\w+)\((.*)\)', action)
        
        if not tool_match:
            return {
                "status": "error",
                "message": f"Invalid tool format: {action}"
            }
        
        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)
        
        # Check if tool exists
        if tool_name not in TOOL_REGISTRY:
            return {
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }
        
        # Parse arguments
        kwargs = {}
        if args_str.strip():
            # Simple argument parsing for key=value format
            arg_pattern = r'(\w+)=(?:"([^"]*)"|\'([^\']*)\')'
            for match in re.finditer(arg_pattern, args_str):
                key = match.group(1)
                value = match.group(2) or match.group(3)
                kwargs[key] = value
        
        # Execute tool
        tool_func = TOOL_REGISTRY[tool_name]
        result = tool_func(**kwargs)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}"
        }


def format_observation(tool_result: Dict[str, Any]) -> str:
    """
    Format tool execution result as a readable observation.
    
    Args:
        tool_result: Dictionary result from tool execution
        
    Returns:
        Formatted observation string
    """
    if tool_result.get("status") == "error":
        return f"Error: {tool_result.get('message', 'Unknown error')}"
    
    # Format based on result structure
    if "summary" in tool_result:
        return tool_result["summary"]
    elif "tasks" in tool_result:
        tasks = tool_result["tasks"]
        if not tasks:
            return "No tasks found"
        
        lines = ["Tasks:"]
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. {task['title']} (due: {task['due_date']})")
        return "\n".join(lines)
    else:
        return tool_result.get("message", json.dumps(tool_result))
