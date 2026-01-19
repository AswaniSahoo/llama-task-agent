"""Data models for task agent."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Represents a single task."""
    
    title: str = Field(..., description="Task description")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format")
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    completed: bool = False


class ToolInvocation(BaseModel):
    """Model for tool-based responses."""
    
    analysis: str = Field(..., description="Reasoning about why a tool is needed")
    action: str = Field(..., description="Tool call with arguments")


class DirectResponse(BaseModel):
    """Model for direct conversational responses."""
    
    final: str = Field(..., description="Natural language response when no tool is required")


class ChatRequest(BaseModel):
    """API request schema."""
    
    message: str


class ChatResponse(BaseModel):
    """API response schema."""
    
    response: str
    tool_used: Optional[str] = None
    observation: Optional[str] = None
