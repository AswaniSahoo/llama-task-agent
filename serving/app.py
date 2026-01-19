"""FastAPI server for task agent."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent.schemas import ChatRequest, ChatResponse
from agent.executor import parse_output, execute_tool, format_observation
from serving.inference import TaskAgentInference
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Task Agent API",
    description="Fine-tuned LLaMA agent for task-oriented tool execution",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global inference_engine
    print("Loading model...")
    inference_engine = TaskAgentInference()
    inference_engine.load_model()
    print("Model ready!")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Processes user message, generates response, and executes tools if needed.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, message="Model not loaded")
    
    try:
        # Generate response from model
        model_output = inference_engine.generate(request.message)
        
        # Parse output
        response_type, analysis, action_or_final = parse_output(model_output)
        
        # Handle response
        if response_type == "tool":
            # Execute tool
            tool_result = execute_tool(action_or_final)
            observation = format_observation(tool_result)
            
            # Extract tool name
            tool_name = action_or_final.split("(")[0] if "(" in action_or_final else "unknown"
            
            return ChatResponse(
                response=observation,
                tool_used=tool_name,
                observation=observation
            )
        else:
            # Direct response
            return ChatResponse(
                response=action_or_final,
                tool_used=None,
                observation=None
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Task Agent API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        },
        "description": "Fine-tuned LLaMA agent for task management"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
