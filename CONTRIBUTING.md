# Contributing to LLaMA Task Agent

Thank you for your interest in this project. This guide will help you reproduce the work and understand the development process.

## Table of Contents

- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)

## Getting Started

### Environment Setup

This project uses UV for package management:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or download from https://github.com/astral-sh/uv

# Clone repository
git clone <repository-url>
cd personaltaskai

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Testing Installation

Verify the installation works:

```bash
# Generate dataset
python data/generate_dataset.py

# Test agent components
python test_agent.py
```

Expected output: All tests should pass showing tool execution, parsing, and formatting working correctly.

## Training the Model

### Option 1: Kaggle (Recommended)

**Advantages**: Free T4 GPU, no setup required, 30 hours/week limit

**Steps**:
1. Create Kaggle account at kaggle.com
2. Create new notebook
3. Settings:
   - Accelerator: GPU T4 x2
   - Internet: ON
4. Upload `train-standalone.ipynb`
5. Run all cells

**HuggingFace Token**:
- Get token from https://huggingface.co/settings/tokens
- Accept LLaMA license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Add token when prompted in notebook

**Training Time**: Approximately 90 minutes

**Output**: Adapter saved to `models/lora-adapter/` (Download as dataset)

### Option 2: Google Colab

**Advantages**: Similar to Kaggle, Google integration

**Steps**:
1. Upload `train-standalone.ipynb` to Google Drive
2. Open with Google Colab
3. Runtime → Change runtime type → GPU → T4
4. Run all cells
5. Download adapter when complete

### Option 3: Local Training (Advanced)

**Requirements**:
- NVIDIA GPU with 15GB+ VRAM (RTX 4090, A100, etc.)
- CUDA 11.8+
- 32GB+ system RAM

**Not recommended** unless you have appropriate hardware. The 8B model requires significant VRAM even with 4-bit quantization.

### Dataset Generation

The dataset is synthetically generated and deterministic:

```python
# In data/generate_dataset.py
random.seed(42)  # Ensures reproducible results
```

Expected output:
- Total samples: 360
- Train samples: 324
- Eval samples: 36
- Distribution: 73% tool invocations, 27% direct responses

### Training Parameters

```json
{
  "learning_rate": 2e-4,
  "num_epochs": 3,
  "batch_size": 4,
  "gradient_accumulation_steps": 2,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

These parameters are optimized for T4 GPU (16GB VRAM).

## Testing

### Interactive Testing (Kaggle/Colab)

Use `kaggle-demo.ipynb` to test the trained model:

1. Upload notebook to Kaggle
2. Add dataset:
   - Upload your `models/lora-adapter` folder as Kaggle dataset
   - Or use publicly shared adapter
3. Enable GPU T4 x2
4. Run all cells

**

What to Test**:
- Comparison cell (Section 5): Base vs fine-tuned model
- Tool execution: add_task, list_tasks, summarize_tasks
- Conversational queries: Proper use of `<final>` tag
- Edge cases: Ambiguous queries, multiple tools

### Local Testing (No Training)

Test core components without GPU:

```bash
python test_agent.py
```

This verifies:
- Tool implementations
- Parser logic
- Executor functionality
- Schema validation

### API Testing (Requires GPU)

If you have a GPU with 6GB+ VRAM:

```bash
# Start server
python serving/app.py

# In another terminal
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Add a task to test the API"}'
```

## Development Workflow

### Adding New Tools

1. **Define tool function** in `agent/tools.py`:
```python
def new_tool(param1: str) -> Dict:
    # Implementation
    return {"status": "success", "result": ...}
```

2. **Add to executor** in `agent/executor.py`:
```python
elif "new_tool" in action_str:
    # Parse parameters
    # Call function
    return new_tool(param)
```

3. **Generate training data** in `data/generate_dataset.py`:
```python
# Add samples for new tool
for _ in range(60):
    samples.append({
        "instruction": "Use new tool",
        "analysis": "User wants to use new tool",
        "action": "new_tool(param=\"value\")"
    })
```

4. **Retrain model** with updated dataset

### Project Structure Principles

- **agent/**: Core logic, should be framework-agnostic
- **serving/**: Deployment code, framework-specific
- **training/**: Scripts for model fine-tuning
- **data/**: Dataset generation and management

### Code Standards

**Style**:
- Follow PEP 8 for Python code
- Use type hints where applicable
- Keep functions focused and small

**Documentation**:
- Docstrings for public functions
- Inline comments for complex logic
- Update README when adding features

**Testing**:
- Test new tools locally before training
- Verify parsing logic handles new formats
- Run full test suite before commits

## Common Issues

### Training

**Issue**: Out of memory during training
**Solution**: Reduce batch size or use gradient checkpointing

**Issue**: HuggingFace authentication failed
**Solution**: Verify token and LLaMA license acceptance

### Inference

**Issue**: Model too slow on CPU
**Solution**: Use Kag gle/Colab for inference, or quantize further

**Issue**: Parser fails on model output
**Solution**: Check if output follows `<tag>content</tag>` format exactly

## Versioning

This project follows semantic versioning:
- v1.0.0: Initial release with basic tools
- Future versions will add features incrementally

## Questions?

For questions or issues:
1. Check existing GitHub issues
2. Review documentation (README, QUICKSTART, API_EXAMPLES)
3. Open new issue with detailed description

## Acknowledgments

Thanks to all contributors and the open-source ML community for making projects like this possible.
