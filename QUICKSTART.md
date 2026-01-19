# Quick Start Guide

Get started with the LLaMA Task Agent in under 10 minutes.

## Choose Your Path

### Path 1: Test Locally (No GPU, 5 min)

Test core components without training:

```bash
# Setup
uv venv && .venv\Scripts\activate  # Windows
uv pip install -e .

# Generate dataset
python data/generate_dataset.py

# Test agent
python test_agent.py
```

**Result**: Verify tools, parser, and executor work correctly.

---

### Path 2: Train on Kaggle (Free GPU, 90 min)

**Prerequisites**:
- Kaggle account (free)
- HuggingFace account and LLaMA license

**Steps**:
1. Go to kaggle.com → Code → New Notebook
2. Settings → Accelerator → GPU T4 x2
3. Settings → Internet → ON
4. Upload `train-standalone.ipynb`
5. Run all cells
6. Download adapter from `models/lora-adapter/`

**HuggingFace Setup**:
- Token: https://huggingface.co/settings/tokens
- License: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

### Path 3: Test Trained Model (Kaggle, 10 min)

**Prerequisites**: Trained adapter (from Path 2 or provided)

**Steps**:
1. Upload `kaggle-demo.ipynb` to Kaggle
2. Settings → GPU T4 x2
3. Add Data → Upload `models/lora-adapter` as dataset
4. Run all cells

**Result**: See base vs fine-tuned comparison and live tool execution.

---

## Expected Outputs

### Local Testing (`test_agent.py`)
```
Testing Agent Tools
============================================================
1. Testing add_task...
Result: {'status': 'success', 'message': "Task added: 'Buy groceries'..."}

ALL TESTS PASSED!
```

### Training (`train-standalone.ipynb`)
```
Epoch 1/3: 100%
Epoch 2/3: 100%
Epoch 3/3: 100%
Model saved to models/lora-adapter/
```

### Demo (`kaggle-demo.ipynb`)
```
Base Model - Has <action> tag: True/False
Fine-Tuned - Has <action> tag: True

EXECUTION RESULT:
{'status': 'success', 'task_count': 1}
```

## Troubleshooting

### Import Errors
Ensure UV environment is activated: `.venv\Scripts\activate`

### GPU Not Available (Kaggle)
Check Settings → Accelerator → GPU T4 x2 is selected

### HuggingFace Authentication
Run login cell with your token from https://huggingface.co/settings/tokens

### Dataset Path Issues (Kaggle)
Update dataset path:
```python
dataset_path = '/kaggle/input/YOUR-DATASET-NAME'
```

## Next Steps

After quick start:
1. Review `README.md` for full documentation
2. Check `API_EXAMPLES.md` for API usage
3. See `CONTRIBUTING.md` for development guidelines

## Support

For detailed instructions, see:
- Training: Section in README
- API Deployment: `API_EXAMPLES.md`
- Contributing: `CONTRIBUTING.md`
