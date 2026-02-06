---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.1-8B-Instruct
- lora
- transformers
---

# LLaMA Task Agent — LoRA Adapter

LoRA adapter for a task management agent fine-tuned from LLaMA-3.1-8B-Instruct. Trained to produce structured tool-calling output with strict format compliance.

## Model Details

- **Developed by:** Aswani Sahoo
- **Model type:** Causal LM (LoRA adapter)
- **Language:** English
- **License:** MIT
- **Base model:** meta-llama/Llama-3.1-8B-Instruct

## Training

| Parameter | Value |
|-----------|-------|
| Method | LoRA |
| Rank | 16 |
| Alpha | 32 |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Dataset | 360 synthetic task management samples |
| Epochs | 3 |
| Hardware | Kaggle T4 GPU (~90 min) |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "models/lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

## Results

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Format compliance | ~45% | 100% |
| Tool selection accuracy | ~62% | ~94% |

## Links

- **Repository:** [AswaniSahoo/llama-task-agent](https://github.com/AswaniSahoo/llama-task-agent)
- **Demo:** [Kaggle Notebook](https://www.kaggle.com/code/aswanisahoo/llama-task-agent-demo)

### Framework versions

- PEFT 0.17.1
