"""Fine-tune LLaMA model using LoRA for task-oriented agent behavior."""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(config_path: str = "configs/agent_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset_file(file_path: str):
    """Load dataset from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_sample(sample: dict) -> str:
    """
    Format a training sample into instruction-response format.
    
    Expected format:
    - Tool invocation: analysis + action
    - Direct response: final
    """
    instruction = sample["instruction"]
    
    if "action" in sample:
        # Tool invocation format
        analysis = sample.get("analysis", "")
        action = sample["action"]
        response = f"<analysis>\n{analysis}\n</analysis>\n\n<action>\n{action}\n</action>"
    else:
        # Direct response format
        final = sample["final"]
        response = f"<final>\n{final}\n</final>"
    
    return {
        "instruction": instruction,
        "response": response
    }


def prepare_dataset(data_path: str, tokenizer):
    """Prepare dataset for training."""
    # Load raw data
    raw_data = load_dataset_file(data_path)
    
    # Format samples
    formatted_data = [format_sample(sample) for sample in raw_data]
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Apply chat template
    def apply_template(example):
        messages = [
            {
                "role": "system",
                "content": "You are a task management assistant. When a user needs a tool, respond with <analysis> and <action>. For conversation, use <final>."
            },
            {
                "role": "user",
                "content": example["instruction"]
            },
            {
                "role": "assistant",
                "content": example["response"]
            }
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}
    
    dataset = dataset.map(apply_template, remove_columns=["instruction", "response"])
    
    # Tokenize
    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = dataset.map(tokenize, remove_columns=["text"])
    
    return dataset


def main():
    """Main training function."""
    print("Loading configuration...")
    config = load_config()
    
    model_name = config["model"]["base_model"]
    training_config = config["training"]
    
    print(f"Base model: {model_name}")
    print(f"LoRA rank: {training_config['lora_r']}")
    print()
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        target_modules=training_config["target_modules"],
        lora_dropout=training_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print()
    
    # Load and prepare datasets
    print("Loading training dataset...")
    train_dataset = prepare_dataset("data/train.json", tokenizer)
    print(f"Training samples: {len(train_dataset)}")
    
    print("Loading evaluation dataset...")
    eval_dataset = prepare_dataset("data/eval.json", tokenizer)
    print(f"Evaluation samples: {len(eval_dataset)}")
    print()
    
    # Training arguments
    output_dir = "models/lora-adapter"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("Starting training...")
    print("=" * 50)
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nYou can now use this adapter for inference.")


if __name__ == "__main__":
    main()
