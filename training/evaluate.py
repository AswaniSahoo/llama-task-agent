"""Evaluate base model vs fine-tuned model performance."""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re


def load_config(config_path: str = "configs/agent_config.json"):
    """Load configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_eval_data(eval_path: str = "data/eval.json"):
    """Load evaluation dataset."""
    with open(eval_path, "r") as f:
        return json.load(f)


def setup_model(model_name: str, adapter_path: str = None):
    """Load model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256):
    """Generate response for a given instruction."""
    system_prompt = "You are a task management assistant. When a user needs a tool, respond with <analysis> and <action>. For conversation, use <final>."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_expected(sample: Dict[str, Any]) -> tuple:
    """Extract expected output type and content."""
    if "action" in sample:
        return "tool", sample["action"]
    else:
        return "direct", sample["final"]


def parse_model_output(output: str) -> tuple:
    """Parse model output to determine type."""
    # Check for action
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    if action_match:
        return "tool", action_match.group(1).strip()
    
    # Check for final
    final_match = re.search(r'<final>(.*?)</final>', output, re.DOTALL)
    if final_match:
        return "direct", final_match.group(1).strip()
    
    # Invalid format
    return "invalid", output


def evaluate_model(model, tokenizer, eval_data: List[Dict], model_name_label: str):
    """Evaluate model on dataset."""
    print(f"\nEvaluating {model_name_label}...")
    print("=" * 60)
    
    results = {
        "correct_tool_selection": 0,
        "incorrect_tool_selection": 0,
        "format_errors": 0,
        "total": len(eval_data),
    }
    
    errors = []
    
    for i, sample in enumerate(eval_data):
        instruction = sample["instruction"]
        expected_type, expected_content = extract_expected(sample)
        
        # Generate response
        try:
            output = generate_response(model, tokenizer, instruction)
            predicted_type, predicted_content = parse_model_output(output)
            
            # Evaluate
            if predicted_type == "invalid":
                results["format_errors"] += 1
                errors.append({
                    "instruction": instruction,
                    "expected_type": expected_type,
                    "output": output,
                    "error": "Invalid format"
                })
            elif predicted_type == expected_type:
                results["correct_tool_selection"] += 1
            else:
                results["incorrect_tool_selection"] += 1
                errors.append({
                    "instruction": instruction,
                    "expected_type": expected_type,
                    "predicted_type": predicted_type,
                    "output": output
                })
                
        except Exception as e:
            results["format_errors"] += 1
            errors.append({
                "instruction": instruction,
                "error": str(e)
            })
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(eval_data)} samples...")
    
    # Calculate metrics
    accuracy = results["correct_tool_selection"] / results["total"] * 100
    format_compliance = (1 - results["format_errors"] / results["total"]) * 100
    
    print(f"\nResults for {model_name_label}:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Format Compliance: {format_compliance:.2f}%")
    print(f"  Correct: {results['correct_tool_selection']}")
    print(f"  Incorrect: {results['incorrect_tool_selection']}")
    print(f"  Format Errors: {results['format_errors']}")
    
    return {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "results": results,
        "errors": errors[:10]  # Save first 10 errors for analysis
    }


def main():
    """Run evaluation experiment."""
    print("Loading configuration...")
    config = load_config()
    model_name = config["model"]["base_model"]
    adapter_path = config["model"]["adapter_path"]
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_eval_data()
    print(f"Evaluation samples: {len(eval_data)}")
    
    # Evaluate base model
    base_model, base_tokenizer = setup_model(model_name)
    base_results = evaluate_model(base_model, base_tokenizer, eval_data, "Base Model (Prompt Only)")
    
    # Clean up
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    if Path(adapter_path).exists():
        ft_model, ft_tokenizer = setup_model(model_name, adapter_path)
        ft_results = evaluate_model(ft_model, ft_tokenizer, eval_data, "Fine-Tuned Model")
        
        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Base Model Accuracy: {base_results['accuracy']:.2f}%")
        print(f"Fine-Tuned Model Accuracy: {ft_results['accuracy']:.2f}%")
        print(f"Improvement: {ft_results['accuracy'] - base_results['accuracy']:.2f}%")
        print()
        print(f"Base Model Format Compliance: {base_results['format_compliance']:.2f}%")
        print(f"Fine-Tuned Model Format Compliance: {ft_results['format_compliance']:.2f}%")
        print(f"Improvement: {ft_results['format_compliance'] - base_results['format_compliance']:.2f}%")
        
        # Save results
        results_file = "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "base_model": base_results,
                "fine_tuned_model": ft_results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    else:
        print(f"\nWarning: Fine-tuned model not found at {adapter_path}")
        print("Train the model first using training/finetune_lora.py")


if __name__ == "__main__":
    main()
