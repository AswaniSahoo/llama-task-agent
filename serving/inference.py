"""Inference engine for fine-tuned task agent."""

import json
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class TaskAgentInference:
    """Inference wrapper for the fine-tuned task agent."""
    
    def __init__(self, config_path: str = "configs/agent_config.json"):
        """Initialize the inference engine."""
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.system_prompt = (
            "You are a task management assistant. "
            "When a user needs a tool, respond with <analysis> and <action>. "
            "For conversation, use <final>."
        )
    
    def load_config(self, config_path: str):
        """Load configuration."""
        with open(config_path, "r") as f:
            return json.load(f)
    
    def load_model(self):
        """Load the fine-tuned model."""
        model_name = self.config["model"]["base_model"]
        adapter_path = self.config["model"]["adapter_path"]
        
        print(f"Loading model: {model_name}...")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter: {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        print("Model loaded successfully!")
    
    def generate(self, user_message: str, max_new_tokens: int = 256) -> str:
        """
        Generate a response for the user message.
        
        Args:
            user_message: User's input message
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


if __name__ == "__main__":
    # Test the inference engine
    print("Initializing inference engine...")
    agent = TaskAgentInference()
    agent.load_model()
    
    print("\nTesting inference...")
    test_queries = [
        "Add a task to buy groceries tomorrow",
        "What tasks do I have?",
        "Hello, how are you?",
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = agent.generate(query)
        print(f"Agent: {response}")
