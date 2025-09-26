
#!/usr/bin/env python3
"""
LLM Quantization Script for TinyLlama-1.1B-Chat-v1.0
This script loads the model and benchmarks its performance for CPU inference.
"""

import os
import time
import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


class LLMQuantizer:
    """Class to handle LLM loading and benchmarking."""
    
    def __init__(self, model_name="KBlueLeaf/TIPO-100M"):
        self.model_name = model_name
        self.tokenizer = None
        self.models = {}
        self.benchmark_results = {}
        
    def load_tokenizer(self):
        """Load the tokenizer for the model."""
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded successfully.")
        
    def get_model_size(self, model_instance):
        """Calculate the size of the model in memory."""
        if model_instance:
            # Estimate model size by summing parameter sizes
            total_params = sum(p.numel() for p in model_instance.parameters())
            # Assuming float16 for parameters (2 bytes per parameter)
            # This is an approximation; actual memory usage might differ
            return total_params * 2  # in bytes
        return 0
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
        }
    
    def load_model(self):
        """Load the model for CPU inference."""
        print("Loading model for CPU inference...")
        start_time = time.time()
        
        # Clear GPU cache if available (though we're targeting CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32, # Use float32 for CPU for better compatibility
                device_map="cpu", # Explicitly set to CPU
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            memory_usage = self.get_memory_usage()
            model_size = self.get_model_size(model)
            
            self.models['cpu_model'] = model
            self.benchmark_results['cpu_model'] = {
                'load_time': load_time,
                'memory_usage': memory_usage,
                'model_size': model_size,
                'quantization_config': 'None (full precision on CPU)'
            }
            
            print(f"CPU model loaded in {load_time:.2f} seconds")
            return model
            
        except Exception as e:
            print(f"Error loading CPU model: {e}")
            return None
    
    def benchmark_inference(self, model, model_type, test_prompts=None):
        """Benchmark inference speed and quality for a model."""
        if test_prompts is None:
            test_prompts = [
                "What is artificial intelligence?",
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the benefits and risks of AI technology?",
                "How does natural language processing work?"
            ]
        
        print(f"Benchmarking {model_type} model inference...")
        
        inference_times = []
        responses = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"  Processing prompt {i+1}/{len(test_prompts)}")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cpu") for k, v in inputs.items()} # Ensure inputs are on CPU
            
            # Measure inference time
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append({
                    'prompt': prompt,
                    'response': response,
                    'inference_time': inference_time
                })
                
            except Exception as e:
                print(f"    Error during inference: {e}")
                inference_times.append(float('inf'))
                responses.append({
                    'prompt': prompt,
                    'response': f"Error: {e}",
                    'inference_time': float('inf')
                })
        
        # Calculate statistics
        valid_times = [t for t in inference_times if t != float('inf')]
        avg_inference_time = sum(valid_times) / len(valid_times) if valid_times else float('inf')
        
        self.benchmark_results[model_type].update({
            'avg_inference_time': avg_inference_time,
            'inference_times': inference_times,
            'responses': responses,
            'successful_inferences': len(valid_times),
            'total_prompts': len(test_prompts)
        })
        
        print(f"  Average inference time: {avg_inference_time:.2f} seconds")
        print(f"  Successful inferences: {len(valid_times)}/{len(test_prompts)}")
        
        return responses
    
    def save_benchmark_results(self, filename="benchmark_results.json"):
        """Save benchmark results to a JSON file."""
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for model_type, results in self.benchmark_results.items():
            serializable_results[model_type] = {}
            for key, value in results.items():
                if key == 'memory_usage':
                    serializable_results[model_type][key] = value
                elif isinstance(value, (int, float, str, list, dict)):
                    serializable_results[model_type][key] = value
                else:
                    serializable_results[model_type][key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for model_type, results in self.benchmark_results.items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Load Time: {results.get('load_time', 'N/A'):.2f} seconds")
            print(f"  Memory Usage (RSS): {results.get('memory_usage', {}).get('rss', 0) / 1024 / 1024:.2f} MB")
            print(f"  Model Size (Estimated): {results.get('model_size', 0) / 1024 / 1024:.2f} MB")
            print(f"  Average Inference Time: {results.get('avg_inference_time', 'N/A'):.2f} seconds")
            print(f"  Quantization: {results.get('quantization_config', 'None')}")
            print(f"  Successful Inferences: {results.get('successful_inferences', 0)}/{results.get('total_prompts', 0)}")
    
    def cleanup_model(self, model_type):
        """Clean up a model from memory."""
        if model_type in self.models:
            del self.models[model_type]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Cleaned up {model_type} model from memory")


def main():
    """Main function to run the model loading and benchmarking."""
    print("Starting LLM Loading and Benchmarking (CPU-only)")
    print("="*50)
    
    # Initialize quantizer
    quantizer = LLMQuantizer()
    
    # Load tokenizer
    quantizer.load_tokenizer()
    
    # Test prompts for benchmarking
    test_prompts = [
        "What is artificial intelligence?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits and risks of AI technology?",
        "How does natural language processing work?"
    ]
    
    # Load and benchmark CPU model
    print("\n" + "-"*50)
    print("BENCHMARKING CPU MODEL")
    print("-"*50)
    cpu_model = quantizer.load_model()
    if cpu_model:
        quantizer.benchmark_inference(cpu_model, 'cpu_model', test_prompts)
        quantizer.cleanup_model('cpu_model')
    
    # Save and print results
    quantizer.save_benchmark_results()
    quantizer.print_summary()
    
    print("\nBenchmarking completed!")


if __name__ == "__main__":
    main()

