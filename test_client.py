#!/usr/bin/env python3
"""
Test client for the LLM Inference Service
Demonstrates API functionality and performance testing.
"""

import requests
import json
import time
from typing import List, Dict, Any


class LLMClient:
    """Client for interacting with the LLM Inference Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "demo-key-12345"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the service."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        response = requests.get(f"{self.base_url}/model/info", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using the LLM."""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        response = requests.get(f"{self.base_url}/metrics")
        return response.text
    
    def benchmark_performance(self, prompts: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Benchmark the performance of the service."""
        results = {
            "total_requests": len(prompts) * iterations,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0,
            "average_response_time": 0,
            "responses": []
        }
        
        start_time = time.time()
        
        for iteration in range(iterations):
            print(f"Running iteration {iteration + 1}/{iterations}")
            
            for i, prompt in enumerate(prompts):
                print(f"  Processing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")
                
                request_start = time.time()
                response = self.generate_text(prompt)
                request_time = time.time() - request_start
                
                if "error" not in response:
                    results["successful_requests"] += 1
                    results["responses"].append({
                        "prompt": prompt,
                        "response": response.get("generated_text", ""),
                        "generation_time": response.get("generation_time", 0),
                        "request_time": request_time,
                        "iteration": iteration + 1
                    })
                else:
                    results["failed_requests"] += 1
                    results["responses"].append({
                        "prompt": prompt,
                        "error": response.get("error", "Unknown error"),
                        "request_time": request_time,
                        "iteration": iteration + 1
                    })
                
                # Small delay between requests
                time.sleep(0.5)
        
        results["total_time"] = time.time() - start_time
        if results["successful_requests"] > 0:
            total_generation_time = sum(
                r.get("generation_time", 0) for r in results["responses"] 
                if "generation_time" in r
            )
            results["average_response_time"] = total_generation_time / results["successful_requests"]
        
        return results


def main():
    """Main function to test the LLM service."""
    print("LLM Inference Service Test Client")
    print("=" * 50)
    
    # Initialize client
    client = LLMClient()
    
    # Test health check
    print("\n1. Health Check:")
    try:
        health = client.health_check()
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Model Loaded: {health.get('model_loaded', False)}")
        print(f"   Memory Usage: {health.get('memory_usage', {}).get('rss_mb', 0):.2f} MB")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Test model info
    print("\n2. Model Information:")
    try:
        model_info = client.get_model_info()
        if "error" not in model_info:
            print(f"   Model: {model_info.get('model_name', 'unknown')}")
            print(f"   Quantization: {model_info.get('quantization', 'unknown')}")
            print(f"   Parameters: {model_info.get('parameters', 'unknown')}")
            print(f"   Device: {model_info.get('device', 'unknown')}")
        else:
            print(f"   Error: {model_info['error']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test text generation
    print("\n3. Text Generation Test:")
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "Write a short poem about technology.",
        "What are the benefits of renewable energy?",
        "How does blockchain technology work?"
    ]
    
    for i, prompt in enumerate(test_prompts[:2]):  # Test first 2 prompts
        print(f"\n   Test {i + 1}: {prompt}")
        try:
            result = client.generate_text(prompt, max_tokens=50)
            if "error" not in result:
                print(f"   Response: {result['generated_text'][:100]}...")
                print(f"   Generation Time: {result['generation_time']:.2f}s")
            else:
                print(f"   Error: {result['error']}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Performance benchmark
    print("\n4. Performance Benchmark:")
    try:
        benchmark_results = client.benchmark_performance(test_prompts[:3], iterations=2)
        print(f"   Total Requests: {benchmark_results['total_requests']}")
        print(f"   Successful: {benchmark_results['successful_requests']}")
        print(f"   Failed: {benchmark_results['failed_requests']}")
        print(f"   Total Time: {benchmark_results['total_time']:.2f}s")
        print(f"   Average Response Time: {benchmark_results['average_response_time']:.2f}s")
        
        # Save detailed results
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print("   Detailed results saved to benchmark_results.json")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test metrics endpoint
    print("\n5. Metrics Test:")
    try:
        metrics = client.get_metrics()
        metrics_lines = metrics.split('\n')
        relevant_metrics = [line for line in metrics_lines if 'llm_' in line and not line.startswith('#')]
        print(f"   Found {len(relevant_metrics)} LLM-specific metrics")
        for metric in relevant_metrics[:5]:  # Show first 5 metrics
            print(f"   {metric}")
        if len(relevant_metrics) > 5:
            print(f"   ... and {len(relevant_metrics) - 5} more")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
