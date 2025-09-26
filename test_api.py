import requests
import time
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "hackathon-key-67890"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test prompts
TEST_PROMPTS = [
    "Explain quantum computing",
    "What is artificial intelligence?",
    "How does model quantization work?",
    "Describe the benefits of FastAPI",
    "What are the challenges of deploying LLMs on resource-constrained hardware?"
]

def test_health( ):
    """Test the health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    response = requests.get(f"{API_URL}/model/info", headers=HEADERS)
    print(f"Model info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_metrics():
    """Test the metrics endpoint"""
    response = requests.get(f"{API_URL}/metrics")
    print(f"Metrics: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_generate():
    """Test the generate endpoint with multiple prompts"""
    results = []
    
    for prompt in TEST_PROMPTS:
        payload = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/generate", headers=HEADERS, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result["request_time"] = end_time - start_time
            result["prompt"] = prompt
            results.append(result)
            
            print(f"Generated text for prompt: '{prompt}'")
            print(f"Time: {result['generation_time']:.2f}s (API) / {result['request_time']:.2f}s (total)")
            print(f"Text: {result['text'][:100]}...\n")
        else:
            print(f"Error for prompt '{prompt}': {response.status_code}")
            print(response.text)
    
    return results

def visualize_results(results):
    """Create visualizations from the test results"""
    if not results:
        print("No results to visualize")
        return
    
    # Extract data
    prompts = [r["prompt"][:20] + "..." for r in results]
    api_times = [r["generation_time"] for r in results]
    total_times = [r["request_time"] for r in results]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    x = np.arange(len(prompts))
    width = 0.35
    
    plt.bar(x - width/2, api_times, width, label='API Processing Time')
    plt.bar(x + width/2, total_times, width, label='Total Request Time')
    
    plt.xlabel('Prompts')
    plt.ylabel('Time (seconds)')
    plt.title('LLM Inference Performance')
    plt.xticks(x, prompts, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('inference_performance.png')
    print("Visualization saved as 'inference_performance.png'")

def run_all_tests():
    """Run all tests and visualize results"""
    print("Starting API tests...\n")
    
    health_ok = test_health()
    print("\n" + "-"*50 + "\n")
    
    model_ok = test_model_info()
    print("\n" + "-"*50 + "\n")
    
    metrics_ok = test_metrics()
    print("\n" + "-"*50 + "\n")
    
    if health_ok and model_ok and metrics_ok:
        results = test_generate()
        if results:
            visualize_results(results)
    
    print("\nTests completed!")

if __name__ == "__main__":
    run_all_tests()
