from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="API for LLM inference on resource-constrained environments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Valid API keys
API_KEYS = ["demo-key-12345", "hackathon-key-67890", "test-key-abcdef"]

# Request model
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

# Response model
class GenerateResponse(BaseModel):
    text: str
    generation_time: float
    model_name: str

# API key verification
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "LLM Inference API for Resource-Constrained Environments",
        "model": "KBlueLeaf/TIPO-100M (simulated)",
        "endpoints": {
            "POST /generate": "Generate text from a prompt",
            "GET /health": "Health check",
            "GET /model/info": "Get model information",
            "GET /metrics": "Prometheus metrics"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return {
        "llm_requests_total": 42,
        "llm_errors_total": 2,
        "llm_request_duration_seconds_avg": 0.789,
        "llm_model_memory_usage_mb": 700.83,
        "llm_uptime_seconds": 3600
    }

# Model info endpoint
@app.get("/model/info")
def model_info(api_key: str = Depends(verify_api_key)):
    return {
        "name": "KBlueLeaf/TIPO-100M (simulated)",
        "parameters": "~100M",
        "quantization": "None (full precision on CPU)",
        "device": "CPU",
        "memory_usage_mb": 700.83,
        "load_time_seconds": 3.59,
        "avg_inference_time_seconds": 7.89
    }

# Text generation endpoint
@app.post("/generate")
def generate_text(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(1)
    
    # Simulate generated text based on prompt
    prompt = request.prompt.strip()
    
    if "quantum" in prompt.lower():
        generated_text = prompt + " is a field of study that uses quantum mechanics principles like superposition and entanglement to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously, potentially solving certain problems much faster."
    elif "artificial intelligence" in prompt.lower():
        generated_text = prompt + " refers to the simulation of human intelligence in machines programmed to think and learn like humans. It encompasses various technologies including machine learning, natural language processing, computer vision, and robotics. AI systems can analyze data, recognize patterns, make decisions, and improve over time through experience."
    elif "optimization" in prompt.lower():
        generated_text = prompt + " involves finding the best solution from all feasible solutions. In the context of LLMs, optimization techniques like quantization reduce model size and memory footprint while maintaining performance. This is crucial for deploying large models on resource-constrained hardware."
    else:
        generated_text = prompt + " is an interesting topic that has many applications across various fields. Researchers continue to explore new approaches and methodologies to advance our understanding in this area. Recent developments have shown promising results that could lead to significant breakthroughs in the near future."
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    return {
        "text": generated_text,
        "generation_time": generation_time,
        "model_name": "KBlueLeaf/TIPO-100M (simulated)"
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
