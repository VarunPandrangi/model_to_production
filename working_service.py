from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="API for LLM inference on CPU",
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

# Use a guaranteed public model
model_name = "distilgpt2"  # This is a small model that's publicly available
tokenizer = None
model = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    try:
        # Set HF_HUB_DISABLE_SYMLINKS_WARNING to suppress warnings
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

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
        "message": "LLM Inference API",
        "model": model_name,
        "endpoints": {
            "POST /generate": "Generate text from a prompt",
            "GET /health": "Health check",
            "GET /model/info": "Get model information"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Model info endpoint
@app.get("/model/info")
def model_info(api_key: str = Depends(verify_api_key)):
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "name": model_name,
        "parameters": "~82M",
        "quantization": "None (full precision on CPU)",
        "device": "CPU"
    }

# Text generation endpoint
@app.post("/generate")
def generate_text(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        return {
            "text": generated_text,
            "generation_time": generation_time,
            "model_name": model_name
        }
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
