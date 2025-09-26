# Optimized LLM Inference for Resource-Constrained Environments

## Hackathon Submission by Team OptimizeAI

This repository contains our hackathon submission for optimizing Large Language Model (LLM) inference in resource-constrained environments, specifically targeting systems with limited GPU capabilities or CPU-only configurations.

## Project Overview

Our solution addresses the challenge of deploying LLMs on hardware with limited resources, such as the AMD Ryzen 9 6900HS system specified in the hackathon requirements. We've developed a comprehensive solution that includes:

1. A production-ready FastAPI service for LLM inference
2. A React-based monitoring dashboard
3. Comprehensive benchmarking and documentation

## Key Features

- **Optimized Model Selection**: We selected `KBlueLeaf/TIPO-100M`, a small, open-access LLM that can run efficiently on CPU-only hardware.
- **Production-Ready API**: Our FastAPI service includes security features (API key authentication), rate limiting, and comprehensive observability.
- **Real-time Monitoring**: The React dashboard provides visualization of key performance metrics.
- **Containerization**: Docker and Docker Compose for easy deployment and scaling.
- **Comprehensive Documentation**: Detailed research paper and benchmarking results.

## Repository Structure

```
hackathon_submission/
├── Dockerfile                  # Container definition for the LLM service
├── docker-compose.yml          # Multi-container orchestration
├── llm_service.py              # FastAPI service for LLM inference
├── quantize_model.py           # Benchmarking and quantization script
├── requirements.txt            # Python dependencies
├── test_client.py              # Test client for the API
├── prometheus.yml              # Prometheus configuration
├── research_paper.md           # Detailed research paper
├── architecture_diagram.png    # Solution architecture diagram
├── benchmarking_process.png    # Benchmarking process diagram
├── visualizations/             # Generated visualizations
│   ├── benchmark_results.png
│   └── benchmark_table.png
├── llm-dashboard/              # React monitoring dashboard
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   └── index.css
│   └── package.json
└── presentation/               # Presentation slides
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 16+ (for dashboard development)

### Installation & Deployment

1. Clone this repository:
   ```bash
   git clone https://github.com/optimizeai/llm-optimization-hackathon.git
   cd llm-optimization-hackathon
   ```

2. Deploy with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access the services:
   - LLM API: http://localhost:8000
   - Monitoring Dashboard: http://localhost:3000
   - Prometheus: http://localhost:9090

### API Usage

```bash
curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer hackathon-key-67890" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_tokens": 100}'
```

## Benchmarking Results

Our benchmarking shows that the `KBlueLeaf/TIPO-100M` model can be successfully loaded and run on a CPU-only system with 16GB RAM:

| Metric                   | Value                               |
| :----------------------- | :---------------------------------- |
| Model Name               | KBlueLeaf/TIPO-100M                 |
| Quantization             | None (full precision on CPU)        |
| Load Time                | 3.59 seconds                        |
| Memory Usage (RSS)       | 700.83 MB                           |
| Model Size (Estimated)   | 191.46 MB                           |
| Average Inference Time   | 7.89 seconds                        |
| Successful Inferences    | 5/5                                 |

## Future Work

- Explore CPU-specific quantization techniques (e.g., ONNX Runtime)
- Implement model distillation to create even smaller, task-specific models
- Enhance caching strategies to reduce redundant computations
- Develop adaptive inference based on available system resources
- Explore AMD ROCm support for GPU acceleration on AMD hardware

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing access to open-source LLMs
- FastAPI for the high-performance web framework
- React for the frontend dashboard framework
