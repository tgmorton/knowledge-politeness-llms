# Query Generator Dockerfile
#
# Packages Python scripts for querying vLLM models in Kubernetes
# Supports all 4 experiments: Study 1&2, Experiment 1&2
#
# OPTIMIZATION: Uses pre-built PyTorch base image to avoid downloading
# 5GB+ of NVIDIA CUDA libraries during build. This dramatically reduces:
# - Build time: ~4 min -> ~30 sec
# - Image push time: ~85 min -> ~2 min
# - Final image size: ~8GB -> ~3GB (shared base layers cached)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: PyTorch base already has many dependencies, just add what's missing
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (from docker/query-generator/)
COPY docker/query-generator/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (including utils modules)
COPY src/ /app/src/

# Copy input data (CSV files included in image)
COPY data/ /app/data/

# Create output directory
RUN mkdir -p /app/outputs

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python3", "-m", "src.query_study1_exp1", "--help"]
