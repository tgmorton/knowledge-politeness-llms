# Query Generator Dockerfile
#
# Packages Python scripts for querying vLLM models in Kubernetes
# Supports all 4 experiments: Study 1&2, Experiment 1&2

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python3", "-m", "src.query_study1_exp1", "--help"]
