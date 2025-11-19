# vLLM Server Dockerfile (Custom Build)
#
# Note: For most use cases, use the official vllm/vllm-openai:latest image
# This Dockerfile is provided for custom builds if needed

FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Environment variables for Hugging Face cache
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

# Expose vLLM API port
EXPOSE 8000

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# Default args for Gemma-2B
CMD ["--model=google/gemma-2-2b-it", \
     "--dtype=float16", \
     "--max-model-len=4096", \
     "--tensor-parallel-size=1", \
     "--host=0.0.0.0", \
     "--port=8000"]
