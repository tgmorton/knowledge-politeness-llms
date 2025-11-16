# Grace Project - Kubernetes-Based Experiment Architecture

## Executive Summary

This document outlines the architecture for running distributed LLM experiments on the National Research Platform (NRP) using Kubernetes. The system will support both raw response generation (Experiment 1 & 2) and structured output extraction (Study 1 & 2) across multiple open-source models.

## Architecture Principles

### 1. **Separation of Concerns**
- **Model Serving Layer**: Self-hosted transformer models in dedicated pods
- **Query Generation Layer**: Jobs that send prompts and collect responses
- **Structured Output Layer**: Post-processing with DeepSeek for data extraction
- **Storage Layer**: Persistent volumes for raw and processed data

### 2. **Scalability & Resource Optimization**
- One model per GPU-enabled pod for optimal performance
- Parallel job execution for different studies/models
- Resource quotas to prevent cluster saturation
- Auto-scaling for query generation workloads

### 3. **Data Integrity & Reproducibility**
- Immutable input data via ConfigMaps
- Versioned output data with timestamps
- Complete audit trail of model configurations
- Deterministic experiment execution

## System Components

### Component 1: Model Serving Infrastructure

**Purpose**: Host open-source transformer models with inference APIs

**Models to Deploy**:
- Gemma-2 (2B, 7B, 27B variants)
- Llama-3 (8B, 70B variants)
- OpenAI OSS models (GPT-2, etc.)
- DeepSeek-V2/V3 (for structured output)

**Technology Stack**:
- **vLLM**: High-throughput inference server with PagedAttention
- **Text Generation Inference (TGI)**: Alternative for certain models
- **FastAPI**: Wrapper for custom endpoints if needed

**Why vLLM?**
- 24x higher throughput than HuggingFace Transformers
- Continuous batching for efficient GPU utilization
- OpenAI-compatible API endpoints
- Supports probability/logprob extraction (critical for Experiment 2)

**Deployment Pattern**:
```
StatefulSet per model:
  - Single replica (can scale if needed)
  - GPU node affinity
  - Model weights downloaded at init (or mounted from PVC)
  - Service exposing port 8000
  - Liveness/readiness probes
```

### Component 2: Query Generation Jobs

**Purpose**: Generate responses from input CSVs using model serving endpoints

**For Study 1 (Experiment 1)**:
- Input: `study1.csv` (300 rows)
- Construct prompts from: story_setup, priorQ, speech, speechQ
- Query each model sequentially
- Collect raw text responses
- Output: `{model}_study1_raw_responses.csv`

**For Study 2 (Experiment 1)**:
- Input: `study2.csv` (2,424 rows)
- Construct prompts from: Scenario, Goal, State
- Query each model
- Collect "was/wasn't good/bad/terrible/amazing" style responses
- Output: `{model}_study2_raw_responses.csv`

**For Study 1 (Experiment 2)**:
- Same input as Experiment 1
- **For each trial, query 4 times (once per state 0, 1, 2, 3)**
- For each state, request logprobs for percentage tokens: "0%", "10%", "20%", ..., "100%"
- This creates a 4×11 probability matrix showing model's uncertainty about each state
- Request logprobs for tokens: "yes", "no" (for knowledge questions)
- Normalize probabilities using softmax
- Calculate mean, std, entropy for each state's distribution
- Output: `{model}_study1_probabilities.csv` (68 columns per row)

**For Study 2 (Experiment 2)**:
- Same input as Experiment 1
- Request logprobs for: "was"/"wasn't" and "good"/"bad"/"terrible"/"amazing"
- Calculate probability distribution over response space
- Output: `{model}_study2_probabilities.csv`

**Technology Stack**:
- Python 3.11+ with asyncio for concurrent requests
- `httpx` for async HTTP requests to vLLM endpoints
- `pandas` for CSV manipulation
- Kubernetes Job with parallelism configuration

### Component 3: Structured Output Extraction

**Purpose**: Convert raw LLM responses into structured CSV/JSON using DeepSeek

**For Experiment 1 Outputs**:
- Read raw response CSVs
- Construct extraction prompts with JSON schema
- Query DeepSeek API (or self-hosted DeepSeek)
- Validate output against schema
- Merge with original data
- Output: `{model}_study{1|2}_structured.csv`

**JSON Schema for Study 1**:
```json
{
  "Prior_A": "integer [0-3]",
  "0": "integer [0-100]",
  "1": "integer [0-100]",
  "2": "integer [0-100]",
  "3": "integer [0-100]",
  "Knowledge_A": "string [yes|no]"
}
```

**JSON Schema for Study 2**:
```json
{
  "Was/Wasn't": "string [was|wasn't]",
  "Assessment": "string [terrible|bad|good|amazing]",
  "response_text": "string"
}
```

**Technology Stack**:
- Python with `pydantic` for schema validation
- DeepSeek API client (or vLLM endpoint)
- Retry logic with exponential backoff
- Output validation and error logging

### Component 4: Storage Architecture

**Persistent Volume Claims (PVCs)**:

1. **Input Data PVC** (5GB, ReadOnlyMany)
   - Stores: study1.csv, study2.csv
   - Mounted by all query jobs

2. **Model Weights PVC** (500GB, ReadWriteOnce per model)
   - Stores: Downloaded model weights
   - Mounted by model serving pods
   - One PVC per model to avoid conflicts

3. **Output Data PVC** (50GB, ReadWriteMany)
   - Stores: All generated responses, structured outputs
   - Mounted by query jobs and extraction jobs
   - Organized by: `/{experiment}/{model}/{timestamp}/`

4. **Logs PVC** (10GB, ReadWriteMany)
   - Stores: Job logs, error reports, metadata
   - Useful for debugging and auditing

**Storage Class**: 
- Use NRP-provided storage class (likely Rook-Ceph or NFS)
- Request fast SSD for model weights
- Standard storage for data outputs

## Workflow Orchestration

### Execution Phases

**Phase 1: Infrastructure Setup**
1. Create namespace: `grace-experiments`
2. Apply resource quotas and limits
3. Create all PVCs
4. Deploy secrets for API keys (if using external APIs)

**Phase 2: Model Deployment**
1. Deploy model weight download init jobs (if not pre-populated)
2. Deploy StatefulSets for each model
3. Wait for all models to be ready (health checks)
4. Run smoke tests on each endpoint

**Phase 3: Experiment 1 Execution** (Raw Responses)
1. Deploy Study 1 query jobs (one per model) - parallel
2. Deploy Study 2 query jobs (one per model) - parallel
3. Monitor job completion
4. Validate output CSVs exist and are complete

**Phase 4: Structured Output Generation**
1. Deploy DeepSeek structured output jobs
2. Process all Experiment 1 raw outputs
3. Validate structured outputs
4. Generate summary statistics

**Phase 5: Experiment 2 Execution** (Probability Extraction)
1. Deploy Study 1 probability extraction jobs
2. Deploy Study 2 probability extraction jobs
3. Collect and validate probability distributions

**Phase 6: Data Collection**
1. Copy all outputs from PVC to local/external storage
2. Generate experiment manifest (metadata JSON)
3. Clean up resources (optional)

### Orchestration Tools

**Option A: Argo Workflows** (Recommended)
- DAG-based workflow definition
- Built-in retry and error handling
- Parameter passing between steps
- UI for monitoring
- Artifact management

**Option B: Kubernetes Jobs + Shell Scripts**
- Simpler, no additional tools
- Less visibility
- Manual dependency management

**Option C: Airflow on Kubernetes**
- Overkill for this use case
- More complex setup

**Recommendation**: Use Argo Workflows for its Kubernetes-native approach and excellent DAG visualization.

## Resource Requirements

### Per Model Serving Pod

| Model | GPU | CPU | Memory | Disk |
|-------|-----|-----|--------|------|
| Gemma-2B | 1x A100 (40GB) | 8 cores | 32GB | 10GB |
| Gemma-7B | 1x A100 (40GB) | 16 cores | 64GB | 30GB |
| Gemma-27B | 2x A100 (40GB) | 32 cores | 128GB | 60GB |
| Llama-8B | 1x A100 (40GB) | 16 cores | 64GB | 40GB |
| Llama-70B | 4x A100 (40GB) | 64 cores | 256GB | 150GB |
| DeepSeek-V2 | 2x A100 (80GB) | 32 cores | 128GB | 80GB |

### Per Query Job
- CPU: 4 cores
- Memory: 8GB
- No GPU required
- Parallelism: 10-50 concurrent requests

### Total Cluster Requirements
- **GPUs**: 10-15 A100 GPUs (depending on model selection)
- **CPU**: 200+ cores
- **Memory**: 800GB+ RAM
- **Storage**: 600GB persistent storage

## Security & Compliance

1. **API Keys**: Store in Kubernetes Secrets, never in code
2. **Network Policies**: Restrict pod-to-pod communication
3. **RBAC**: Minimal permissions for service accounts
4. **Data Privacy**: Ensure no PII in logs or error messages
5. **Model Licenses**: Verify commercial use rights for all models

## Monitoring & Observability

1. **Prometheus + Grafana**: Resource utilization metrics
2. **vLLM Metrics**: Request latency, throughput, queue depth
3. **Custom Metrics**: Rows processed, completion percentage
4. **Alerting**: Job failures, OOM events, GPU errors
5. **Logging**: Centralized with ELK or Loki

## Cost Optimization

1. **Spot Instances**: If NRP supports preemptible nodes
2. **Model Quantization**: Use 4-bit/8-bit quantized models
3. **Batch Size Tuning**: Maximize GPU utilization
4. **Auto-shutdown**: Stop model pods when experiments complete
5. **Shared Base Models**: Use LoRA adapters instead of full models where applicable

## Next Steps

1. Review Kubernetes architecture with NRP documentation
2. Create detailed Kubernetes manifests (see `02_KUBERNETES_SPECS.md`)
3. Develop Python scripts for query generation and extraction
4. Build Docker images with dependencies
5. Create Argo Workflow definitions
6. Set up monitoring and alerting
7. Run small-scale pilot test
8. Execute full experiments

## Open Questions & Decisions Needed

1. **Model Selection**: Which specific model variants to deploy?
2. **DeepSeek Hosting**: Self-host or use API? (API is simpler, self-host is cheaper at scale)
3. **Quantization**: Use fp16, int8, or int4 quantization?
4. **Batch Processing**: Row-by-row or batch prompts?
5. **Retry Logic**: How many retries for failed requests?
6. **Timeout Values**: Per-request timeout limits?
7. **Validation Rules**: What constitutes a "valid" response?
8. **NRP Specifics**: Storage class names, GPU node selectors, networking setup?
