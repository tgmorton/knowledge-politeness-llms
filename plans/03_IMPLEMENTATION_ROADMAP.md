# Implementation Roadmap

## Overview

This roadmap breaks down the implementation into manageable phases with clear deliverables, dependencies, and success criteria.

---

## Phase 0: Preparation & Setup (Week 1)

### Objectives
- Set up development environment
- Gather NRP credentials and documentation
- Validate access to models and resources
- Set up version control and project structure

### Tasks

#### 0.1: Development Environment
- [ ] Install kubectl, helm, and k9s
- [ ] Configure kubectl context for NRP cluster
- [ ] Install Docker Desktop or Podman
- [ ] Set up Python 3.11+ virtual environment
- [ ] Install dependencies: vllm, transformers, httpx, pandas, pydantic

#### 0.2: NRP Cluster Access
- [ ] Obtain NRP credentials and kubeconfig
- [ ] Test cluster connectivity: `kubectl cluster-info`
- [ ] Identify available GPU node types and quotas
- [ ] Document storage classes and network policies (see `docs/NRP_STORAGE_GUIDE.md`)
- [ ] **Join Matrix chat**: https://matrix.to/#/#nrp:matrix.org
- [ ] **Request exception for Deployments >2 weeks** (model servers)
- [ ] **Request exception for >2 GPUs per pod** (Llama-70B needs 4 GPUs)
- [ ] Request GPU quota increase if needed
- [ ] Review `docs/NRP_CLUSTER_GUIDE.md` for cluster policies

#### 0.3: Model Access
- [ ] Create HuggingFace account and get API token
- [ ] Test downloading models locally: Gemma-2B, Llama-8B
- [ ] Verify model licenses for research use
- [ ] Decide on DeepSeek approach (API vs self-hosted)

#### 0.4: Project Structure
```
10-GraceProject/
├── kubernetes/
│   ├── base/                    # Base manifests
│   ├── models/                  # Model-specific configs
│   ├── jobs/                    # Job templates
│   └── kustomization.yaml       # Kustomize config
├── docker/
│   ├── query-generator/         # Query job image
│   ├── probability-extractor/   # Probability extraction image
│   └── structured-extractor/    # DeepSeek extraction image
├── src/
│   ├── query_study1_exp1.py
│   ├── query_study2_exp1.py
│   ├── extract_probabilities_study1.py
│   ├── extract_probabilities_study2.py
│   ├── extract_structured_output.py
│   ├── schemas/                 # JSON schemas
│   └── utils/                   # Shared utilities
├── tests/
│   ├── test_query_generation.py
│   ├── test_probability_extraction.py
│   └── test_structured_output.py
├── experiments/
│   ├── experiment1/             # Exp 1 configs
│   └── experiment2/             # Exp 2 configs
└── plans/                       # This folder
```

#### 0.5: Version Control
- [ ] Initialize git repo (if not already done)
- [ ] Create .gitignore (exclude secrets, large files, outputs)
- [ ] Set up branch strategy (main, dev, feature/*)
- [ ] Create README with setup instructions

### Deliverables
- ✅ Working kubectl access to NRP
- ✅ Local development environment
- ✅ Project structure scaffolding
- ✅ Model download tested locally

### Success Criteria
- Can execute `kubectl get nodes` on NRP cluster
- Can download and load a model with vLLM locally
- Project structure created and committed

---

## Phase 1: Docker Image Development (Week 2)

### Objectives
- Build containerized applications for all job types
- Test images locally before deploying to cluster
- Optimize image sizes and build times

### Tasks

#### 1.1: Base Image Setup
Create a common base image with shared dependencies:

```dockerfile
# docker/base/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
```

#### 1.2: Query Generator Image
- [ ] Create Dockerfile for query generation jobs
- [ ] Implement `query_study1_exp1.py` script
- [ ] Implement `query_study2_exp1.py` script
- [ ] Add prompt construction logic
- [ ] Add retry and error handling
- [ ] Test with local vLLM server

**Key Script Features**:
```python
# Pseudo-code structure
class QueryGenerator:
    def __init__(self, model_endpoint, input_csv, output_csv):
        self.endpoint = model_endpoint
        self.df = pd.read_csv(input_csv)
        self.output = output_csv
    
    def construct_prompt_study1(self, row):
        # Build prompt from story_setup, priorQ, speach, speachQ
        pass
    
    async def query_model(self, prompt, max_retries=3):
        # Async HTTP request with retry logic
        pass
    
    async def process_batch(self, rows):
        # Parallel requests with rate limiting
        pass
    
    def run(self):
        # Main execution loop with checkpointing
        pass
```

#### 1.3: Probability Extractor Image
- [ ] Create Dockerfile for probability extraction
- [ ] Implement `extract_probabilities_study1.py`
- [ ] Implement `extract_probabilities_study2.py`
- [ ] Add logprobs extraction and normalization
- [ ] Handle token-level probability calculation
- [ ] Test with vLLM's logprobs API

**Key Features**:
```python
class ProbabilityExtractor:
    def extract_token_probabilities(self, prompt, tokens):
        # Request logprobs for specific tokens (0,1,2,3,yes,no)
        # Normalize using softmax
        return probabilities
    
    def process_study1(self, row):
        # Extract probabilities for states 0-3
        # Extract yes/no for knowledge question
        pass
```

#### 1.4: Structured Output Extractor Image
- [ ] Create Dockerfile for DeepSeek extraction
- [ ] Implement `extract_structured_output.py`
- [ ] Define JSON schemas for Study 1 and Study 2
- [ ] Add Pydantic validation models
- [ ] Implement retry logic with exponential backoff
- [ ] Test with DeepSeek API or self-hosted model

**JSON Schema Examples**:
```json
// schemas/study1_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "Prior_A": {"type": "integer", "minimum": 0, "maximum": 3},
    "probability_0": {"type": "integer", "minimum": 0, "maximum": 100},
    "probability_1": {"type": "integer", "minimum": 0, "maximum": 100},
    "probability_2": {"type": "integer", "minimum": 0, "maximum": 100},
    "probability_3": {"type": "integer", "minimum": 0, "maximum": 100},
    "Knowledge_A": {"type": "string", "enum": ["yes", "no"]}
  },
  "required": ["Prior_A", "probability_0", "probability_1", "probability_2", "probability_3", "Knowledge_A"]
}
```

#### 1.5: Build and Test Locally
- [ ] Build all Docker images
- [ ] Start local vLLM server with Gemma-2B
- [ ] Run query generator against local server
- [ ] Validate output CSV structure
- [ ] Run structured extractor on sample data

#### 1.6: Push to Registry
- [ ] Set up Docker registry (NRP-provided or Docker Hub)
- [ ] Tag images with version numbers
- [ ] Push all images to registry
- [ ] Document image locations and tags

### Deliverables
- ✅ 3 Docker images built and tested
- ✅ Python scripts with unit tests
- ✅ JSON schemas defined
- ✅ Images pushed to registry

### Success Criteria
- All scripts run successfully against local vLLM server
- Output CSVs have correct structure and valid data
- Unit tests pass (>80% coverage)

---

## Phase 2: Kubernetes Deployment - Model Serving (Week 3)

### Objectives
- Deploy vLLM model servers to NRP cluster
- Validate model endpoints are working
- Optimize resource allocation

### Tasks

#### 2.1: Namespace and Storage Setup
- [ ] Create `grace-experiments` namespace
- [ ] Apply resource quotas
- [ ] Create PVCs for input data, model weights, outputs, logs
- [ ] Upload input CSVs to input-data PVC

**Upload Data**:
```bash
# Create a temporary pod to upload data
kubectl run data-uploader --image=busybox --restart=Never \
  -n grace-experiments \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"grace-input-data"}}],"containers":[{"name":"uploader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}'

kubectl cp data/study1.csv grace-experiments/data-uploader:/data/study1.csv
kubectl cp data/study2.csv grace-experiments/data-uploader:/data/study2.csv
kubectl delete pod data-uploader -n grace-experiments
```

#### 2.2: Secrets and Service Accounts
- [ ] Create HuggingFace token secret
- [ ] Create DeepSeek API key secret (if using API)
- [ ] Create service accounts with RBAC
- [ ] Test service account permissions

#### 2.3: Deploy First Model (Gemma-2B)
- [ ] Apply StatefulSet for Gemma-2B
- [ ] Apply Service for Gemma-2B
- [ ] Monitor init container downloading model weights
- [ ] Wait for pod to be ready (may take 10-20 minutes)
- [ ] Check logs for errors

**Monitor Deployment**:
```bash
kubectl get pods -n grace-experiments -w
kubectl logs -f vllm-gemma-2b-0 -n grace-experiments
kubectl describe pod vllm-gemma-2b-0 -n grace-experiments
```

#### 2.4: Test Model Endpoint
- [ ] Port-forward to model service
- [ ] Test /health endpoint
- [ ] Test /v1/models endpoint
- [ ] Test /v1/completions with sample prompt
- [ ] Test /v1/completions with logprobs parameter

```bash
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# Test completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2b",
    "prompt": "The capital of France is",
    "max_tokens": 5,
    "temperature": 0.7
  }'

# Test logprobs
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2b",
    "prompt": "The answer is",
    "max_tokens": 1,
    "logprobs": 5,
    "temperature": 1.0
  }'
```

#### 2.5: Deploy Additional Models
- [ ] Deploy Llama-8B (follow same process)
- [ ] Deploy other selected models
- [ ] Deploy DeepSeek (if self-hosting)
- [ ] Verify all model endpoints

#### 2.6: Performance Tuning
- [ ] Monitor GPU utilization with `nvidia-smi`
- [ ] Adjust `--gpu-memory-utilization` parameter
- [ ] Test different batch sizes
- [ ] Measure requests per second
- [ ] Document optimal configurations

### Deliverables
- ✅ All model servers deployed and running
- ✅ All endpoints tested and validated
- ✅ Performance benchmarks documented

### Success Criteria
- All models respond to health checks
- Sample prompts return valid completions
- Logprobs extraction works correctly
- GPU utilization > 70%

---

## Phase 3: Job Execution - Experiment 1 (Week 4)

### Objectives
- Execute raw response generation for Study 1 and Study 2
- Generate responses from all models
- Validate output data

### Tasks

#### 3.1: Deploy Study 1 Query Jobs
- [ ] Apply job manifests for all models
- [ ] Monitor job progress
- [ ] Check output CSVs being written
- [ ] Handle any job failures

```bash
# Deploy all Study 1 Exp 1 jobs in parallel
kubectl apply -f kubernetes/jobs/study1-exp1-gemma-2b.yaml
kubectl apply -f kubernetes/jobs/study1-exp1-llama-8b.yaml
# ... etc

# Monitor
kubectl get jobs -n grace-experiments -l study=study1,experiment=exp1 -w
```

#### 3.2: Deploy Study 2 Query Jobs
- [ ] Apply job manifests for all models
- [ ] Monitor job progress
- [ ] Validate outputs

#### 3.3: Monitor and Debug
- [ ] Check job logs for errors
- [ ] Verify row counts match input CSVs
- [ ] Spot-check sample responses for quality
- [ ] Re-run failed jobs if needed

```bash
# Check job status
kubectl get jobs -n grace-experiments

# View logs
kubectl logs job/study1-exp1-gemma-2b -n grace-experiments

# Debug failed jobs
kubectl describe job study1-exp1-gemma-2b -n grace-experiments
```

#### 3.4: Validate Output Data
- [ ] Download output CSVs to local machine
- [ ] Verify all rows processed
- [ ] Check for missing values
- [ ] Inspect sample responses manually
- [ ] Generate data quality report

```bash
# Download outputs
kubectl cp grace-experiments/data-downloader:/output ./outputs

# Or use a Job to tar and export
```

#### 3.5: Structured Output Extraction
- [ ] Wait for all raw response jobs to complete
- [ ] Deploy structured output extraction jobs
- [ ] Monitor DeepSeek API rate limits (if using API)
- [ ] Validate JSON parsing and schema compliance
- [ ] Merge structured data with original CSVs

### Deliverables
- ✅ Raw response CSVs for Study 1 (all models)
- ✅ Raw response CSVs for Study 2 (all models)
- ✅ Structured output CSVs for all responses
- ✅ Data quality report

### Success Criteria
- All jobs complete successfully (or < 1% failure rate)
- Output CSVs have expected row counts
- Structured output validation passes
- No missing data in critical columns

---

## Phase 4: Job Execution - Experiment 2 (Week 5)

### Objectives
- Execute probability extraction for Study 1 and Study 2
- Collect token-level logprobs
- Compute normalized probability distributions

### Tasks

#### 4.1: Deploy Study 1 Probability Jobs
- [ ] Apply probability extraction job manifests
- [ ] **For each trial, query 4 times (once per state)**
- [ ] For each state, extract P(0%), P(10%), P(20%), ..., P(100%)
- [ ] For knowledge question, extract P(yes), P(no)
- [ ] Normalize probabilities using softmax
- [ ] Calculate mean, std deviation, and entropy for each state distribution
- [ ] Write to output CSV (68 columns per row)

**Note**: Study 1 Experiment 2 requires **4× more API calls** than initially planned (4 queries per trial × 300 trials × N models = 1,200×N total queries). Budget approximately 4× longer execution time than Experiment 1.

#### 4.2: Deploy Study 2 Probability Jobs
- [ ] Apply probability extraction job manifests
- [ ] Extract P(was), P(wasn't)
- [ ] Extract P(terrible), P(bad), P(good), P(amazing)
- [ ] Compute joint probability distribution
- [ ] Write to output CSV

#### 4.3: Validate Probability Data
- [ ] Check that probabilities sum to 1.0 (within epsilon)
- [ ] Verify probability ranges [0, 1]
- [ ] Compare distributions across models
- [ ] Generate probability distribution plots

#### 4.4: Sanity Checks
- [ ] Compare Exp 1 raw responses to Exp 2 probabilities
- [ ] Check if argmax(P) matches raw response
- [ ] Identify discrepancies and investigate
- [ ] Document any unexpected patterns

### Deliverables
- ✅ Probability CSVs for Study 1 (all models)
- ✅ Probability CSVs for Study 2 (all models)
- ✅ Validation report
- ✅ Comparison analysis between Exp 1 and Exp 2

### Success Criteria
- All probability values are valid [0, 1]
- Probabilities sum to ~1.0
- High correlation between raw responses and argmax(P)

---

## Phase 5: Analysis & Visualization (Week 6)

### Objectives
- Replicate existing R analysis scripts
- Generate comparative visualizations across models
- Produce statistical summaries

### Tasks

#### 5.1: Adapt R Scripts
- [ ] Update Study1.R to accept multiple model inputs
- [ ] Update Study2.R to accept multiple model inputs
- [ ] Add model comparison visualizations
- [ ] Generate per-model statistics

#### 5.2: Comparative Analysis
- [ ] Compare probability distributions across models
- [ ] Run ANOVA with model as additional factor
- [ ] Identify model-specific biases or patterns
- [ ] Statistical tests for model differences

#### 5.3: Visualization
- [ ] Bar plots: probability distributions by model
- [ ] Heatmaps: model × condition matrices
- [ ] Line plots: Study 2 response patterns by model
- [ ] Correlation plots: human vs model responses

#### 5.4: Generate Report
- [ ] Executive summary
- [ ] Model performance comparison table
- [ ] Key findings and insights
- [ ] Recommendations for model selection
- [ ] Limitations and future work

### Deliverables
- ✅ Updated R scripts
- ✅ Publication-quality figures
- ✅ Statistical analysis report
- ✅ Model comparison summary

### Success Criteria
- All analyses run without errors
- Visualizations clearly show model differences
- Statistical tests properly documented
- Report is ready for paper draft

---

## Phase 6: Cleanup & Documentation (Week 7)

### Objectives
- Clean up cluster resources
- Document entire pipeline
- Create reproducibility guide
- Archive data and code

### Tasks

#### 6.1: Resource Cleanup
- [ ] Stop all model serving pods (or scale to 0)
- [ ] Delete completed jobs
- [ ] Archive output PVC data to external storage
- [ ] Delete PVCs if not needed for future runs
- [ ] Release GPU quota

#### 6.2: Documentation
- [ ] Complete README with full pipeline instructions
- [ ] Document all configuration options
- [ ] Create troubleshooting guide
- [ ] Add architecture diagrams
- [ ] Write Kubernetes deployment guide

#### 6.3: Code Quality
- [ ] Run linters (black, flake8, mypy)
- [ ] Add docstrings to all functions
- [ ] Achieve >80% test coverage
- [ ] Code review and refactoring

#### 6.4: Reproducibility Package
- [ ] Create `requirements.txt` and `environment.yml`
- [ ] Document exact model versions used
- [ ] Include all Kubernetes manifests
- [ ] Provide sample data and expected outputs
- [ ] Write step-by-step reproduction guide

#### 6.5: Archival
- [ ] Export all outputs to long-term storage
- [ ] Create Zenodo/OSF repository (if publishing data)
- [ ] Tag GitHub release
- [ ] Create Docker image manifest

### Deliverables
- ✅ Complete documentation
- ✅ Reproducibility package
- ✅ Archived data
- ✅ GitHub release

### Success Criteria
- Another researcher can reproduce experiments from docs
- All code passes quality checks
- Data is backed up and accessible

---

## Risk Management

### Risk 1: GPU Quota Insufficient
**Mitigation**: 
- Test with fewer models first
- Use model quantization (int8/int4)
- Time-share GPUs across experiments

### Risk 2: Model Download Failures
**Mitigation**:
- Pre-download models to PVC before experiments
- Use HuggingFace mirrors
- Implement retry logic in init containers

### Risk 3: vLLM OOM Errors
**Mitigation**:
- Adjust `--max-model-len` parameter
- Reduce `--gpu-memory-utilization`
- Use smaller batch sizes

### Risk 4: Job Failures
**Mitigation**:
- Implement checkpointing in scripts
- Set `backoffLimit` in job specs
- Add comprehensive error logging

### Risk 5: DeepSeek API Rate Limits
**Mitigation**:
- Self-host DeepSeek model
- Implement exponential backoff
- Process in smaller batches

### Risk 6: Data Loss
**Mitigation**:
- Use PVCs with proper backup policies
- Export data incrementally during experiments
- Enable Kubernetes volume snapshots

---

## Success Metrics

### Technical Metrics
- **Uptime**: Model servers available >99% of experiment duration
- **Throughput**: Process 300 rows (Study 1) in <2 hours per model
- **Accuracy**: <1% job failure rate
- **Validation**: >95% structured output passes schema validation

### Research Metrics
- **Coverage**: All planned model × study × experiment combinations complete
- **Quality**: Human review finds >90% of model responses reasonable
- **Reproducibility**: Can re-run experiments and get same results (±5% variance)

---

## Timeline Summary

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Phase 0 | Week 1 | NRP access + dev environment ready |
| Phase 1 | Week 2 | Docker images built and tested |
| Phase 2 | Week 3 | All models deployed and validated |
| Phase 3 | Week 4 | Experiment 1 complete (raw responses) |
| Phase 4 | Week 5 | Experiment 2 complete (probabilities) |
| Phase 5 | Week 6 | Analysis and visualization complete |
| Phase 6 | Week 7 | Documentation and cleanup |

**Total Duration**: 7 weeks (can be compressed with parallel work)

---

## Next Actions

1. **Immediate**: Set up NRP access and test cluster connectivity
2. **This Week**: Create project structure and start Docker image development
3. **Next Week**: Deploy first model (Gemma-2B) and run pilot test

---

## Notes & Open Questions

1. **Model Selection**: Which exact model variants to use?
   - Gemma: 2B, 7B, or 27B?
   - Llama: 8B or 70B?
   - Need to balance quality vs compute cost

2. **DeepSeek Decision**: Self-host or use API?
   - API: Simpler, faster setup, pay per token
   - Self-host: More control, better for large volumes, needs GPUs

3. **Batch Size**: What's optimal for throughput vs quality?
   - Need to test different batch sizes
   - May vary by model size

4. **Temperature**: Use 0.7 (like original script) or 1.0 for probabilities?
   - Exp 1: Maybe use 0.7 for more coherent responses
   - Exp 2: Use 1.0 for unbiased probability distributions

5. **Validation Threshold**: What % of structured output errors is acceptable?
   - Need to decide if manual cleanup is needed
   - Or re-run with better prompts

**Decision log should be maintained throughout project**
