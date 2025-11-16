# Implementation Roadmap - Incremental Risk Strategy

## Overview

This roadmap is organized to **minimize dependency on NRP resources** until the very end. You'll build and test everything locally first, then deploy on non-A100 GPUs for validation, and only request A100 access when you're ready for production runs.

**Philosophy**: Work from low-risk (local development) to high-risk (A100 requests), validating at each stage.

---

## Phase 0: Local Development & Validation (Week 1-2)

**No NRP access required yet**

### Objectives
- Set up local development environment
- Build and test all Docker images locally
- Validate entire pipeline with small dataset
- Prepare for cluster deployment

### Why This First?
- ✅ No NRP access needed
- ✅ No GPU quota needed
- ✅ Can iterate quickly on local machine
- ✅ Identify issues before cluster deployment

---

### Tasks

#### 0.1: Development Environment Setup
- [ ] Install Docker Desktop or Podman
- [ ] Install Python 3.11+ virtual environment
- [ ] Install dependencies: `pip install vllm transformers httpx pandas pydantic`
- [ ] Install kubectl (for later phases)
- [ ] Set up code editor (VS Code, PyCharm, etc.)

#### 0.2: HuggingFace Setup
- [ ] Create HuggingFace account: https://huggingface.co/
- [ ] Generate API token: Settings → Access Tokens
- [ ] Test model download locally:
  ```bash
  huggingface-cli login
  huggingface-cli download google/gemma-2-2b-it
  ```
- [ ] Verify model licenses for research use

#### 0.3: Local vLLM Testing
- [ ] Download Gemma-2B model locally
- [ ] Start vLLM server locally:
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-2b-it \
    --dtype float16 \
    --max-model-len 2048 \
    --port 8000
  ```
- [ ] Test endpoints:
  ```bash
  curl http://localhost:8000/health
  curl http://localhost:8000/v1/models
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gemma-2-2b-it","prompt":"Test","max_tokens":10}'
  ```

#### 0.4: Project Structure
```
10-GraceProject/
├── docker/
│   ├── query-generator/         # Query job image
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── probability-extractor/   # Probability extraction image
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── structured-extractor/    # DeepSeek extraction image
│       ├── Dockerfile
│       └── requirements.txt
├── src/
│   ├── query_study1_exp1.py
│   ├── query_study2_exp1.py
│   ├── extract_probabilities_study1.py
│   ├── extract_probabilities_study2.py
│   ├── extract_structured_output.py
│   ├── schemas/
│   │   ├── study1_schema.json
│   │   └── study2_schema.json
│   └── utils/
│       ├── __init__.py
│       ├── api_client.py
│       └── validation.py
├── tests/
│   ├── test_query_generation.py
│   ├── test_probability_extraction.py
│   └── test_structured_output.py
├── kubernetes/              # Will create in Phase 2
├── data/
│   ├── study1.csv          # Existing
│   ├── study2.csv          # Existing
│   └── test_sample.csv     # Small test dataset (10 rows)
└── outputs/                # Local test outputs
```

- [ ] Create directory structure
- [ ] Initialize git repo
- [ ] Create `.gitignore` (exclude `outputs/`, `.env`, etc.)
- [ ] Commit initial structure

#### 0.5: Test Dataset Creation
- [ ] Extract first 10 rows from study1.csv → `data/test_sample_study1.csv`
- [ ] Extract first 10 rows from study2.csv → `data/test_sample_study2.csv`
- [ ] Verify CSV structure is correct

#### 0.6: Python Scripts Development

**Implement Query Generator** (`src/query_study1_exp1.py`):
```python
#!/usr/bin/env python3
"""
Query generator for Study 1, Experiment 1.
Generates raw text responses from LLM.
"""
import argparse
import asyncio
import pandas as pd
import httpx
from datetime import datetime

class Study1QueryGenerator:
    def __init__(self, input_csv, output_csv, model_endpoint, model_name):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model_endpoint = model_endpoint
        self.model_name = model_name
    
    def construct_prompt(self, row):
        return f"""Context: {row['story_setup']}

Initial Question: {row['priorQ']}

New Information: {row['speach']}

Updated Question: {row['speachQ']}

Please provide your answers in this format:
Initial Answer: [0-3]
Probabilities: 0: X%, 1: Y%, 2: Z%, 3: W%
Knowledge: [yes/no]
"""
    
    async def query_model(self, prompt, client):
        response = await client.post(
            f"{self.model_endpoint}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.7
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()
    
    async def run(self):
        df = pd.read_csv(self.input_csv)
        df['model_name'] = self.model_name
        df['model_response'] = None
        df['response_timestamp'] = None
        
        async with httpx.AsyncClient() as client:
            for idx, row in df.iterrows():
                print(f"Processing row {idx+1}/{len(df)}")
                prompt = self.construct_prompt(row)
                result = await self.query_model(prompt, client)
                df.at[idx, 'model_response'] = result['choices'][0]['message']['content']
                df.at[idx, 'response_timestamp'] = datetime.now().isoformat()
        
        df.to_csv(self.output_csv, index=False)
        print(f"Saved to {self.output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-endpoint', default='http://localhost:8000')
    parser.add_argument('--model-name', default='gemma-2-2b-it')
    args = parser.parse_args()
    
    generator = Study1QueryGenerator(
        args.input, args.output, args.model_endpoint, args.model_name
    )
    asyncio.run(generator.run())

if __name__ == '__main__':
    main()
```

- [ ] Implement `query_study1_exp1.py` (see template above)
- [ ] Implement similar script for `query_study2_exp1.py`
- [ ] Implement `extract_probabilities_study1.py` (logprobs extraction)
- [ ] Implement `extract_structured_output.py` (DeepSeek parsing)

#### 0.7: Local Testing (Critical!)
- [ ] Start local vLLM server with Gemma-2B
- [ ] Run query generator on test_sample_study1.csv (10 rows):
  ```bash
  python src/query_study1_exp1.py \
    --input data/test_sample_study1.csv \
    --output outputs/test_study1_gemma2b.csv \
    --model-endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
  ```
- [ ] Verify output CSV has all expected columns
- [ ] Manually inspect responses for quality
- [ ] Fix any bugs or issues

#### 0.8: Docker Images (Build Locally)
- [ ] Create Dockerfile for query-generator
- [ ] Create Dockerfile for probability-extractor
- [ ] Create Dockerfile for structured-extractor
- [ ] Build all images locally:
  ```bash
  docker build -t grace/query-generator:latest docker/query-generator/
  docker build -t grace/probability-extractor:latest docker/probability-extractor/
  docker build -t grace/structured-extractor:latest docker/structured-extractor/
  ```
- [ ] Test Docker images locally:
  ```bash
  docker run --network=host grace/query-generator:latest \
    --input /data/test_sample_study1.csv \
    --output /output/test.csv
  ```

### Deliverables
- ✅ Working Python scripts (tested locally)
- ✅ Docker images (tested locally)
- ✅ Test outputs from 10-row sample
- ✅ Bugs identified and fixed

### Success Criteria
- Scripts run successfully against local vLLM
- Output CSVs have correct structure
- Responses look reasonable (manual inspection)
- Ready to deploy to cluster

---

## Phase 1: NRP Access & Basic Cluster Testing (Week 3)

**Minimal NRP resources required**

### Objectives
- Get NRP access
- Test basic Kubernetes functionality
- Deploy on cheapest/easiest GPU first (RTX 3090 or similar)
- Validate cluster deployment works

### Why This Second?
- ✅ Everything is tested locally already
- ✅ Use non-A100 GPUs (no special access needed)
- ✅ Validate Kubernetes setup before requesting A100s
- ✅ Can still proceed if A100 access is denied

---

### Tasks

#### 1.1: NRP Cluster Access
- [ ] Request NRP account: https://nautilus.optiputer.net/
- [ ] Obtain kubeconfig file
- [ ] Configure kubectl:
  ```bash
  export KUBECONFIG=~/nrp-kubeconfig.yaml
  kubectl cluster-info
  kubectl get nodes
  ```
- [ ] Join NRP Matrix chat: https://matrix.to/#/#nrp:matrix.org
- [ ] Review cluster policies: `docs/NRP_CLUSTER_GUIDE.md`

#### 1.2: Explore Available GPUs
- [ ] Check available GPU types:
  ```bash
  kubectl get nodes -L nvidia.com/gpu.product | grep -E "RTX|Tesla|A100"
  ```
- [ ] Identify RTX 3090 nodes (no special access needed):
  ```bash
  kubectl get nodes -L nvidia.com/gpu.product | grep "RTX-3090"
  ```
- [ ] Document available nodes for later use

#### 1.3: Create Namespace & Storage
- [ ] Create namespace:
  ```yaml
  apiVersion: v1
  kind: Namespace
  metadata:
    name: grace-experiments
  ```
- [ ] Create small test PVC (5Gi):
  ```yaml
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    name: grace-test-data
    namespace: grace-experiments
  spec:
    accessModes:
      - ReadWriteMany
    resources:
      requests:
        storage: 5Gi
    storageClassName: rook-cephfs
  ```
- [ ] Upload test dataset to PVC

#### 1.4: Deploy Test Model (Gemma-2B on RTX 3090)

**Why Gemma-2B on RTX 3090?**
- ✅ Small model (5GB), fits easily on 24GB GPU
- ✅ RTX 3090 is most available GPU on NRP (~50 nodes)
- ✅ No special access or quota needed
- ✅ Fast deployment for testing

**Create Deployment** (note: using Deployment, not StatefulSet):
```yaml
apiVersion: apps/v1
kind: Deployment  # Using Deployment (compliant with NRP policies)
metadata:
  name: vllm-gemma-2b-test
  namespace: grace-experiments
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
      model: gemma-2b
  template:
    metadata:
      labels:
        app: vllm-server
        model: gemma-2b
    spec:
      # Target RTX 3090 (most available, no special access)
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-GeForce-RTX-3090
      
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=google/gemma-2-2b-it
        - --host=0.0.0.0
        - --port=8000
        - --dtype=float16
        - --max-model-len=2048
        - --gpu-memory-utilization=0.90
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: 16Gi
          limits:
            nvidia.com/gpu: 1
            cpu: "5"      # Within 20% of request (125%)
            memory: 19Gi  # Within 20% of request (119%)
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-token
              key: token
```

- [ ] Create HuggingFace token secret
- [ ] Apply Deployment
- [ ] Monitor deployment:
  ```bash
  kubectl get pods -n grace-experiments -w
  kubectl logs -f deployment/vllm-gemma-2b-test -n grace-experiments
  ```

#### 1.5: Test Model Endpoint on Cluster
- [ ] Create Service for model
- [ ] Port-forward to test:
  ```bash
  kubectl port-forward -n grace-experiments deployment/vllm-gemma-2b-test 8000:8000
  ```
- [ ] Test endpoints (same as local tests)
- [ ] Verify GPU is being used:
  ```bash
  kubectl exec -it deployment/vllm-gemma-2b-test -n grace-experiments -- nvidia-smi
  ```

#### 1.6: Run Test Job on Cluster

**Create Job** (for query generation):
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: test-query-gemma2b
  namespace: grace-experiments
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: query-generator
        image: grace/query-generator:latest  # Your Docker image
        args:
        - --input=/data/test_sample_study1.csv
        - --output=/output/test_output.csv
        - --model-endpoint=http://vllm-gemma-2b-test:8000
        - --model-name=gemma-2-2b-it
        resources:
          requests:
            cpu: "2"
            memory: 4Gi
          limits:
            cpu: "2"      # Guaranteed QoS (limit = request)
            memory: 4Gi
        volumeMounts:
        - name: data
          mountPath: /data
        - name: output
          mountPath: /output
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: grace-test-data
      - name: output
        persistentVolumeClaim:
          claimName: grace-test-data
```

- [ ] Push Docker images to registry (Docker Hub or NRP Harbor)
- [ ] Apply Job manifest
- [ ] Monitor job:
  ```bash
  kubectl get jobs -n grace-experiments -w
  kubectl logs job/test-query-gemma2b -n grace-experiments
  ```
- [ ] Download and inspect output

#### 1.7: Validate End-to-End Pipeline
- [ ] Verify output CSV is correct
- [ ] Compare cluster output to local output (should be similar)
- [ ] Check for any errors or issues
- [ ] **Decision point**: If this works, we're ready to scale up!

#### 1.8: Cleanup Test Resources
- [ ] Delete test Job
- [ ] Keep test Deployment running for now (will use in Phase 2)
- [ ] Keep PVC (will reuse)

### Deliverables
- ✅ NRP access confirmed
- ✅ Test model deployed on RTX 3090
- ✅ Test job ran successfully
- ✅ Pipeline validated on cluster

### Success Criteria
- Can deploy pods to NRP cluster
- Model endpoint works on cluster (same as local)
- Job completes and produces valid output
- No blocking issues

---

## Phase 2: Full Testing on Non-A100 GPUs (Week 4)

**Still no A100 access needed**

### Objectives
- Run full experiments on RTX 3090 cluster with quantized models
- Validate complete pipeline (all models, all experiments)
- Generate preliminary results
- Identify any remaining issues

### Why This Third?
- ✅ Proves entire pipeline works end-to-end
- ✅ Uses readily available RTX 3090 GPUs (no waiting for A100 quota)
- ✅ Generates usable research data (with quantization)
- ✅ Identifies issues before committing to A100s
- ✅ **Fallback option**: If A100 access is denied, you can use these results

---

### Tasks

#### 2.1: Deploy All Models on RTX 3090 (Quantized)

**Model Lineup** (see `docs/GPU_CONTINGENCY_PLAN.md`):

| Model | Quantization | VRAM | Target Node |
|-------|--------------|------|-------------|
| Gemma-2B | FP16 | ~5GB | suncave-0 (RTX 3090) |
| Gemma-9B | AWQ 4-bit | ~6GB | suncave-1 (RTX 3090) |
| Gemma-27B | AWQ 4-bit | ~18GB | suncave-2 (RTX 3090) |
| Llama-8B | FP16 | ~18GB | suncave-3 (RTX 3090) |
| Llama-70B | AWQ 4-bit | ~45GB | suncave-4 + suncave-6 (2x RTX 3090) |
| GPT-OSS-20B | AWQ 4-bit | ~8GB | suncave-7 (RTX 3090) |

- [ ] Deploy Gemma-2B (already done in Phase 1)
- [ ] Deploy Gemma-9B (AWQ):
  ```yaml
  args:
  - --model=TheBloke/gemma-2-9b-it-AWQ
  - --quantization=awq
  - --dtype=half
  ```
- [ ] Deploy Gemma-27B (AWQ)
- [ ] Deploy Llama-8B (FP16)
- [ ] Deploy Llama-70B (AWQ, 2 GPUs)
- [ ] Deploy GPT-OSS-20B (AWQ)

**Node Selectors**: Use Suncave cluster (all RTX 3090, co-located at SDSC)
```yaml
nodeSelector:
  kubernetes.io/hostname: suncave-0  # Specific node
```

#### 2.2: Run Experiment 1 (Raw Responses)

**Deploy Jobs for Study 1**:
- [ ] job-study1-exp1-gemma2b.yaml
- [ ] job-study1-exp1-gemma9b.yaml
- [ ] job-study1-exp1-gemma27b.yaml
- [ ] job-study1-exp1-llama8b.yaml
- [ ] job-study1-exp1-llama70b.yaml
- [ ] job-study1-exp1-gpt-oss-20b.yaml

- [ ] Apply all jobs in parallel
- [ ] Monitor progress (should complete in ~2-4 hours)
- [ ] Download outputs

**Deploy Jobs for Study 2**:
- [ ] Same process for Study 2 (2,424 rows)
- [ ] Monitor progress (should complete in ~4-8 hours)

#### 2.3: Run Structured Output Extraction
- [ ] Deploy DeepSeek extraction jobs (or use DeepSeek API)
- [ ] Process all Experiment 1 outputs
- [ ] Validate structured outputs
- [ ] Generate data quality report

#### 2.4: Run Experiment 2 (Probability Extraction)
- [ ] Deploy probability extraction jobs for Study 1
- [ ] Deploy probability extraction jobs for Study 2
- [ ] Validate probability distributions
- [ ] Check that probabilities sum to 1.0

#### 2.5: Data Quality Check
- [ ] Verify all row counts match input
- [ ] Check for missing values
- [ ] Validate probability ranges
- [ ] Spot-check response quality
- [ ] **Decision point**: Is quality good enough?

#### 2.6: Preliminary Analysis
- [ ] Run basic R analysis scripts
- [ ] Generate comparison plots
- [ ] Check if quantization significantly affects results
- [ ] Document any quality issues

### Deliverables
- ✅ Complete dataset from all 6 models (quantized)
- ✅ All experiments completed (Exp 1 & 2, Study 1 & 2)
- ✅ Preliminary analysis results
- ✅ Quality assessment report

### Success Criteria
- All jobs complete successfully
- Data quality is acceptable for research
- Pipeline runs smoothly without manual intervention
- **You have usable results** (even if A100s become available later)

### Decision Point: Proceed to A100s?

**If quality is sufficient** → You're done! Use these results.

**If you want higher quality** → Proceed to Phase 3 (A100 deployment)

---

## Phase 3: A100 Deployment Request (Week 5)

**First time requesting special access**

### Objectives
- Request A100 access
- Request necessary exceptions
- Wait for approvals
- Plan A100 deployment

### Why This Fourth?
- ✅ You've already proven the pipeline works
- ✅ You have preliminary results as backup
- ✅ You can show NRP admins exactly what you need
- ✅ If denied, Phase 2 results are still valid

---

### Tasks

#### 3.1: Prepare Exception Requests

**Based on sequential deployment** (see `docs/NRP_POLICY_COMPLIANCE_AUDIT.md`):

**Request 1: A100 Access Form**
- [ ] Complete A100 access request form: https://nautilus.optiputer.net/a100-request
- [ ] Provide details:
  - Project: Grace Project - Comparative LLM Study
  - Workflow: Sequential model deployment (1 model at a time)
  - Max concurrent GPUs: 4 (for Llama-70B)
  - Duration: 6 days of GPU time (1 day per model × 6 models)
  - Justification: Academic research, comparing model architectures

**Request 2: Multi-GPU Exception (if needed for Llama-70B)**

If only A100-40GB available:
- [ ] Post in Matrix chat:
  ```
  Subject: Exception Request - Multi-GPU Pod for Llama-70B
  
  Namespace: grace-experiments
  Model: Llama-70B (140GB model weights)
  Requested: 4x A100-40GB GPUs per pod (tensor parallelism)
  Duration: <1 day (then will be deleted)
  Reason: 70B parameter model requires tensor parallelism across 4 GPUs
  Sequential deployment: Only 1 model running at a time
  ```

If A100-80GB available:
- [ ] No multi-GPU exception needed! (Llama-70B fits on 2x A100-80GB)

**Request 3: High RAM Exception (if needed)**

If Llama-70B or large models:
- [ ] Post in Matrix chat:
  ```
  Subject: Exception Request - High RAM Pod
  
  Namespace: grace-experiments
  Pod: vllm-llama-70b
  Requested RAM: 80GB (or 160GB if using 4 GPUs)
  Duration: <1 day
  Reason: 70B parameter model + KV cache requires high RAM
  ```

#### 3.2: Wait for Approvals
- [ ] Monitor Matrix chat for responses
- [ ] Answer any admin questions
- [ ] Typical approval time: 1-3 days

#### 3.3: During Wait: Prepare A100 Manifests
- [ ] Create Kubernetes manifests for A100 deployment
- [ ] Use node selectors for A100-80GB (see `docs/NRP_A100_NODES.md`)
- [ ] Update resource limits to be within 20% of requests
- [ ] Review all manifests for NRP policy compliance

**Example A100-80GB Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b-a100
  namespace: grace-experiments
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
      model: llama-70b
  template:
    metadata:
      labels:
        app: vllm-server
        model: llama-70b
    spec:
      # Target A100-80GB nodes (SDSC preferred)
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: kubernetes.io/hostname
                operator: In
                values:
                - node-1-1.sdsc.optiputer.net
                - node-1-3.sdsc.optiputer.net
                - node-1-4.sdsc.optiputer.net
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe
      
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=meta-llama/Meta-Llama-3-70B-Instruct
        - --host=0.0.0.0
        - --port=8000
        - --dtype=bfloat16
        - --max-model-len=4096
        - --gpu-memory-utilization=0.92
        - --tensor-parallel-size=2  # 2x A100-80GB
        resources:
          requests:
            nvidia.com/gpu: 2
            cpu: "16"
            memory: 80Gi
          limits:
            nvidia.com/gpu: 2
            cpu: "19"      # Within 20% (119%)
            memory: 96Gi   # Within 20% (120%)
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-token
              key: token
```

#### 3.4: Verify A100 Availability
Once approved:
- [ ] Check available A100 nodes:
  ```bash
  kubectl get nodes -L nvidia.com/gpu.product | grep "A100-SXM4-80GB"
  kubectl describe node node-1-1.sdsc.optiputer.net | grep "Allocated resources"
  ```
- [ ] Identify which nodes have free GPUs
- [ ] Update manifests with specific node selectors

### Deliverables
- ✅ A100 access approved
- ✅ Exception requests approved (if needed)
- ✅ A100 deployment manifests ready

### Success Criteria
- A100 quota granted
- Exceptions approved (if needed)
- Ready to deploy on A100s

---

## Phase 4: A100 Deployment & Production Runs (Week 6)

**Using A100s - high-value resource**

### Objectives
- Deploy models on A100 GPUs (FP16, no quantization)
- Run full experiments with highest quality
- Generate final production datasets
- Complete all analysis

### Why This Last?
- ✅ Everything is tested and validated
- ✅ Pipeline runs smoothly
- ✅ Minimal risk of wasting GPU time
- ✅ Can complete quickly (all models in ~6 days)

---

### Tasks

#### 4.1: Sequential Model Deployment

**Deploy one model at a time, run experiments, delete, move to next**

**Day 1: Gemma-2B**
- [ ] Deploy Gemma-2B on 1x A100-80GB (FP16)
- [ ] Run Study 1 Exp 1 & 2
- [ ] Run Study 2 Exp 1 & 2
- [ ] Download outputs
- [ ] **Delete Deployment** (free up GPU)

**Day 2: Gemma-9B**
- [ ] Deploy Gemma-9B on 1x A100-80GB (FP16)
- [ ] Run all experiments
- [ ] Download outputs
- [ ] **Delete Deployment**

**Day 3: Gemma-27B**
- [ ] Deploy Gemma-27B on 1-2x A100-80GB (FP16)
- [ ] Run all experiments
- [ ] Download outputs
- [ ] **Delete Deployment**

**Day 4: Llama-8B**
- [ ] Deploy Llama-8B on 1x A100-80GB (FP16)
- [ ] Run all experiments
- [ ] Download outputs
- [ ] **Delete Deployment**

**Day 5-6: Llama-70B**
- [ ] Deploy Llama-70B on 2x A100-80GB (FP16)
- [ ] Run all experiments (may take longer due to size)
- [ ] Download outputs
- [ ] **Delete Deployment**

**Day 6: GPT-OSS-20B**
- [ ] Deploy GPT-OSS-20B on 1x A100-80GB (FP16)
- [ ] Run all experiments
- [ ] Download outputs
- [ ] **Delete Deployment**

#### 4.2: Monitor Each Run
- [ ] Check GPU utilization (should be >40%, ideally >80%)
- [ ] Monitor for OOM errors
- [ ] Check NRP violations page daily: https://nautilus.optiputer.net/violations
- [ ] Verify output quality in real-time

#### 4.3: Data Collection & Validation
- [ ] Download all outputs after each model
- [ ] Backup to external storage (not just NRP)
- [ ] Validate data quality
- [ ] Compare to Phase 2 results (quantized vs FP16)

#### 4.4: Structured Output Extraction
- [ ] Run DeepSeek extraction on all A100 outputs
- [ ] Validate against JSON schemas
- [ ] Merge with original data

### Deliverables
- ✅ Complete dataset from all 6 models (FP16, highest quality)
- ✅ All experiments completed
- ✅ Data backed up externally

### Success Criteria
- All models deployed and tested on A100s
- All experiments complete
- Data quality exceeds Phase 2 results
- No GPU time wasted

---

## Phase 5: Analysis & Publication (Week 7)

**No NRP resources needed**

### Objectives
- Run final analysis
- Generate publication-quality figures
- Write up results
- Clean up cluster resources

---

### Tasks

#### 5.1: Comparative Analysis
- [ ] Compare Phase 2 (quantized) vs Phase 4 (FP16) results
- [ ] Assess impact of quantization on quality
- [ ] Run statistical analyses
- [ ] Generate comparative visualizations

#### 5.2: Final R Analysis
- [ ] Update R scripts with all model data
- [ ] Generate all figures for publication
- [ ] Run statistical tests
- [ ] Create summary tables

#### 5.3: Write Up Results
- [ ] Methods section (describe pipeline)
- [ ] Results section (comparative findings)
- [ ] Discussion (implications)
- [ ] Supplementary materials

#### 5.4: Cluster Cleanup

**CRITICAL: Clean up within 30 days to avoid data purge**

- [ ] Delete all Deployments:
  ```bash
  kubectl delete deployment -l app=vllm-server -n grace-experiments
  ```
- [ ] Delete all Jobs:
  ```bash
  kubectl delete jobs -l project=grace -n grace-experiments
  ```
- [ ] **Archive data** before deleting PVCs:
  ```bash
  # Create tar archive
  kubectl run archiver --image=busybox -n grace-experiments --restart=Never \
    --overrides='{...}'
  kubectl exec archiver -- tar czf /output/grace-archive.tar.gz /output
  kubectl cp grace-experiments/archiver:/output/grace-archive.tar.gz ./
  ```
- [ ] Delete PVCs (saves 465Gi):
  ```bash
  kubectl delete pvc -l project=grace -n grace-experiments
  ```
- [ ] Verify all resources cleaned up:
  ```bash
  kubectl get all -n grace-experiments
  ```

### Deliverables
- ✅ Publication-ready manuscript
- ✅ All data archived externally
- ✅ NRP resources cleaned up
- ✅ Reproducibility package

### Success Criteria
- Paper submitted or ready for submission
- All data safely archived
- NRP cluster fully cleaned up
- Reproducibility guide complete

---

## Timeline Summary

| Phase | Duration | Key Activities | NRP Resources |
|-------|----------|----------------|---------------|
| **Phase 0** | Week 1-2 | Local development, testing | None |
| **Phase 1** | Week 3 | Basic cluster testing (RTX 3090) | Minimal (no quota) |
| **Phase 2** | Week 4 | Full testing on RTX 3090 (quantized) | 8 RTX 3090 GPUs |
| **Phase 3** | Week 5 | Request A100 access, wait for approval | None (waiting) |
| **Phase 4** | Week 6 | Production runs on A100s (FP16) | 1-4 A100 GPUs (sequential) |
| **Phase 5** | Week 7 | Analysis, publication, cleanup | None |

**Total Duration**: 7 weeks  
**A100 GPU Time**: 6 days (1 day per model)  
**Peak A100 Usage**: 2 GPUs (for Llama-70B), then back to 0

---

## Risk Mitigation Strategy

### Risk: A100 Access Denied

**Mitigation**: 
- ✅ Phase 2 results (RTX 3090 + quantization) are still publishable
- ✅ AWQ 4-bit has <3% quality degradation
- ✅ Still demonstrates cross-model comparison

**Outcome**: Project succeeds regardless of A100 access

### Risk: A100 Quota Insufficient

**Mitigation**:
- ✅ Sequential deployment (max 2 GPUs at a time)
- ✅ Can run subset of models on A100, rest on RTX 3090
- ✅ Can use A100-40GB with quantization if 80GB unavailable

### Risk: Pipeline Breaks on Cluster

**Mitigation**:
- ✅ Extensively tested locally (Phase 0)
- ✅ Validated on RTX 3090 (Phase 1-2)
- ✅ Issues fixed before using A100s

---

## Success Metrics

### Phase 0 Success
- [ ] All scripts run locally without errors
- [ ] Test outputs look correct
- [ ] Docker images build successfully

### Phase 1 Success
- [ ] Can deploy to NRP cluster
- [ ] Model endpoint works
- [ ] Test job completes

### Phase 2 Success
- [ ] All 6 models deployed on RTX 3090
- [ ] All experiments complete
- [ ] **Have usable research data**

### Phase 3 Success
- [ ] A100 access granted (or decided to use Phase 2 results)

### Phase 4 Success
- [ ] All models run on A100s
- [ ] Higher quality data than Phase 2

### Phase 5 Success
- [ ] Paper ready for submission
- [ ] Cluster cleaned up

---

## Decision Points

### After Phase 0
**Question**: Is the local pipeline working correctly?
- **Yes** → Proceed to Phase 1
- **No** → Fix issues, iterate

### After Phase 1
**Question**: Does cluster deployment work?
- **Yes** → Proceed to Phase 2
- **No** → Debug cluster issues

### After Phase 2
**Question**: Is Phase 2 data quality sufficient for publication?
- **Yes, quality is good enough** → Skip Phase 3-4, proceed to Phase 5 (Analysis)
- **No, want higher quality** → Proceed to Phase 3 (Request A100s)

### After Phase 3
**Question**: Was A100 access granted?
- **Yes** → Proceed to Phase 4
- **No** → Use Phase 2 results, proceed to Phase 5

---

## Key Advantages of This Approach

1. **No Wasted A100 Time**: Everything tested before touching A100s
2. **Fallback Options**: Phase 2 results are publishable if A100s unavailable
3. **Low Risk**: Each phase validates the next
4. **Respectful of Resources**: Only request A100s when truly ready
5. **Sequential Deployment**: No >2 week exception needed
6. **Minimal Exceptions**: Possibly none if using A100-80GB

---

## Next Actions

### This Week (Phase 0)
1. [ ] Set up local Python environment
2. [ ] Download Gemma-2B model
3. [ ] Start local vLLM server
4. [ ] Implement query_study1_exp1.py
5. [ ] Test with 10 rows locally

### Next Week (Phase 1)
1. [ ] Request NRP access
2. [ ] Get kubeconfig
3. [ ] Deploy test model on RTX 3090
4. [ ] Run test job

### Week After (Phase 2)
1. [ ] Deploy all 6 models on RTX 3090
2. [ ] Run full experiments
3. [ ] Assess data quality

**A100 request comes ONLY after Phase 2 is complete and validated!**

---

**Document Status**: ✅ Complete - Reorganized for incremental risk  
**Philosophy**: Earn the right to use A100s by proving everything works first
