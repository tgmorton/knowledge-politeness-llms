# Grace Project - PVC-Based Deployment Guide

**No Docker build/push required!** This approach uses persistent volumes to store code and dependencies, avoiding all Docker registry issues.

## Overview

Instead of building a custom Docker image, we:
1. Create persistent storage (PVCs)
2. Install code + dependencies to PVCs once
3. Run experiments using base PyTorch image + PVC code

## Prerequisites

- kubectl configured for NRP cluster
- Access to `lemn-lab` namespace
- HuggingFace token secret (optional, for gated models)

## Step-by-Step Deployment

### 1. Create PVCs (One-time)

```bash
kubectl apply -f 00-pvcs.yaml
```

**Creates:**
- `grace-code` (10Gi) - Python code and dependencies
- `grace-model-cache` (100Gi) - Model weights cache
- `grace-results` (50Gi) - Experiment outputs

**Verify:**
```bash
kubectl get pvc -n lemn-lab | grep grace
```

---

### 2. Run Setup Job (One-time)

```bash
kubectl apply -f 01-setup-environment.yaml
```

**This job:**
- Clones source code from GitHub
- Installs Python dependencies (httpx, pandas, transformers, etc.)
- Organizes directory structure
- Takes ~2-3 minutes

**Monitor:**
```bash
# Watch job progress
kubectl logs -f job/grace-setup-environment -n lemn-lab

# Check status
kubectl get job/grace-setup-environment -n lemn-lab
```

**Expected output:**
```
✅ Environment ready for experiments!
```

**If you need to re-run setup** (e.g., to update code):
```bash
kubectl delete job grace-setup-environment -n lemn-lab
kubectl apply -f 01-setup-environment.yaml
```

---

### 3. Deploy vLLM Server

```bash
kubectl apply -f 02-vllm-gemma-2b.yaml
```

**This deploys:**
- vLLM server with Gemma-2-2B model
- OpenAI-compatible API on port 8000
- Internal service: `http://vllm-gemma-2b:8000`

**Monitor startup:**
```bash
# Watch pod creation
kubectl get pods -n lemn-lab -l model=gemma-2b -w

# Check logs (model loading takes 2-3 minutes)
kubectl logs -f deployment/vllm-gemma-2b -n lemn-lab

# Verify health
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- curl http://localhost:8000/health
```

**Expected:**
- Pod status: `Running`
- Health endpoint: `{"status": "ok"}`
- Logs show: `Uvicorn running on http://0.0.0.0:8000`

---

### 4. Run Test Experiment

```bash
kubectl apply -f 03-test-study1-exp1-gemma-2b.yaml
```

**This job:**
- Runs Study 1 Experiment 1 with **10 trials** (for testing)
- Uses code from PVC
- Queries vLLM server via internal DNS
- Saves results to PVC

**Monitor:**
```bash
# Watch job
kubectl logs -f job/test-study1-exp1-gemma-2b -n lemn-lab

# Check results
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- ls -lh /models/../results/
```

**Expected output:**
```
✅ Test Complete
Check /results/ for output file
```

---

## Troubleshooting

### Setup job fails to clone repo
```bash
# Check if PVC is mounted
kubectl describe pod <setup-pod-name> -n lemn-lab

# Check if git is available
kubectl logs job/grace-setup-environment -n lemn-lab
```

### vLLM server won't start
```bash
# Check GPU allocation
kubectl describe pod -l model=gemma-2b -n lemn-lab | grep -A 5 "Allocated resources"

# Check for RTX 3090 nodes
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090

# View detailed logs
kubectl logs -f deployment/vllm-gemma-2b -n lemn-lab
```

**Common issues:**
- **No GPUs available**: RTX 3090 nodes may be busy, wait or use A100
- **OOM (Out of Memory)**: Model too large for GPU, check dtype=bfloat16
- **Model download slow**: First run downloads 5GB model to cache PVC

### Experiment job fails
```bash
# Check if vLLM is healthy
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- curl http://localhost:8000/health

# Check Python imports
kubectl logs job/test-study1-exp1-gemma-2b -n lemn-lab | grep "import"

# Verify code PVC contents
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- ls -R /code/
```

---

## Viewing Results

```bash
# List result files
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- ls -lh /models/../results/

# Download a result file
kubectl cp lemn-lab/<pod-name>:/results/test_study1_exp1_gemma-2b_*.json ./local-results/

# Or view directly
kubectl exec -it deployment/vllm-gemma-2b -n lemn-lab -- cat /models/../results/test_*.json
```

---

## Running Full Experiments

Once the test succeeds, you can run full experiments by modifying the job YAML:

```yaml
# Remove --limit flag for full dataset
args:
  - |
    python3 /code/src/query_study1_exp1.py \
      --input=/code/data/study1.csv \
      --output=/results/study1_exp1_gemma-2b_full_$(date +%Y%m%d_%H%M%S).json \
      --endpoint=http://vllm-gemma-2b:8000 \
      --model-name=google/gemma-2-2b-it
      # No --limit flag = all 300 trials
```

---

## Cleanup

### Delete experiment job (keeps PVCs)
```bash
kubectl delete job test-study1-exp1-gemma-2b -n lemn-lab
```

### Stop vLLM server (keeps PVCs)
```bash
kubectl delete deployment vllm-gemma-2b -n lemn-lab
kubectl delete service vllm-gemma-2b -n lemn-lab
```

### Delete ALL resources including PVCs
```bash
kubectl delete -f 03-test-study1-exp1-gemma-2b.yaml
kubectl delete -f 02-vllm-gemma-2b.yaml
kubectl delete job grace-setup-environment -n lemn-lab
kubectl delete -f 00-pvcs.yaml
```

---

## Advantages of This Approach

✅ **No Docker build** - No waiting for builds
✅ **No Docker push** - No registry timeouts
✅ **Fast updates** - Change code by re-running setup job
✅ **Portable** - Works on any Kubernetes cluster
✅ **Cost effective** - Base images are free and cached
✅ **Simple** - No CI/CD pipeline needed

---

## Next Steps

After successful test:
1. Run full Study 1 Exp 1 (300 trials)
2. Deploy other models (Gemma-9B, Gemma-27B)
3. Run Study 1 Exp 2 (probability extraction)
4. Run Study 2 experiments

See parent documentation for full experimental pipeline.
