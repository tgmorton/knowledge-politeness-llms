# Kubernetes Deployment Guide

## Overview

This guide walks you through deploying vLLM on the NRP Kubernetes cluster **step-by-step**. Don't worry if you're new to Kubernetes - we'll explain everything!

## Prerequisites

### 1. Cluster Access

You need:
- ✅ NRP cluster account
- ✅ `kubectl` installed on your Mac
- ✅ Kubeconfig file configured

**Check if you have access:**
```bash
kubectl get nodes

# Should show list of nodes (if configured correctly)
# If error, contact NRP support or check: https://docs.nationalresearchplatform.org/
```

### 2. Verify GPU Availability

```bash
# Check available A100 nodes
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB

# Should show several nodes with STATUS=Ready
```

If no A100 nodes, you can use RTX 3090:
```bash
# Check RTX 3090 nodes
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090

# Edit kubernetes/vllm-deployment.yaml and change nodeSelector to:
#   nvidia.com/gpu.product: NVIDIA-GeForce-RTX-3090
```

## Architecture: Hybrid Deployment

Our project uses a **hybrid architecture** that optimizes for both performance and cost:

### Experiment 1: vLLM API (Text Generation)
- **What**: Study 1 Exp 1 and Study 2 Exp 1
- **How**: Uses vLLM server deployed as Kubernetes Deployment
- **Job Requirements**: CPU-only (4Gi RAM, 2 CPU cores)
- **Why**: Fast text generation with OpenAI-compatible API

### Experiment 2: Direct Model Scoring (Probability Extraction)
- **What**: Study 1 Exp 2 and Study 2 Exp 2
- **How**: Loads model directly in Kubernetes Job (no vLLM server)
- **Job Requirements**: GPU required (1x A100-80GB, 32Gi RAM, 8 CPU cores)
- **Why**: Accurate probability extraction by scoring predefined options

**Benefits of Hybrid Approach:**
- ✅ Exp 1: Faster, efficient text generation
- ✅ Exp 2: Accurate probabilities (no generation ambiguity)
- ✅ No need to keep vLLM running during Exp 2 (saves GPU time)
- ✅ Each approach optimized for its task

## Compliance Review Summary

✅ **Our manifests are NRP-compliant!**

Key points:
- ✅ Resource limits within 20% of requests
- ✅ Using Deployment (correct for stateless servers)
- ✅ Using Jobs (correct for batch processing)
- ✅ Peak GPU: 1 per job/deployment (within default limit of 2)
- ✅ **No exception required** (with default resource limits)

See full compliance audit in the agent output above.

## Deployment Steps

### Step 1: Create Namespace

```bash
# Create the grace-experiments namespace
kubectl apply -f kubernetes/namespace.yaml

# Verify
kubectl get namespace grace-experiments
```

**What this does:**
- Creates isolated namespace for your experiments
- Keeps your resources separate from others

### Step 2: Deploy vLLM Server

```bash
# Deploy Gemma-2B with vLLM
kubectl apply -f kubernetes/vllm-deployment.yaml

# Check deployment status
kubectl get deployments -n grace-experiments

# Watch pods start up
kubectl get pods -n grace-experiments -w
```

**What to expect:**
1. **ContainerCreating** (1-2 minutes) - Downloading vLLM image
2. **Running** but not Ready (2-5 minutes) - Loading model from HuggingFace
3. **Running** and Ready (READY shows 1/1) - Server is up!

**Example output:**
```
NAME                            READY   STATUS    AGE
vllm-gemma-2b-7d9f5b8c6-x7k2p   1/1     Running   5m
```

### Step 3: Create Service

```bash
# Create service to expose vLLM endpoint
kubectl apply -f kubernetes/service.yaml

# Verify service exists
kubectl get service -n grace-experiments
```

**What this does:**
- Creates internal endpoint: `vllm-gemma-2b.grace-experiments.svc.cluster.local:8000`
- Allows other pods to connect to vLLM

### Step 4: Check Server Health

```bash
# Get pod name
export POD_NAME=$(kubectl get pods -n grace-experiments -l app=vllm -o jsonpath='{.items[0].metadata.name}')

# Check logs
kubectl logs -f $POD_NAME -n grace-experiments

# Should see:
# "Uvicorn running on http://0.0.0.0:8000"
# "INFO: Successfully loaded model..."
```

### Step 5: Port Forward (Access from Your Mac)

```bash
# Forward port 8000 to your local machine
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# Now vLLM is available at: http://localhost:8000
# Keep this terminal open!
```

**Test it works:**
```bash
# In another terminal:
curl http://localhost:8000/health

# Should return: {"status":"ok"}
```

### Step 6: Run Experiments

**Option A: Automated Script (Recommended)**

Use the automated deployment script that handles everything:

```bash
# Deploy model, run all 4 experiments, download results
./scripts/deploy_model_k8s.sh gemma-2b

# Available models: gemma-2b, gemma-9b, gemma-27b, llama-70b
```

This script:
1. Deploys vLLM server
2. Runs Experiment 1 for both studies (vLLM API)
3. Runs Experiment 2 for both studies (direct scoring with GPU)
4. Downloads results
5. Cleans up

**Option B: Manual Port-Forward (Local Development)**

For testing from your Mac with port-forward:

```bash
# In one terminal: Port forward
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# In another terminal: Run experiments locally
python3 src/query_study1_exp1.py \
    --input data/study1.csv \
    --output outputs/study1_exp1_gemma2b.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
```

**Option C: Run Jobs Directly**

For more control, deploy jobs manually:

```bash
# Experiment 1 (needs vLLM server running)
kubectl apply -f kubernetes/job-exp1-template.yaml

# Experiment 2 (no vLLM server needed, uses GPU directly)
kubectl apply -f kubernetes/job-exp2-template.yaml

# Monitor progress
kubectl get jobs -n grace-experiments -w
```

## Monitoring Your Deployment

### Check Resource Usage

```bash
# Get pod name
export POD_NAME=$(kubectl get pods -n grace-experiments -l app=vllm -o jsonpath='{.items[0].metadata.name}')

# Check CPU/Memory usage
kubectl top pod $POD_NAME -n grace-experiments

# Check GPU usage (exec into pod)
kubectl exec -it $POD_NAME -n grace-experiments -- nvidia-smi
```

**Expected GPU utilization:**
- During queries: 60-90%
- Idle: 10-20% (model loaded in VRAM)

**NRP Requirement**: Must maintain >40% GPU utilization on average, or may be flagged for waste.

### Check for Violations

NRP monitors resource usage. Check if you have any violations:

1. Visit: https://nautilus.optiputer.net/violations
2. Filter by namespace: `grace-experiments`
3. Look for warnings

**Common violations:**
- Low GPU utilization (<40%)
- Resource limits not within 20% of requests
- Using forbidden patterns

**Our deployment avoids all of these!** ✅

### View Logs

```bash
# Real-time logs
kubectl logs -f $POD_NAME -n grace-experiments

# Last 100 lines
kubectl logs --tail=100 $POD_NAME -n grace-experiments

# Logs with timestamps
kubectl logs --timestamps=true $POD_NAME -n grace-experiments
```

## Cleanup After Experiments

### Temporary Cleanup (Between Models)

```bash
# Delete deployment (keeps namespace)
kubectl delete deployment vllm-gemma-2b -n grace-experiments

# Verify deleted
kubectl get deployments -n grace-experiments
```

**When to do this:**
- Between models in sequential deployment
- Frees up GPU for next model
- Namespace persists (service remains)

### Full Cleanup (After All Experiments)

```bash
# Delete everything in namespace
kubectl delete namespace grace-experiments

# This deletes:
# - All deployments
# - All services
# - All pods
# - The namespace itself
```

**When to do this:**
- After completing all 6 models
- Final project cleanup
- Releases all cluster resources

## Troubleshooting

### Pod Stuck in "Pending"

**Symptoms:**
```
NAME                            READY   STATUS    AGE
vllm-gemma-2b-xxx               0/1     Pending   5m
```

**Check why:**
```bash
kubectl describe pod $POD_NAME -n grace-experiments | grep -A 10 Events

# Common reasons:
# - No GPU nodes available
# - Insufficient memory
# - Node selector too restrictive
```

**Solutions:**
1. Wait (GPU nodes may be busy)
2. Try RTX 3090 nodes (change nodeSelector)
3. Reduce resource requests

### Pod Crashes or OOMKilled

**Symptoms:**
```
NAME                            READY   STATUS    AGE
vllm-gemma-2b-xxx               0/1     OOMKilled 2m
```

**This means**: Out of Memory - pod used more memory than limit.

**Solutions:**
1. Reduce `--max-model-len` (try 2048 instead of 4096)
2. Use quantization: add `--quantization awq` to args
3. Increase memory limits (requires exception)

**Edit deployment:**
```bash
kubectl edit deployment vllm-gemma-2b -n grace-experiments

# Change args section:
# - --max-model-len=2048  # Reduced from 4096
# - --quantization=awq     # Add 4-bit quantization
```

### Port Forward Disconnects

**Symptoms:**
```
error: lost connection to pod
```

**Solution:**
Just restart port-forward:
```bash
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments
```

**Better solution** (auto-reconnect):
```bash
# Install kubefwd (optional)
brew install txn2/tap/kubefwd

# Or use a loop
while true; do
  kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments
  sleep 5
done
```

### Health Check Failures

**Symptoms:**
Pod shows Running but not Ready (0/1 Ready)

**Check:**
```bash
# See pod events
kubectl describe pod $POD_NAME -n grace-experiments

# Look for:
# Liveness probe failed
# Readiness probe failed
```

**Common causes:**
- Model still loading (wait longer)
- Server crashed (check logs)
- Port not accessible (check port 8000)

**Solution:**
```bash
# Check logs for errors
kubectl logs $POD_NAME -n grace-experiments | tail -50

# If model loading slow, increase initialDelaySeconds
kubectl edit deployment vllm-gemma-2b -n grace-experiments
# Change readinessProbe.initialDelaySeconds to higher value (e.g., 180)
```

### Can't Access from Mac

**Symptoms:**
```bash
curl http://localhost:8000/health
# curl: (7) Failed to connect to localhost port 8000
```

**Check:**
1. Is port-forward running?
   ```bash
   # Look for this process:
   ps aux | grep "port-forward"
   ```

2. Is pod running?
   ```bash
   kubectl get pods -n grace-experiments
   ```

3. Is service correct?
   ```bash
   kubectl get service vllm-gemma-2b -n grace-experiments
   ```

## Sequential Deployment Workflow

For deploying all 6 models one at a time:

```bash
# Model 1: Gemma-2B
kubectl apply -f kubernetes/vllm-deployment.yaml
# Wait for Ready
kubectl wait --for=condition=available deployment/vllm-gemma-2b -n grace-experiments --timeout=600s
# Run experiments...
# Delete when done
kubectl delete deployment vllm-gemma-2b -n grace-experiments

# Model 2: Gemma-9B
# Edit vllm-deployment.yaml: change model to google/gemma-2-9b-it
kubectl apply -f kubernetes/vllm-deployment.yaml
# Repeat process...

# Continue for all 6 models...
```

**Time per model**: ~1 day (setup 1hr + experiments ~4hrs + cleanup 5min)

## Advanced: Using Large Models

For Gemma-27B, Llama-70B, or GPT-OSS-120B:

```bash
# Use high-resource deployment
kubectl apply -f kubernetes/vllm-deployment-large.yaml

# IMPORTANT: Request exception FIRST via Matrix
# https://matrix.to/#/#nrp:matrix.org
```

This deployment uses:
- 2 GPUs (tensor parallelism)
- 128Gi RAM
- 32 CPU cores

**Requires exception approval** before deployment.

## Best Practices

### 1. Monitor Resource Usage

```bash
# Check every few hours
kubectl top pod $POD_NAME -n grace-experiments
kubectl exec -it $POD_NAME -n grace-experiments -- nvidia-smi
```

### 2. Clean Up Promptly

- Delete deployments when not in use
- Don't leave idle pods running (wastes resources)
- Sequential deployment prevents waste

### 3. Check Violations Daily

- Visit: https://nautilus.optiputer.net/violations
- Address any issues promptly
- Low GPU utilization = warning

### 4. Use Labels

All our manifests have clear labels:
```yaml
labels:
  app: vllm
  model: gemma-2b
  project: grace-experiments
```

Makes it easy to find and manage resources.

### 5. Save Outputs Locally

Don't store outputs in cluster - download to your Mac:
```bash
# If outputs are in pod, copy them:
kubectl cp $POD_NAME:/app/outputs ./outputs -n grace-experiments

# Better: Run scripts from Mac via port-forward
# Outputs save directly to your local disk
```

## Helpful Commands Cheat Sheet

```bash
# Get pod name
export POD_NAME=$(kubectl get pods -n grace-experiments -l app=vllm -o jsonpath='{.items[0].metadata.name}')

# Check everything
kubectl get all -n grace-experiments

# Describe pod (detailed info + events)
kubectl describe pod $POD_NAME -n grace-experiments

# Logs (real-time)
kubectl logs -f $POD_NAME -n grace-experiments

# Exec into pod (interactive shell)
kubectl exec -it $POD_NAME -n grace-experiments -- /bin/bash

# Port forward
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments

# Delete everything
kubectl delete namespace grace-experiments

# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB

# Top pods (CPU/Memory)
kubectl top pods -n grace-experiments

# GPU usage
kubectl exec -it $POD_NAME -n grace-experiments -- nvidia-smi
```

## Getting Help

### NRP Resources

1. **Documentation**: https://docs.nationalresearchplatform.org/
2. **Matrix Chat**: https://matrix.to/#/#nrp:matrix.org
3. **Portal**: https://nautilus.optiputer.net/
4. **Violations**: https://nautilus.optiputer.net/violations

### Ask in Matrix

When asking for help, provide:
- Namespace: `grace-experiments`
- Pod name: (from `kubectl get pods`)
- Error message: (from `kubectl describe pod` or logs)
- What you tried

Example:
```
Hi, I'm getting OOMKilled on my vLLM pod in grace-experiments namespace.
Pod: vllm-gemma-2b-xxx
Model: Gemma-2B
Resources: 32Gi RAM, 16 CPU, 1 GPU

Tried reducing max-model-len to 2048 but still crashing.
Logs show: [paste relevant logs]

Any suggestions?
```

## Next Steps

After successful K8s deployment:

1. ✅ **Test with 10 trials** - Validate end-to-end
2. 📊 **Run full Study 1** - All 300 trials
3. 📊 **Run full Study 2** - All 2,424 trials
4. 🔄 **Repeat for next model** - Sequential deployment
5. 📈 **Analyze results** - R scripts in Analysis/

See main README.md for full project workflow.

## Summary

**Deployment is straightforward:**
1. Create namespace
2. Deploy vLLM
3. Create service
4. Port-forward to Mac
5. Run experiments
6. Clean up

**Our manifests are compliant** - no exceptions needed for Gemma-2B!

**For help**: NRP Matrix chat is very responsive.

Good luck with your deployments! 🚀
