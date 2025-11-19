# GPU Selection Guide for Grace Project

## Available Configurations

### Configuration 1: A100-80GB (Primary - Best Quality)
- **Files:** `vllm-deployment.yaml`, `job-exp2-template.yaml`
- **Node Count:** 20 nodes available
- **VRAM:** 80GB per GPU
- **Best For:** All models at full precision

**Models Supported:**
- ✅ Gemma-2B (1 GPU, fp16)
- ✅ Gemma-9B (1 GPU, fp16)
- ✅ Gemma-27B (2 GPUs, fp16, tensor parallelism)
- ✅ Llama-70B (2 GPUs, fp16, tensor parallelism)
- ✅ GPT-OSS-20B (1 GPU, fp16)
- ✅ GPT-OSS-120B (2 GPUs, fp16, tensor parallelism)

**When to Use:**
- Production runs for publication
- Need highest quality results
- Full precision required

**Deploy:**
```bash
kubectl apply -f kubernetes/vllm-deployment.yaml
kubectl apply -f kubernetes/job-exp2-template.yaml
```

---

### Configuration 2: RTX 3090 (Fallback - High Availability)
- **Files:** `vllm-deployment-rtx3090.yaml`, `job-exp2-rtx3090-template.yaml`
- **Node Count:** 48 nodes available (most common!)
- **VRAM:** 24GB per GPU
- **Best For:** Testing, development, smaller models

**Models Supported:**
- ✅ Gemma-2B (1 GPU, fp16, no issues)
- ✅ Gemma-9B (1 GPU, fp16 or AWQ 4-bit)
- ⚠️  Gemma-27B (1 GPU, AWQ 4-bit required)
- ❌ Llama-70B (need heavy quantization or A100)

**When to Use:**
- A100s are busy/unavailable
- Testing and development
- Faster deployment (more nodes available)
- Cost-conscious runs (if quota applies)

**Deploy:**
```bash
kubectl apply -f kubernetes/vllm-deployment-rtx3090.yaml
kubectl apply -f kubernetes/job-exp2-rtx3090-template.yaml
```

---

## Quick Decision Tree

**Question 1: Is this for publication-quality results?**
- YES → Use A100-80GB
- NO → Continue to Question 2

**Question 2: Are A100 nodes available?**
```bash
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB | grep Ready | wc -l
```
- If > 0 → Use A100-80GB
- If 0 → Use RTX 3090

**Question 3: Which model are you running?**
- Gemma-2B or Gemma-9B → RTX 3090 is fine
- Gemma-27B → Prefer A100, RTX 3090 needs quantization
- Llama-70B → Must use A100

---

## Resource Comparison

| GPU Type    | VRAM  | Nodes | Memory Request | CPU Request | Best Use Case              |
|-------------|-------|-------|----------------|-------------|----------------------------|
| A100-80GB   | 80GB  | 20    | 32Gi           | 16 cores    | Production, large models   |
| RTX 3090    | 24GB  | 48    | 16Gi           | 8 cores     | Testing, small models      |

---

## Switching Between Configurations

### From A100 to RTX 3090
```bash
# Delete A100 deployment
kubectl delete deployment vllm-gemma-2b -n lemn-lab

# Deploy RTX 3090 version
kubectl apply -f kubernetes/vllm-deployment-rtx3090.yaml

# Update jobs to use RTX 3090
kubectl apply -f kubernetes/job-exp2-rtx3090-template.yaml
```

### From RTX 3090 to A100
```bash
# Delete RTX 3090 deployment
kubectl delete deployment vllm-gemma-2b-rtx3090 -n lemn-lab

# Deploy A100 version
kubectl apply -f kubernetes/vllm-deployment.yaml

# Update jobs to use A100
kubectl apply -f kubernetes/job-exp2-template.yaml
```

---

## Quantization Notes

### When to Use Quantization

**RTX 3090 (24GB VRAM):**
- Gemma-2B: No quantization needed
- Gemma-9B: Optional (fp16 fits, but AWQ 4-bit gives headroom)
- Gemma-27B: **Required** (AWQ 4-bit)

**A100-80GB:**
- No quantization needed for any model in our lineup

### How to Enable Quantization

Edit deployment YAML, add to `args:`:
```yaml
args:
  - --model=google/gemma-2-27b-it
  - --quantization=awq
```

**Note:** Model must have pre-quantized AWQ weights available on HuggingFace.

---

## Troubleshooting

### Pod Stuck in Pending (A100)
**Symptom:** Pod shows "Pending" for >5 minutes

**Check:**
```bash
kubectl describe pod <pod-name> -n lemn-lab | grep -A 10 Events
```

**Likely Cause:** No A100 nodes available

**Solution:** Switch to RTX 3090 configuration

### Out of Memory (RTX 3090)
**Symptom:** Pod crashes with OOMKilled

**Solution:**
1. Enable quantization (add `--quantization=awq`)
2. Reduce `--max-model-len` (from 4096 to 2048)
3. Switch to A100 configuration

### Model Loading Slow
**Both GPUs:** First load downloads from HuggingFace (~5-10 min for large models)

**Solution:** Be patient on first run, subsequent runs are cached

---

## Recommendations

**For Phase 1 Testing:**
- Start with RTX 3090 (Gemma-2B)
- Validate entire pipeline works
- Switch to A100 for larger models

**For Phase 2+ Production:**
- Use A100-80GB for all models
- Keep RTX 3090 configs as fallback
- Document which GPU was used for each experiment

---

## Node Availability Check

```bash
# Check A100 availability
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB -o wide

# Check RTX 3090 availability
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090 -o wide

# Run full inventory
./scripts/inventory_cluster_gpus.sh
```
