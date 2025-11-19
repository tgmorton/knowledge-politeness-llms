# GPU Contingency Plan - Alternative Hardware Strategies

**Date**: November 16, 2025  
**Purpose**: Define fallback strategies if A100 GPUs are unavailable

---

## Executive Summary

If A100 GPUs are unavailable or quota is denied, NRP has **extensive alternative GPU resources** that can still run the Grace Project experiments. This document outlines contingency plans using RTX 3090, RTX 4090, L40, and other available GPUs.

**Key Finding**: With quantization and strategic model selection, Grace Project is **fully viable** on non-A100 hardware.

---

## Available GPU Resources (Non-A100)

### Tier 1: High-End Consumer/Professional GPUs

| GPU Model | VRAM | Count | Locations | Suitable For |
|-----------|------|-------|-----------|--------------|
| **RTX 4090** | 24GB | ~5+ nodes | UNL, Calit2 | Most models with quantization |
| **RTX 3090** | 24GB | ~50+ nodes | Multiple sites | All models with quantization |
| **L40** | 48GB | ~17 nodes | SDSU Tide cluster | Large models, near-A100 performance |
| **RTX A6000** | 48GB | ~10+ nodes | Multiple sites | Large models, professional-grade |

### Tier 2: Mid-Range Professional GPUs

| GPU Model | VRAM | Count | Locations | Suitable For |
|-----------|------|-------|-----------|--------------|
| **A40** | 48GB | ~4+ nodes | Calit2, Montana | Large models |
| **RTX A5000** | 24GB | ~3+ nodes | CSUSB, CWRU | Medium models |
| **A10** | 24GB | ~20+ nodes | MGHPCC, UNL | Medium models |
| **L4** | 24GB | ~6 nodes | Fullerton | Medium models, efficient |

### Tier 3: Legacy High-Memory GPUs

| GPU Model | VRAM | Count | Locations | Suitable For |
|-----------|------|-------|-----------|--------------|
| **V100-32GB** | 32GB | ~8 nodes | Humboldt | Medium-large models |
| **V100-16GB** | 16GB | ~10+ nodes | Multiple | Small-medium models |
| **Titan RTX** | 24GB | ~3 nodes | UCSD, SDSC | Medium models |

---

## Contingency Strategy 1: RTX 3090 Cluster (Most Likely Fallback)

### GPU Characteristics
- **VRAM**: 24GB GDDR6X
- **Availability**: ~50+ nodes (most common GPU on NRP)
- **Performance**: ~70% of A100 throughput
- **Cost**: Free (no special quota needed)

### Recommended Model Lineup (Quantized)

| Model | Quantization | VRAM Required | GPUs Needed | Node Examples |
|-------|--------------|---------------|-------------|---------------|
| **Gemma-2B** | FP16 | ~5GB | 1x RTX 3090 | suncave-*, ry-gpu-* |
| **Gemma-9B** | AWQ 4-bit | ~6GB | 1x RTX 3090 | suncave-*, ry-gpu-* |
| **Gemma-27B** | AWQ 4-bit | ~18GB | 1x RTX 3090 | suncave-*, ry-gpu-* |
| **Llama-8B** | FP16 | ~18GB | 1x RTX 3090 | ry-gpu-*, k8s-3090-* |
| **Llama-70B** | AWQ 4-bit | ~45GB | **2x RTX 3090** | ry-gpu-01 + ry-gpu-02 |
| **GPT-OSS-20B** | AWQ 4-bit | ~8GB | 1x RTX 3090 | Any suncave node |

**Total**: 8 RTX 3090 GPUs needed (very achievable)

### Node Allocation Strategy

**Suncave Cluster** (SDSC - 18 nodes, all RTX 3090):
```yaml
# All co-located at SDSC, low latency
suncave-0, suncave-1, suncave-2, suncave-3, suncave-4, 
suncave-6, suncave-7, suncave-8, suncave-9, suncave-10,
suncave-11, suncave-12, suncave-13, suncave-14, suncave-15, suncave-17
```

**RY-GPU Cluster** (SDSC - 16 nodes, mostly RTX 3090):
```yaml
ry-gpu-01 through ry-gpu-14 (RTX 3090)
ry-gpu-15, ry-gpu-16 (RTX A4000 - 16GB, skip these)
```

**Deployment Plan**:
1. Gemma-2B → suncave-0 (1 GPU)
2. Gemma-9B → suncave-1 (1 GPU)
3. Gemma-27B → suncave-2 (1 GPU)
4. Llama-8B → suncave-3 (1 GPU)
5. Llama-70B → suncave-4 + suncave-6 (2 GPUs, tensor parallelism)
6. GPT-OSS-20B → suncave-7 (1 GPU)

**Node Selector Example**:
```yaml
nodeSelector:
  kubernetes.io/hostname: suncave-0
resources:
  requests:
    nvidia.com/gpu: 1
```

### Quantization Commands (vLLM with AWQ)

```bash
# Gemma-9B (AWQ 4-bit)
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/gemma-2-9b-it-AWQ \
  --quantization awq \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1

# Llama-70B (AWQ 4-bit, 2 GPUs)
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3-70B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2
```

**Quality Impact**: AWQ 4-bit typically has <3% quality degradation vs FP16

---

## Contingency Strategy 2: L40 Cluster (Best Non-A100 Option)

### GPU Characteristics
- **VRAM**: 48GB GDDR6
- **Availability**: 17 nodes at SDSU (rci-tide-gpu-01 through gpu-17)
- **Performance**: ~85% of A100 throughput
- **Cost**: Free (may need SDSU access)

### Recommended Model Lineup (Minimal Quantization)

| Model | Quantization | VRAM Required | GPUs Needed | Node Examples |
|-------|--------------|---------------|-------------|---------------|
| **Gemma-2B** | FP16 | ~5GB | 1x L40 | rci-tide-gpu-01 |
| **Gemma-9B** | FP16 | ~20GB | 1x L40 | rci-tide-gpu-02 |
| **Gemma-27B** | FP16 | ~55GB | **2x L40** | rci-tide-gpu-03 + gpu-04 |
| **Llama-8B** | FP16 | ~18GB | 1x L40 | rci-tide-gpu-05 |
| **Llama-70B** | AWQ 4-bit | ~45GB | 1x L40 | rci-tide-gpu-06 |
| **GPT-OSS-20B** | FP16 | ~15GB | 1x L40 | rci-tide-gpu-07 |

**Total**: 6 L40 GPUs needed

**Advantages**:
- ✅ 48GB VRAM allows FP16 for most models
- ✅ Only Llama-70B needs quantization
- ✅ All nodes co-located at SDSU
- ✅ Professional datacenter GPUs (not consumer)

**Disadvantages**:
- ⚠️ May need special access to SDSU Tide cluster
- ⚠️ Less availability than RTX 3090

---

## Contingency Strategy 3: RTX A6000 + A40 Mix (Professional GPUs)

### GPU Characteristics
- **RTX A6000**: 48GB, professional workstation GPU
- **A40**: 48GB, datacenter GPU
- **Availability**: ~14 combined nodes

### Available Nodes

**RTX A6000 Nodes**:
```yaml
k8s-a6000-01.calit2.optiputer.net
k8s-a6000-01.csus.edu
k8s-a6000-01.unm.edu
k8s-gpu-5.ucsc.edu
gpu00.nrp.hpc.udel.edu
nautilus02.hsrn.nyu.edu
nautilusg01.sci.cwru.edu
nrp-01.csumb.edu
nrp-a6000-01.csuchico.edu
hcc-prp-c5036.unl.edu
hcc-prp-c5038.unl.edu
```

**A40 Nodes**:
```yaml
k8s-a40-01.calit2.optiputer.net
k8s-usra-01.calit2.optiputer.net
rci-nautilus01.msu.montana.edu
```

### Recommended Model Lineup

| Model | Quantization | VRAM Required | GPUs Needed | Strategy |
|-------|--------------|---------------|-------------|----------|
| **Gemma-2B** | FP16 | ~5GB | 1x A6000/A40 | Any node |
| **Gemma-9B** | FP16 | ~20GB | 1x A6000/A40 | Any node |
| **Gemma-27B** | FP16 | ~55GB | **2x A6000/A40** | Tensor parallel |
| **Llama-8B** | FP16 | ~18GB | 1x A6000/A40 | Any node |
| **Llama-70B** | AWQ 4-bit | ~45GB | 1x A6000/A40 | Any node |
| **GPT-OSS-20B** | FP16 | ~15GB | 1x A6000/A40 | Any node |

**Total**: 7 GPUs needed (1 A6000/A40 each, except Gemma-27B needs 2)

**Advantages**:
- ✅ Professional/datacenter GPUs (reliable)
- ✅ 48GB VRAM (same as L40)
- ✅ Distributed across many sites (high availability)

**Disadvantages**:
- ⚠️ Nodes are geographically distributed (higher latency)
- ⚠️ Need to carefully select co-located nodes for multi-GPU

---

## Contingency Strategy 4: V100-32GB Cluster (Legacy but Capable)

### GPU Characteristics
- **VRAM**: 32GB HBM2
- **Availability**: 8 nodes at Humboldt (cph-dgx-node*)
- **Performance**: ~60% of A100 throughput
- **Cost**: Free

### Available Nodes
```yaml
cph-dgx-node1.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node2.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node5.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node6.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node7.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node8.humboldt.edu    Tesla-V100-SXM2-32GB
cph-dgx-node9.humboldt.edu    Tesla-V100-SXM2-32GB
v100-cc-star-01.noc.ucsb.edu  Tesla-V100-SXM2-32GB
```

### Recommended Model Lineup (Quantized)

| Model | Quantization | VRAM Required | GPUs Needed | Strategy |
|-------|--------------|---------------|-------------|----------|
| **Gemma-2B** | FP16 | ~5GB | 1x V100-32GB | Any node |
| **Gemma-9B** | AWQ 4-bit | ~6GB | 1x V100-32GB | Any node |
| **Gemma-27B** | AWQ 4-bit | ~18GB | 1x V100-32GB | Any node |
| **Llama-8B** | FP16 | ~18GB | 1x V100-32GB | Any node |
| **Llama-70B** | AWQ 4-bit | ~45GB | **2x V100-32GB** | Tensor parallel |
| **GPT-OSS-20B** | AWQ 4-bit | ~8GB | 1x V100-32GB | Any node |

**Total**: 7 V100-32GB GPUs needed

**Advantages**:
- ✅ All nodes at Humboldt (co-located)
- ✅ DGX nodes (likely 8 GPUs each)
- ✅ NVLink for fast multi-GPU communication

**Disadvantages**:
- ⚠️ Older architecture (slower than RTX 3090/4090)
- ⚠️ Only 32GB VRAM (requires more quantization)

---

## Contingency Strategy 5: Hybrid Multi-Site Deployment

### Concept
Use **best available GPU at each site** and distribute models geographically.

### Example Distribution

| Model | GPU Type | VRAM | Location | Node |
|-------|----------|------|----------|------|
| Gemma-2B | RTX 3090 | 24GB | SDSC | suncave-0 |
| Gemma-9B | RTX A6000 | 48GB | Calit2 | k8s-a6000-01.calit2 |
| Gemma-27B | 2x L40 | 48GB | SDSU | rci-tide-gpu-01 + gpu-02 |
| Llama-8B | RTX 4090 | 24GB | UNL | hcc-nrp-shor-c5226.unl.edu |
| Llama-70B | 2x A40 | 48GB | Montana | rci-nautilus01.msu.montana.edu |
| GPT-OSS-20B | RTX 3090 | 24GB | NYU | nautilus01.hsrn.nyu.edu |

**Advantages**:
- ✅ Uses best GPU type per site
- ✅ High availability (geographic distribution)
- ✅ Maximizes use of available resources

**Disadvantages**:
- ⚠️ Higher inter-pod latency (doesn't matter for your use case)
- ⚠️ More complex deployment (need site-specific node selectors)

---

## Quantization Strategy Reference

### AWQ 4-bit (Recommended)

**Pros**:
- ~4x memory reduction
- ~2x faster inference (less memory bandwidth)
- <3% quality degradation
- Well-supported in vLLM

**Cons**:
- Need pre-quantized models (TheBloke on HuggingFace)
- Slight quality loss

**vLLM Command**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3-70B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --max-model-len 4096
```

### GPTQ 4-bit (Alternative)

**Pros**:
- Similar to AWQ
- Wider model support

**Cons**:
- Slightly slower than AWQ
- Less vLLM optimization

### FP8 (If Supported by GPU)

**Pros**:
- 2x memory reduction
- Minimal quality loss (<1%)
- Fast on newer GPUs (RTX 4090, L40)

**Cons**:
- Requires Hopper/Ada architecture (RTX 4090, L40, H100)
- Not all models support FP8

---

## Model Selection for GPU-Constrained Scenarios

### Tier 1: Essential Models (Minimum Viable)

If extremely GPU-constrained, run only these:

1. **Gemma-2B** (baseline, 1 GPU)
2. **Gemma-9B** (mid-range, 1 GPU)
3. **Llama-8B** (alternative mid-range, 1 GPU)

**Total**: 3 GPUs (any 24GB+ GPU)

**Rationale**: Still provides model diversity (Google vs Meta), size comparison (2B vs 8B-9B)

### Tier 2: Recommended Minimum

Add these for better coverage:

4. **Gemma-27B** (larger Gemma, 1-2 GPUs)
5. **GPT-OSS-20B** (MoE architecture, 1 GPU)

**Total**: 5-6 GPUs

### Tier 3: Full Lineup

Add for complete comparison:

6. **Llama-70B** (large model, 2 GPUs with quantization)

**Total**: 7-8 GPUs

---

## GPU Memory Requirement Calculator

### Formula

```
Memory Required = (Params × Precision Bytes) + KV Cache + Overhead

Precision Bytes:
- FP16: 2 bytes/param
- AWQ 4-bit: 0.5 bytes/param
- GPTQ 4-bit: 0.5 bytes/param
- FP8: 1 byte/param

KV Cache: ~20-30% of model weights (depends on context length)
Overhead: ~2-5GB (vLLM, CUDA, etc.)
```

### Examples

**Llama-70B FP16**:
- Weights: 70B × 2 = 140GB
- KV Cache: 140GB × 0.25 = 35GB
- Overhead: 5GB
- **Total: 180GB** (needs 3x A100-80GB or 8x RTX 3090)

**Llama-70B AWQ 4-bit**:
- Weights: 70B × 0.5 = 35GB
- KV Cache: 35GB × 0.25 = 8.75GB
- Overhead: 5GB
- **Total: 48.75GB** (fits on 2x RTX 3090 or 1x L40/A6000)

**Gemma-27B AWQ 4-bit**:
- Weights: 27B × 0.5 = 13.5GB
- KV Cache: 13.5GB × 0.25 = 3.375GB
- Overhead: 3GB
- **Total: 19.875GB** (fits on 1x RTX 3090)

---

## Node Selector Templates for Each Contingency

### RTX 3090 (Suncave Cluster)

```yaml
nodeSelector:
  kubernetes.io/hostname: suncave-0
# Or label-based (if labels exist):
# nodeSelector:
#   nvidia.com/gpu.product: NVIDIA-GeForce-RTX-3090
```

### L40 (SDSU Tide Cluster)

```yaml
nodeSelector:
  kubernetes.io/hostname: rci-tide-gpu-01.sdsu.edu
# Or:
# nodeSelector:
#   nvidia.com/gpu.product: NVIDIA-L40
```

### RTX A6000 (Multiple Sites)

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - NVIDIA-RTX-A6000
```

### V100-32GB (Humboldt Cluster)

```yaml
nodeSelector:
  kubernetes.io/hostname: cph-dgx-node1.humboldt.edu
# Or:
# nodeSelector:
#   nvidia.com/gpu.product: Tesla-V100-SXM2-32GB
```

---

## Performance Comparison

### Inference Speed (Relative to A100-80GB)

| GPU | Memory | Throughput | Cost (Relative) | Availability |
|-----|--------|------------|-----------------|--------------|
| A100-80GB | 80GB | 100% | Baseline | Limited |
| L40 | 48GB | ~85% | Free | Medium (17 nodes) |
| RTX 4090 | 24GB | ~75% | Free | Low (5 nodes) |
| RTX 3090 | 24GB | ~70% | Free | High (50+ nodes) |
| A40 | 48GB | ~70% | Free | Medium (4 nodes) |
| RTX A6000 | 48GB | ~65% | Free | Medium (11 nodes) |
| V100-32GB | 32GB | ~60% | Free | Medium (8 nodes) |
| A10 | 24GB | ~50% | Free | High (20+ nodes) |

**Note**: All are free on NRP; "cost" is opportunity cost vs A100

### Expected Experiment Duration

**With A100-80GB** (baseline):
- Full experiments (all models): ~2-3 days

**With RTX 3090** (most likely fallback):
- Full experiments: ~3-4 days (40% longer)

**With L40**:
- Full experiments: ~2.5-3.5 days (25% longer)

**With V100-32GB**:
- Full experiments: ~4-5 days (60% longer)

**Key Insight**: Even with slower GPUs, experiments are still **very feasible** (days, not weeks)

---

## Recommended Fallback Priority

### Priority 1: L40 Cluster (SDSU)
- **Best balance**: High VRAM (48GB), good performance, co-located
- **Action**: Request access to SDSU Tide cluster

### Priority 2: RTX 3090 (Suncave/RY-GPU at SDSC)
- **Most available**: 50+ nodes, co-located clusters
- **Action**: Use with AWQ quantization

### Priority 3: RTX A6000 + A40 Mix
- **Professional GPUs**: Reliable, 48GB VRAM
- **Action**: Use for models that don't fit on RTX 3090

### Priority 4: V100-32GB (Humboldt)
- **Legacy but capable**: Co-located cluster, NVLink
- **Action**: Fallback if newer GPUs unavailable

### Priority 5: Hybrid Multi-Site
- **Last resort**: Mix different GPU types across sites
- **Action**: Only if other strategies fail

---

## Pre-Deployment Checklist

Before deploying on non-A100 hardware:

### 1. Verify Node Availability
```bash
# Check RTX 3090 nodes
kubectl get nodes -L nvidia.com/gpu.product | grep RTX-3090

# Check current allocation
kubectl describe node suncave-0 | grep -A 10 "Allocated resources"
```

### 2. Test Quantized Models Locally
```bash
# Download quantized model
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/gemma-2-9b-it-AWQ \
  --quantization awq

# Test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-2-9b-it","prompt":"Test","max_tokens":10}'
```

### 3. Create AWQ Model List

Pre-quantized models available on HuggingFace (TheBloke):
- `TheBloke/gemma-2-9b-it-AWQ`
- `TheBloke/gemma-2-27b-it-AWQ`
- `TheBloke/Meta-Llama-3-8B-Instruct-AWQ`
- `TheBloke/Meta-Llama-3-70B-Instruct-AWQ`

**If not available**: Use AutoAWQ to quantize yourself (add time for quantization)

### 4. Update Kubernetes Manifests

Add quantization flags to vLLM command:
```yaml
args:
- --model=/model-weights
- --quantization=awq  # Add this
- --dtype=half
- --max-model-len=4096
```

---

## Summary: What's Possible Without A100s

### Absolutely Viable ✅

Using RTX 3090 (most common GPU on NRP):
- All 6 models can run with AWQ 4-bit quantization
- Total: 8 RTX 3090 GPUs (highly available)
- Timeline: 3-4 days for full experiments
- Quality: <3% degradation vs full precision

### Recommended Approach

1. **First choice**: Request A100-80GB access
2. **Second choice**: Use L40 cluster (SDSU Tide)
3. **Third choice**: Use RTX 3090 (Suncave cluster at SDSC)
4. **All choices**: Viable and will produce high-quality results

### Key Takeaway

**Grace Project is NOT blocked by lack of A100 access.** NRP has abundant alternative GPU resources that, with quantization, can successfully run all experiments.

---

## Next Steps Based on GPU Availability

### If A100-80GB Available
→ Use Strategy from NRP_A100_NODES.md (minimal quantization)

### If A100-40GB Only
→ Use AWQ quantization for Llama-70B, rest FP16

### If No A100 Access
→ **Follow this document**:
1. Try L40 cluster first (48GB, best non-A100)
2. Fall back to RTX 3090 cluster (24GB, most available)
3. Use AWQ 4-bit for models >10B parameters
4. Expect ~40% longer runtime (still only days)

---

**Document Status**: ✅ Complete  
**Confidence Level**: High - Multiple viable fallback strategies  
**Blocking Issues**: None - Project is viable on any GPU tier
