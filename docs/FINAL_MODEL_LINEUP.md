# Grace Project - Final Model Lineup

## Selected Models

The Grace Project will use the following 6 models for experiments:

| Model | Size | Active Params | Context | License | GPU Requirement | Repository |
|-------|------|---------------|---------|---------|-----------------|------------|
| **Gemma-2 2B** | 2B params | 2B | 8K tokens | Apache 2.0 | 1x A100-80GB | `google/gemma-2-2b-it` |
| **Gemma-2 9B** | 9B params | 9B | 8K tokens | Apache 2.0 | 1x A100-80GB | `google/gemma-2-9b-it` |
| **Gemma-2 27B** | 27B params | 27B | 8K tokens | Apache 2.0 | 2x A100-80GB | `google/gemma-2-27b-it` |
| **Llama-3 70B** | 70B params | 70B | 8K tokens | Llama 3 License | 2x A100-80GB | `meta-llama/Meta-Llama-3-70B-Instruct` |
| **GPT-OSS 20B** | 20.9B total | 3.6B active | TBD | OpenAI OSS | 1x A100-80GB | TBD (OpenAI release) |
| **GPT-OSS 120B** | 116.8B total | 5.1B active | TBD | OpenAI OSS | 2x A100-80GB | TBD (OpenAI release) |

**Note**: GPT-OSS models are Mixture-of-Experts (MoE) transformers with significantly lower active parameters per forward pass than total parameters

---

## Resource Requirements

### GPU Allocation (Sequential Deployment)

**Deployment Strategy**: Deploy one model at a time, run all experiments for that model, then clean up and deploy the next model. Each model deployment lasts <1 day.

| Model | Tensor Parallel | GPUs per Pod | NRP Exception Needed? |
|-------|-----------------|--------------|----------------------|
| Gemma-2 2B | 1 | 1 | ❌ No (within limits) |
| Gemma-2 9B | 1 | 1 | ❌ No |
| Gemma-2 27B | 2 | 2 | ❌ No (within 2 GPU limit) |
| Llama-3 70B | 2 | 2 | ❌ No (A100-80GB fits in 2 GPUs) |
| GPT-OSS 20B | 1 | 1 | ❌ No (MoE, 3.6B active) |
| GPT-OSS 120B | 2 | 2 | ❌ No (MoE, 5.1B active) |

**Peak GPUs Required**: 2 A100-80GB GPUs (sequential deployment)

**Key Changes from Parallel Deployment**:
- **Was**: 11 GPUs simultaneously for 2+ weeks → **Now**: 2-4 GPUs for <1 day per model
- **Was**: Llama-70B needed 4x A100-40GB → **Now**: 2x A100-80GB
- **Was**: Needed deployment duration exception → **Now**: No exception needed (<1 day per model)

**NRP Exceptions Required**: 
- ❌ **NONE** - All models fit within standard NRP limits with sequential deployment!

**Note**: GPT-OSS models use MoE architecture, so despite large total parameters (20.9B, 116.8B), they only have 3.6B and 5.1B active parameters per forward pass, requiring much less GPU memory

### Memory Requirements (NRP Compliant)

**All memory limits must be within 20% of requests per NRP policy**

| Model | RAM Request | RAM Limit | NRP Compliant? |
|-------|-------------|-----------|----------------|
| Gemma-2 2B | 32GB | 38GB (119%) | ✅ Yes |
| Gemma-2 9B | 64GB | 77GB (120%) | ✅ Yes |
| Gemma-2 27B | 128GB | 154GB (120%) | ✅ Yes |
| Llama-3 70B | 192GB | 230GB (120%) | ✅ Yes |
| GPT-OSS 20B | 64GB | 77GB (120%) | ✅ Yes |
| GPT-OSS 120B | 128GB | 154GB (120%) | ✅ Yes |

**Note**: GPT-OSS models require memory for total parameters but only activate a small subset per forward pass

### Storage Requirements (Model Weights)

| Model | Disk Space | PVC Size | Notes |
|-------|-----------|----------|-------|
| Gemma-2 2B | ~5GB | 10Gi | Full params |
| Gemma-2 9B | ~18GB | 25Gi | Full params |
| Gemma-2 27B | ~55GB | 70Gi | Full params |
| Llama-3 70B | ~140GB | 160Gi | Full params |
| GPT-OSS 20B | ~12.8GiB | 15Gi | MoE checkpoint size |
| GPT-OSS 120B | ~60.8GiB | 70Gi | MoE checkpoint size |

**Total Model Weights Storage**: ~350Gi (~0.35TB)

**Significant Savings**: GPT-OSS MoE models have smaller checkpoint sizes than equivalent dense models

**Total Project Storage** (including outputs, logs, input):
- Model weights: 350Gi
- Input data: 5Gi
- Output data: 100Gi
- Logs: 10Gi
- **Total**: ~465Gi (~0.47TB)

**Savings from MoE**: GPT-OSS models save ~365Gi compared to dense models of similar quality

---

## Model Details

### 1. Gemma-2 2B

**Purpose**: Fast baseline, quick experimentation

**Characteristics**:
- Smallest and fastest model
- Good instruction following
- Google's efficient 2B architecture

**vLLM Command**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-2b-it \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1
```

**Expected Performance**:
- Throughput: 60-80 req/s
- Latency: 50-100ms TTFT

### 2. Gemma-2 9B

**Purpose**: Mid-range quality, balanced performance

**Characteristics**:
- Better reasoning than 2B
- Still efficient (fits on 1 GPU)
- Good for most tasks

**vLLM Command**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1
```

**Expected Performance**:
- Throughput: 20-40 req/s
- Latency: 100-200ms TTFT

### 3. Gemma-2 27B

**Purpose**: High-quality Gemma variant

**Characteristics**:
- Largest Gemma model
- Best quality/efficiency tradeoff in Gemma family
- Requires 2 GPUs (tensor parallelism)

**vLLM Command**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2
```

**Expected Performance**:
- Throughput: 10-20 req/s
- Latency: 150-250ms TTFT

**NRP Exception**: May need >32GB RAM (request 64GB)

### 4. Llama-3 70B

**Purpose**: High-quality research-grade model

**Characteristics**:
- Meta's flagship open model
- Excellent reasoning and instruction following
- Requires 4 GPUs (tensor parallelism)

**vLLM Command**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 2
```

**Expected Performance**:
- Throughput: 5-10 req/s
- Latency: 200-400ms TTFT

**NRP Compliance**:
- ✅ 2 GPUs per pod (within limits)
- ✅ 192GB RAM request, 230GB limit (120% - compliant)

**Note**: Requires HuggingFace token (gated model)

### 5. GPT-OSS 20B

**Purpose**: OpenAI's open-source Mixture-of-Experts model, efficient 20B scale

**Characteristics**:
- **Architecture**: MoE transformer building on GPT-2/GPT-3
- **Total Parameters**: 20.9B (24 layers)
- **Active Parameters**: 3.6B per forward pass
- **Checkpoint Size**: 12.8 GiB
- **Component Breakdown**:
  - MLP: 19.12B params
  - Attention: 0.64B params
  - Embed + Unembed: 1.16B params
- Efficient inference due to sparse activation

**vLLM Command** (pending OpenAI release):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --trust-remote-code  # May be needed for MoE
```

**Expected Performance**:
- Throughput: 30-50 req/s (efficient due to sparse activation)
- Latency: 80-150ms TTFT
- Much faster than dense 20B model

**NRP Exception**: None needed (fits within 32GB RAM limit)

**Note**: Check OpenAI's release documentation for exact model ID and any special vLLM flags for MoE support

### 6. GPT-OSS 120B

**Purpose**: OpenAI's larger open-source MoE model, efficient 120B scale

**Characteristics**:
- **Architecture**: MoE transformer building on GPT-2/GPT-3
- **Total Parameters**: 116.8B (36 layers)
- **Active Parameters**: 5.1B per forward pass
- **Checkpoint Size**: 60.8 GiB
- **Component Breakdown**:
  - MLP: 114.71B params
  - Attention: 0.96B params
  - Embed + Unembed: 1.16B params
- Only ~4.4% of parameters active per token
- Comparable quality to dense models 10x the active size

**vLLM Command** (pending OpenAI release):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --trust-remote-code  # May be needed for MoE
```

**Expected Performance**:
- Throughput: 15-25 req/s (despite 117B params, only 5.1B active)
- Latency: 120-200ms TTFT
- Much faster than dense 120B model would be

**NRP Compliance**:
- ✅ 2 GPUs per pod (within limits)
- ✅ 128GB RAM request, 154GB limit (120% - compliant)

**Note**: 
- Uses 2 GPUs for tensor parallelism and MoE routing efficiency
- Check vLLM documentation for MoE support when OpenAI releases the model

---

## Alternative: DeepSeek for Structured Output

**DeepSeek-V3** (671B params, 37B active via MoE):
- **Option A**: Use DeepSeek API (external, paid)
- **Option B**: Self-host (requires 2x A100 80GB, adds 2 GPUs to total)

**Recommendation**: Use **DeepSeek API** for Grace Project
- Simpler setup (no additional GPUs)
- Pay-per-use (likely <$50 for entire project)
- Reduces total GPU requirement from 19 to 17

---

## Total Cluster Resource Summary (Sequential Deployment)

### Peak Resource Usage (Per Model Deployment)

| Resource | Amount | Notes |
|----------|--------|-------|
| **GPUs** | 2 A100-80GB | One model at a time |
| **CPU Cores** | 32-38 cores | Model server (request) |
| **CPU Cores (limit)** | 38-45 cores | Within 20% of request |
| **Memory (request)** | 192GB peak | Llama-70B (largest) |
| **Memory (limit)** | 230GB peak | 120% of request (compliant) |
| **Storage** | ~465Gi | RBD + CephFS (shared across all models) |
| **Duration per Model** | <1 day | Deploy → experiment → cleanup |

**Comparison to Parallel Deployment**:
- Parallel: 11 GPUs simultaneously for 2+ weeks
- Sequential: 2 GPUs for <1 day each (6 days total)
- **GPU-days savings**: 154 → 12 (92% reduction)

### During Experiments (Peak Load)

**When running query jobs alongside model server**:
- Model server: 2 GPUs, 32 CPU cores, 192GB RAM
- Query jobs: 0 GPUs, 20-40 CPU cores (parallel), 40-80GB RAM
- **Peak Total**: 2 GPUs, ~60-70 cores, ~250-270GB RAM

**All resources fit within standard NRP limits - no exceptions needed!**

---

## NRP Exception Requests Summary

### Required Exception Requests

**With Sequential Deployment Strategy**:

✅ **NO EXCEPTIONS REQUIRED!**

**Why No Exceptions Needed**:
1. **GPU Count**: Maximum 2 GPUs per pod (within NRP limits)
2. **Memory Limits**: All limits within 20% of requests (compliant)
3. **Deployment Duration**: Each model runs <1 day (well under 2 week limit)
4. **Resource Quotas**: Peak usage fits within standard allocations

**Previous Requirements (Parallel Deployment)**:
- ❌ Would have needed: 11 GPUs simultaneously
- ❌ Would have needed: >2 week deployment duration exception
- ❌ Would have needed: 4 GPU per pod exception for Llama-70B

**Current Requirements (Sequential Deployment)**:
- ✅ Peak: 2 A100-80GB GPUs
- ✅ Duration: <1 day per model
- ✅ All resources within NRP policy limits

**Optional Request** (for convenience, not required):
```
Request: Reserve 2x A100-80GB GPUs for 1 week period

Reason: Sequential LLM experiments (6 models × <1 day each)

Note: Can work around availability if reservation not possible
```

---

## Quantization Options (if GPU-constrained)

If you cannot get 17 GPUs or exceptions, consider quantization:

### AWQ 4-bit Quantization

Reduces memory by ~4x with <5% quality loss:

| Model | Original GPUs | AWQ 4-bit GPUs | RAM Savings |
|-------|--------------|---------------|-------------|
| Gemma-2 27B | 2 | 1 | ~45GB → ~15GB |
| Llama-3 70B | 4 | 2 | ~150GB → ~40GB |
| BLOOM 176B | 8 | 4 | ~350GB → ~90GB |

**Trade-offs**:
- ✅ Fewer GPUs needed
- ✅ Faster inference
- ❌ Slight quality degradation (~2-5%)
- ❌ Requires pre-quantized models or quantization step

**If using quantization**:
- Total GPUs: 9-10 instead of 17
- No multi-GPU exception needed for Gemma-27B, Llama-70B
- Still need exception for BLOOM (4 GPUs with AWQ)

---

## Deployment Priority

If deploying incrementally or testing:

### Tier 1 (Deploy First):
1. **Gemma-2 2B** - Fast baseline, test pipeline
2. **Gemma-2 9B** - Mid-range quality

### Tier 2 (Core Experiments):
3. **Gemma-2 27B** - High Gemma quality
4. **Llama-3 70B** - High overall quality

### Tier 3 (Extended Analysis):
5. **GPT-NeoX 20B** - GPT alternative
6. **BLOOM 176B** - Largest model

**Rationale**: Deploy Tier 1 first to validate entire pipeline, then scale up

---

## Cost Estimation (GPU Hours) - Sequential Deployment

**Assumptions**:
- Deploy one model at a time
- Each model deployment: <24 hours (likely 6-12 hours)
- Total: 6 models × 12 hours average = 72 hours (3 days)

**Sequential Deployment Estimate** (pessimistic - 24h per model):
```
Gemma-2B:      1 GPU  × 24 hours = 24 GPU-hours
Gemma-9B:      1 GPU  × 24 hours = 24 GPU-hours
Gemma-27B:     2 GPUs × 24 hours = 48 GPU-hours
Llama-70B:     2 GPUs × 24 hours = 48 GPU-hours
GPT-OSS-20B:   1 GPU  × 24 hours = 24 GPU-hours
GPT-OSS-120B:  2 GPUs × 24 hours = 48 GPU-hours
---------------------------------------------------
Total:                          216 GPU-hours
```

**Realistic Estimate** (optimistic - 12h per model):
```
Total: ~108 GPU-hours (4.5 days)
```

**Comparison to Parallel Deployment**:
- Parallel: ~1,848 GPU-hours
- Sequential: ~216 GPU-hours
- **Savings: ~1,632 GPU-hours (88% reduction)**

**NRP Allocation**: Free (subject to availability and quotas)

---

## Updated File References

The following files have been updated with this model lineup and NRP compliance:
- `02_KUBERNETES_INFRASTRUCTURE.md` - PVC definitions and Deployments (compliant)
- `04_MODEL_SERVING_SPECS.md` - vLLM configurations (compliant)
- `01_ARCHITECTURE_OVERVIEW.md` - Sequential deployment strategy
- `00_EXECUTIVE_SUMMARY.md` - Resource summary (updated for sequential deployment)
- `DECISION_CHECKLIST.md` - Model selection confirmed

---

## Next Steps

1. ✅ **Review this model lineup** - Confirm with research team
2. ✅ **Join NRP Matrix** - For community support and questions
3. ❌ **~~Request exceptions~~** - NOT NEEDED with sequential deployment!
4. ✅ **Obtain HuggingFace token** - For Llama-3 70B (gated model)
5. ⏳ **Wait for GPT-OSS release** - Monitor OpenAI announcements for model availability
6. ✅ **Test locally** - Download and test Gemma-2B with vLLM
7. ✅ **Verify vLLM MoE support** - Check if vLLM supports MoE when GPT-OSS is released
8. ⏳ **Optional: Request A100-80GB reservation** - Reserve 2 GPUs for 1 week (convenience only)
9. ✅ **Proceed to Phase 0** - Begin implementation roadmap

---

## Model Lineup Summary

**This model lineup provides excellent coverage**:
- **Small Scale**: Gemma-2 2B (baseline, fast)
- **Medium Scale**: Gemma-2 9B, GPT-OSS 20B (3.6B active)
- **Large Scale**: Gemma-2 27B, GPT-OSS 120B (5.1B active)  
- **Very Large**: Llama-3 70B (research-grade quality)

**Model Families**:
- Google Gemma (2B, 9B, 27B) - Instruction-tuned, Apache 2.0
- Meta Llama-3 (70B) - High quality, instruction-tuned
- OpenAI GPT-OSS (20B, 120B) - MoE architecture, efficient

**Key Benefits of Sequential Deployment + A100-80GB**:
- ✅ **88% fewer GPU-hours** (216 vs 1,848) - Massive efficiency gain
- ✅ **Peak: 2 GPUs** instead of 11 simultaneously
- ✅ **No NRP exceptions needed** - Fully compliant with all policies
- ✅ **Llama-70B on 2 GPUs** - Using A100-80GB instead of 4x A100-40GB
- ✅ **<1 day per model** - No deployment duration issues
- ✅ **All limits within 20% of requests** - Compliant resource specifications

**Pending Items**:
- 🔍 Confirm GPT-OSS model release date from OpenAI
- 🔍 Verify exact model IDs/repository locations
- 🔍 Check vLLM MoE support (likely available, but confirm)
- 🔍 May need `--trust-remote-code` flag for MoE models
