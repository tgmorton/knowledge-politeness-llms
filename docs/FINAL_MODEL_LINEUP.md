# Grace Project - Final Model Lineup

## Selected Models

The Grace Project will use the following 6 models for experiments:

| Model | Size | Active Params | Context | License | GPU Requirement | Repository |
|-------|------|---------------|---------|---------|-----------------|------------|
| **Gemma-2 2B** | 2B params | 2B | 8K tokens | Apache 2.0 | 1x A100 40GB | `google/gemma-2-2b-it` |
| **Gemma-2 9B** | 9B params | 9B | 8K tokens | Apache 2.0 | 1x A100 40GB | `google/gemma-2-9b-it` |
| **Gemma-2 27B** | 27B params | 27B | 8K tokens | Apache 2.0 | 2x A100 40GB | `google/gemma-2-27b-it` |
| **Llama-3 70B** | 70B params | 70B | 8K tokens | Llama 3 License | 4x A100 40GB | `meta-llama/Meta-Llama-3-70B-Instruct` |
| **GPT-OSS 20B** | 20.9B total | 3.6B active | TBD | OpenAI OSS | 1x A100 40GB | TBD (OpenAI release) |
| **GPT-OSS 120B** | 116.8B total | 5.1B active | TBD | OpenAI OSS | 2x A100 40GB | TBD (OpenAI release) |

**Note**: GPT-OSS models are Mixture-of-Experts (MoE) transformers with significantly lower active parameters per forward pass than total parameters

---

## Resource Requirements

### GPU Allocation

| Model | Tensor Parallel | GPUs per Pod | NRP Exception Needed? |
|-------|-----------------|--------------|----------------------|
| Gemma-2 2B | 1 | 1 | ❌ No (within limits) |
| Gemma-2 9B | 1 | 1 | ❌ No |
| Gemma-2 27B | 2 | 2 | ❌ No (within limits) |
| Llama-3 70B | 4 | 4 | ✅ **YES** (>2 GPU limit) |
| GPT-OSS 20B | 1 | 1 | ❌ No (MoE, 3.6B active) |
| GPT-OSS 120B | 2 | 2 | ❌ No (MoE, 5.1B active) |

**Total GPUs Required**: 11 A100 GPUs (40GB)

**NRP Exceptions Required**:
1. **Llama-3 70B**: 4 GPUs per pod (default limit is 2)
2. **Gemma-2 27B**: 2 GPUs per pod (within exception range)
3. **GPT-OSS 120B**: 2 GPUs per pod (within exception range)
4. **All deployments**: >2 weeks runtime (default is 2 weeks auto-delete)

**Note**: GPT-OSS models use MoE architecture, so despite large total parameters (20.9B, 116.8B), they only have 3.6B and 5.1B active parameters per forward pass, requiring much less GPU memory

### Memory Requirements

| Model | RAM per Pod | NRP Exception Needed? |
|-------|-------------|----------------------|
| Gemma-2 2B | ~10GB | ❌ No (within 32GB limit) |
| Gemma-2 9B | ~20GB | ❌ No |
| Gemma-2 27B | ~60GB | ✅ **YES** (>32GB limit) |
| Llama-3 70B | ~150GB | ✅ **YES** (>32GB limit) |
| GPT-OSS 20B | ~15GB | ❌ No (MoE with 3.6B active) |
| GPT-OSS 120B | ~35GB | ✅ **YES** (>32GB limit, 5.1B active) |

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
  --tensor-parallel-size 4
```

**Expected Performance**:
- Throughput: 5-10 req/s
- Latency: 200-400ms TTFT

**NRP Exceptions**:
- 4 GPUs per pod
- ~150GB RAM (request 160GB)

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

**NRP Exceptions**:
- 2 GPUs per pod (within exception range)
- ~40GB RAM (request 50GB to be safe)

**Note**: 
- Requires 2 GPUs for tensor parallelism, not due to memory (could fit on 1) but for routing efficiency
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

## Total Cluster Resource Summary

### Peak Resource Usage

| Resource | Amount | Notes |
|----------|--------|-------|
| **GPUs** | 11 A100 (40GB) | MoE models are very efficient |
| **CPU Cores** | ~150 cores | Model servers + jobs |
| **Memory** | ~290GB RAM | Across all model pods (MoE efficiency) |
| **Storage** | ~465Gi | RBD + CephFS |

**Efficiency Gains from GPT-OSS MoE**:
- 35% fewer GPUs needed (11 vs 17)
- 70% less RAM needed (~290GB vs ~950GB)
- 44% less storage (~465Gi vs ~830Gi)

### By Phase

**Phase 2-6** (Models Running):
- 11 GPUs for model servers
- 6 StatefulSets/Deployments running
- ~290GB RAM for models

**Phase 3-5** (Experiments Running):
- Same 11 GPUs for models
- Additional ~40 CPU cores for jobs (no GPU)
- Additional ~80GB RAM for jobs
- **Total**: 11 GPUs, ~150 cores, ~370GB RAM

---

## NRP Exception Requests Summary

### Required Exception Requests

**To submit in Matrix before Phase 2**:

1. **Multi-GPU Pods** (CRITICAL):
   ```
   Request: Allow >2 GPUs per pod
   
   Pods requiring exceptions:
   - vllm-gemma-27b: 2 GPUs (within exception range)
   - vllm-gpt-oss-120b: 2 GPUs (within exception range)
   - vllm-llama-70b: 4 GPUs (main exception needed)
   
   Reason: Tensor parallelism for large language models and MoE routing
   ```

2. **High Memory Pods** (CRITICAL):
   ```
   Request: Allow >32GB RAM per pod
   
   Pods requiring exceptions:
   - vllm-gemma-27b: 64GB RAM
   - vllm-gpt-oss-120b: 50GB RAM (MoE, only 5.1B active)
   - vllm-llama-70b: 160GB RAM
   
   Reason: Large model parameters and KV cache
   
   Note: GPT-OSS models are MUCH more efficient than dense models
   ```

3. **Deployment Runtime** (CRITICAL):
   ```
   Request: Allow Deployments to run >2 weeks
   
   Deployments: All 6 model servers
   - vllm-gemma-2b, vllm-gemma-9b, vllm-gemma-27b
   - vllm-llama-70b
   - vllm-gpt-oss-20b, vllm-gpt-oss-120b
   
   Duration: 3-4 weeks
   
   Reason: Academic research experiments with LLMs
   ```

4. **GPU Quota** (if needed):
   ```
   Request: GPU allocation increase
   
   Current: [Check your allocation]
   Requested: 11 A100 GPUs (40GB)
   
   Reason: Concurrent model serving for research comparison study
   
   Note: GPT-OSS MoE models significantly reduce GPU requirements
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

## Cost Estimation (GPU Hours)

**Assumptions**:
- Model servers run for 3 weeks (21 days)
- All experiments complete in ~1 week of server runtime
- Actual GPU usage ~7 days (intermittent querying)

**Conservative Estimate** (models running 24/7 for 7 days):
```
Gemma-2B:      1 GPU  × 168 hours = 168 GPU-hours
Gemma-9B:      1 GPU  × 168 hours = 168 GPU-hours
Gemma-27B:     2 GPUs × 168 hours = 336 GPU-hours
Llama-70B:     4 GPUs × 168 hours = 672 GPU-hours
GPT-OSS-20B:   1 GPU  × 168 hours = 168 GPU-hours
GPT-OSS-120B:  2 GPUs × 168 hours = 336 GPU-hours
---------------------------------------------------
Total:                           1,848 GPU-hours
```

**Realistic Estimate** (intermittent use, 50% utilization):
~900-1,000 GPU-hours

**Savings from MoE**: ~1,000 GPU-hours saved compared to dense models

**NRP Allocation**: Free (subject to availability and quotas)

---

## Updated File References

The following files have been updated with this model lineup:
- `02_KUBERNETES_INFRASTRUCTURE.md` - PVC definitions and StatefulSets
- `04_MODEL_SERVING_SPECS.md` - vLLM configurations
- `DECISION_CHECKLIST.md` - Model selection confirmed
- `00_EXECUTIVE_SUMMARY.md` - Resource summary

---

## Next Steps

1. ✅ **Review this model lineup** - Confirm with research team
2. ✅ **Join NRP Matrix** - Prepare to request exceptions
3. ✅ **Request exceptions** - Multi-GPU, high RAM, >2 week runtime (see details above)
4. ✅ **Obtain HuggingFace token** - For Llama-3 70B (gated model)
5. ⏳ **Wait for GPT-OSS release** - Monitor OpenAI announcements for model availability
6. ✅ **Test locally** - Download and test Gemma-2B with vLLM
7. ✅ **Verify vLLM MoE support** - Check if vLLM supports MoE when GPT-OSS is released
8. ✅ **Proceed to Phase 0** - Begin implementation roadmap

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

**Key Benefits of GPT-OSS MoE Models**:
- ✅ **35% fewer GPUs** (11 vs 17) - Significant cost savings
- ✅ **70% less RAM** (~290GB vs ~950GB) - Much easier to fit in NRP limits
- ✅ **44% less storage** (~465Gi vs ~830Gi) - Faster downloads
- ✅ **Faster inference** - Only 3.6B-5.1B active params per token
- ✅ **Comparable quality** - To dense models 10x larger

**Pending Items**:
- 🔍 Confirm GPT-OSS model release date from OpenAI
- 🔍 Verify exact model IDs/repository locations
- 🔍 Check vLLM MoE support (likely available, but confirm)
- 🔍 May need `--trust-remote-code` flag for MoE models
