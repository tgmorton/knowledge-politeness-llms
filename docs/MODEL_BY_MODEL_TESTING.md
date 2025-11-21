# Model-by-Model Testing Guide

Systematic testing of each model locally before K8s deployment.

---

## Testing Order (Easiest → Hardest)

### ✅ Tier 1: Single GPU, Small Models (Easy Local Testing)
1. **Gemma-2 2B** - Smallest, fastest, baseline
2. **Llama-3.2 3B** - Small Llama variant
3. **Gemma-2 9B** - Medium Gemma
4. **Llama-3.1 8B** - Medium Llama

### ⚠️ Tier 2: Multi-GPU Models (K8s Only)
5. **Gemma-2 27B** - 2 GPUs required
6. **Llama-3.1 70B** - 4 GPUs required
7. **DeepSeek-R1 70B** - 4 GPUs required + reasoning

---

## Model 1: Gemma-2 2B ✅ BASELINE

**Purpose:** Smallest model, establishes baseline, validates setup

**Specs:**
- Size: ~5GB
- GPUs: 1x (any 24GB+ GPU)
- VRAM: ~8GB used
- Speed: ~2-5 sec/query

### Setup

```bash
# Terminal 1: Start vLLM
source venv-grace/bin/activate
vllm serve google/gemma-2-2b-it \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

**Wait for:** `Application startup complete`

**Check GPU usage:**
```bash
nvidia-smi
# Should show ~8-10GB VRAM used
```

### Test Experiment 1

```bash
# Terminal 2: Quick test (5 trials)
source venv-grace/bin/activate
./scripts/run_local_exp1.sh gemma-2b-rtx3090 5
```

**Expected output:**
```
outputs/results/study1_exp1_gemma-2b-rtx3090_TIMESTAMP.json
outputs/results/study2_exp1_gemma-2b-rtx3090_TIMESTAMP.json
```

### Validate Output

```bash
# Check JSON is valid
python3 -c "
import json
with open('outputs/results/study1_exp1_gemma-2b-rtx3090_*.json') as f:
    data = json.load(f)
    print(f'✅ Valid JSON with {len(data)} results')
    print(f'✅ Fields: {list(data[0].keys())}')
    print(f'✅ Sample response: {data[0][\"response\"][:100]}...')
"
```

### Success Criteria
- ✅ vLLM starts without errors
- ✅ Both experiments complete
- ✅ JSON output is valid
- ✅ Responses look reasonable (not gibberish)
- ✅ All expected fields present

### Common Issues

**Issue:** `CUDA out of memory`
- **Fix:** Reduce `--max-model-len 2048` or `--gpu-memory-utilization 0.8`

**Issue:** Responses are all errors
- **Check:** `curl http://localhost:8000/health`
- **Check:** vLLM server logs in Terminal 1

---

## Model 2: Llama-3.2 3B

**Purpose:** Test Llama family, cross-validate against Gemma-2B

**Specs:**
- Size: ~6GB
- GPUs: 1x
- VRAM: ~10GB used
- Speed: ~3-6 sec/query

### Setup

```bash
# Terminal 1: Stop previous server (Ctrl+C), start Llama-3B
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

**Note:** First run downloads model (~6GB)

### Test

```bash
# Terminal 2
./scripts/run_local_exp1.sh llama-3b-rtx3090 5
```

### Validate

```bash
# Compare responses with Gemma-2B
python3 -c "
import json

# Load both results
with open('outputs/results/study1_exp1_gemma-2b-rtx3090_*.json') as f:
    gemma = json.load(f)

with open('outputs/results/study1_exp1_llama-3b-rtx3090_*.json') as f:
    llama = json.load(f)

print('Gemma-2B response:', gemma[0]['response'][:100])
print('Llama-3B response:', llama[0]['response'][:100])
print('✅ Both models generating different but valid responses')
"
```

### Success Criteria
- ✅ Llama loads successfully
- ✅ Responses differ from Gemma (different model should give different answers)
- ✅ Response quality is reasonable
- ✅ JSON structure identical to Gemma output

---

## Model 3: Gemma-2 9B

**Purpose:** Test medium-sized model, higher quality responses

**Specs:**
- Size: ~18GB
- GPUs: 1x
- VRAM: ~20GB used
- Speed: ~5-10 sec/query

### Setup

```bash
# Terminal 1: Start Gemma-9B
vllm serve google/gemma-2-9b-it \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95
```

**Note:** This is close to 24GB limit! Monitor GPU usage.

### Test

```bash
# Terminal 2
./scripts/run_local_exp1.sh gemma-9b-rtx3090 5
```

### Validate Quality Improvement

```bash
# Compare 2B vs 9B responses
python3 -c "
import json

with open('outputs/results/study1_exp1_gemma-2b-rtx3090_*.json') as f:
    gemma2b = json.load(f)

with open('outputs/results/study1_exp1_gemma-9b-rtx3090_*.json') as f:
    gemma9b = json.load(f)

print('=== Trial 1 Responses ===')
print('Gemma-2B:', gemma2b[0]['response'])
print()
print('Gemma-9B:', gemma9b[0]['response'])
print()
print('Expect: 9B should be more detailed/nuanced')
"
```

### Success Criteria
- ✅ Fits in 24GB VRAM
- ✅ Responses are noticeably better quality than 2B
- ✅ No CUDA OOM errors
- ✅ JSON output matches format

---

## Model 4: Llama-3.1 8B

**Purpose:** Medium Llama, matched comparison with Gemma-9B

**Specs:**
- Size: ~16GB
- GPUs: 1x
- VRAM: ~19GB used
- Speed: ~4-8 sec/query

### Setup

```bash
# Terminal 1: Start Llama-8B
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### Test

```bash
# Terminal 2
./scripts/run_local_exp1.sh llama-8b-rtx3090 5
```

### Cross-Family Comparison

```bash
# Compare Gemma-9B vs Llama-8B (similar sizes, different families)
python3 -c "
import json

with open('outputs/results/study1_exp1_gemma-9b-rtx3090_*.json') as f:
    gemma = json.load(f)

with open('outputs/results/study1_exp1_llama-8b-rtx3090_*.json') as f:
    llama = json.load(f)

print('=== Cross-Family Comparison (Same Trial) ===')
print('Gemma-9B:', gemma[0]['response'])
print()
print('Llama-8B:', llama[0]['response'])
print()
print('Different training, different responses, but both should be reasonable')
"
```

### Success Criteria
- ✅ Llama-8B loads successfully
- ✅ Responses differ from Gemma-9B but similar quality
- ✅ No errors
- ✅ Validates cross-family experimental design

---

## Model 5: Gemma-2 27B ⚠️ K8S ONLY

**Purpose:** Large Gemma, 2-GPU model

**Why K8s Only:**
- Requires 2 GPUs with tensor parallelism
- ~54GB model → ~27GB with AWQ quantization
- Needs 2x RTX 3090 (48GB total VRAM)

**Local Testing:** Skip unless you have multi-GPU setup

**K8s Testing:**
```bash
# Deploy to cluster
kubectl apply -f kubernetes/generated/deployment-gemma-27b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-27b-rtx3090.yaml

# Wait for pod to be Ready
kubectl get pods -n lemn-lab -l model=gemma-27b-rtx3090 -w

# Port-forward to test locally
kubectl port-forward -n lemn-lab svc/vllm-gemma-27b 8000:8000

# Then run local script against forwarded endpoint
./scripts/run_local_exp1.sh gemma-27b-rtx3090 5
```

---

## Model 6: Llama-3.1 70B ⚠️ K8S ONLY

**Purpose:** Largest standard model, 4-GPU requirement

**Why K8s Only:**
- Requires 4 GPUs
- ~140GB model → ~35GB with AWQ quantization
- Needs 4x RTX 3090 (96GB total VRAM)

**K8s Testing:** Same process as Gemma-27B

---

## Model 7: DeepSeek-R1 70B ⚠️ K8S ONLY + REASONING

**Purpose:** Reasoning model, extracts <think>...</think> traces

**Why K8s Only:**
- Same 4-GPU requirements as Llama-70B
- Plus: Special reasoning extraction logic

**Special Considerations:**
- Outputs reasoning traces in separate fields
- Longer generation time (reasoning + answer)
- Test reasoning extraction works correctly

**K8s Testing:**
```bash
# Deploy DeepSeek-R1
kubectl apply -f kubernetes/generated/deployment-deepseek-r1-70b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-deepseek-r1-70b-rtx3090.yaml

# Run with reasoning extraction
kubectl apply -f kubernetes/generated/job-study1-exp1-deepseek-r1-70b-rtx3090.yaml

# Check logs for reasoning traces
kubectl logs -n lemn-lab job/grace-study1-exp1-deepseek-r1-70b-rtx3090 -f
```

**Validate Reasoning Output:**
```bash
# Should have reasoning_trace field in JSONL file
# Check that <think>...</think> was extracted correctly
```

---

## Testing Checklist

### Tier 1: Local Single-GPU Models

- [ ] **Gemma-2 2B** - Baseline test
  - [ ] vLLM starts
  - [ ] Study 1 Exp 1 runs
  - [ ] Study 2 Exp 1 runs
  - [ ] JSON output valid
  - [ ] Responses look good

- [ ] **Llama-3.2 3B** - Cross-family validation
  - [ ] vLLM starts
  - [ ] Experiments run
  - [ ] JSON valid
  - [ ] Responses differ from Gemma-2B

- [ ] **Gemma-2 9B** - Quality improvement
  - [ ] Fits in VRAM
  - [ ] Experiments run
  - [ ] Responses better than 2B

- [ ] **Llama-3.1 8B** - Matched comparison
  - [ ] vLLM starts
  - [ ] Experiments run
  - [ ] Comparable to Gemma-9B

### Tier 2: K8s Multi-GPU Models

- [ ] **Gemma-2 27B** (2 GPUs)
  - [ ] Deployment successful
  - [ ] vLLM pod Running
  - [ ] Port-forward works
  - [ ] Jobs complete

- [ ] **Llama-3.1 70B** (4 GPUs)
  - [ ] 4-GPU exception approved
  - [ ] Deployment successful
  - [ ] Jobs complete

- [ ] **DeepSeek-R1 70B** (4 GPUs + reasoning)
  - [ ] Deployment successful
  - [ ] Reasoning extraction works
  - [ ] Jobs complete

---

## Testing Script

```bash
#!/bin/bash
# Test all single-GPU models sequentially

MODELS=("google/gemma-2-2b-it" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-9b-it" "meta-llama/Llama-3.1-8B-Instruct")
MODEL_NAMES=("gemma-2b-rtx3090" "llama-3b-rtx3090" "gemma-9b-rtx3090" "llama-8b-rtx3090")

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  NAME="${MODEL_NAMES[$i]}"

  echo "========================================"
  echo "Testing Model: $NAME"
  echo "========================================"

  # Start vLLM
  vllm serve "$MODEL" --port 8000 --dtype bfloat16 --max-model-len 4096 &
  VLLM_PID=$!

  # Wait for startup
  sleep 60

  # Test
  ./scripts/run_local_exp1.sh "$NAME" 5

  # Stop vLLM
  kill $VLLM_PID

  echo ""
  echo "✅ $NAME complete"
  echo ""
  sleep 10
done
```

---

## Next Steps After Local Testing

Once Tier 1 models (1-GPU) are validated:

1. **Fix any issues** found during local testing
2. **Build Docker image** with validated scripts
3. **Deploy to K8s** for Tier 2 models (2-4 GPUs)
4. **Run full experiments** (300 + 2,424 trials)

---

## Quick Reference

**Start vLLM:**
```bash
vllm serve <model-name> --port 8000 --dtype bfloat16 --max-model-len 4096
```

**Test model:**
```bash
./scripts/run_local_exp1.sh <model-key> 5
```

**Check output:**
```bash
ls -lh outputs/results/
cat outputs/results/study1_exp1_<model>_*.json | head -50
```

**Compare models:**
```bash
python3 -c "
import json
with open('outputs/results/study1_exp1_gemma-2b-rtx3090_*.json') as f:
    data = json.load(f)
    print(data[0]['response'])
"
```
