# Kubernetes Manifest Generation Guide

**Config-driven system for generating Kubernetes manifests**

Instead of manually maintaining 24+ YAML files, this system uses centralized configuration files to generate all deployments, services, and jobs.

---

## Overview

**Single source of truth**: Edit `config/models.yaml` and `config/experiments.yaml`

**Generate all manifests**: Run `python3 scripts/generate_manifests.py`

**Output**: 24 Kubernetes manifest files in `kubernetes/generated/`

---

## Architecture

```
config/
├── models.yaml           # Model definitions (4 models)
└── experiments.yaml      # Experiment definitions (4 experiments)

scripts/
└── generate_manifests.py # Generator script

kubernetes/
└── generated/            # Auto-generated (gitignored)
    ├── deployment-gemma-2b-rtx3090.yaml
    ├── service-gemma-2b-rtx3090.yaml
    ├── job-study1-exp1-gemma-2b-rtx3090.yaml
    └── ... (21 more files)
```

---

## Quick Start

### 1. Generate All Manifests

```bash
# Activate virtual environment
source venv-grace/bin/activate

# Generate all manifests (4 models × 4 experiments = 24 files)
python3 scripts/generate_manifests.py
```

**Output:**
```
✅ Generated 24 files in kubernetes/generated/
   - 8 deployment/service files
   - 16 job files

⚠️  1 Warning: llama-70b-rtx3090 requires 4 GPUs (exception needed)
```

### 2. Deploy a Model

```bash
# Deploy Gemma-2B on RTX 3090
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml

# Check status
kubectl get pods -n lemn-lab -l model=gemma-2b-rtx3090 -w
```

### 3. Run an Experiment

```bash
# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l model=gemma-2b-rtx3090 -n lemn-lab

# Run Study 1 Experiment 1
kubectl apply -f kubernetes/generated/job-study1-exp1-gemma-2b-rtx3090.yaml

# Monitor job
kubectl get jobs -n lemn-lab -w
kubectl logs -f job/grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
```

---

## Configuration Files

### `config/models.yaml`

Defines models and their resource requirements.

**Structure:**
```yaml
models:
  gemma-2b-rtx3090:
    display_name: "Gemma 2B"
    huggingface_name: google/gemma-2-2b-it

    gpu:
      count: 1
      type: RTX-3090
      node_selector: NVIDIA-GeForce-RTX-3090

    vllm:
      dtype: bfloat16
      quantization: null  # or "awq" for 4-bit
      max_model_len: 4096
      tensor_parallel_size: 1

    resources:
      memory_request: 16Gi
      memory_limit: 19Gi    # 118.75% (NRP compliant)
      cpu_request: "8"
      cpu_limit: "9600m"    # 120% (NRP compliant)
      shm_size: 8Gi

    deployment:
      name: vllm-gemma-2b-rtx3090
      service_name: vllm-gemma-2b
```

**Current models:**
- `gemma-2b-rtx3090`: 1 GPU, no quantization
- `gemma-9b-rtx3090`: 1 GPU, no quantization
- `gemma-27b-rtx3090`: 2 GPUs, AWQ quantization
- `llama-70b-rtx3090`: 4 GPUs, AWQ quantization (⚠️ requires NRP exception)

### `config/experiments.yaml`

Defines experiments and their parameters.

**Structure:**
```yaml
experiments:
  study1-exp1:
    name: "Study 1 Experiment 1"
    description: "Knowledge attribution - raw text responses"
    script: query_study1_exp1.py
    input_file: /app/data/study1.csv
    output_format: json

    study_number: 1
    experiment_number: 1

    parameters:
      temperature: 0.7
      max_tokens: 512
      include_reasoning_trace: true

    resources:
      memory_request: 4Gi
      memory_limit: 4800Mi  # 120% (compliant)
      cpu_request: "2"
      cpu_limit: "2400m"    # 120% (compliant)
```

**Current experiments:**
- `study1-exp1`: Knowledge attribution - raw text
- `study1-exp2`: Knowledge attribution - probabilities
- `study2-exp1`: Politeness - raw text
- `study2-exp2`: Politeness - probabilities

---

## Generator Script Usage

### Generate Everything

```bash
python3 scripts/generate_manifests.py
```

### Generate for Specific Model

```bash
# Only generate for Gemma-2B
python3 scripts/generate_manifests.py --model gemma-2b-rtx3090
```

### Generate Deployments Only

```bash
# Skip job generation
python3 scripts/generate_manifests.py --deployments-only
```

### Generate Jobs Only

```bash
# Skip deployment/service generation
python3 scripts/generate_manifests.py --jobs-only
```

### Custom Directories

```bash
# Use different config/output directories
python3 scripts/generate_manifests.py \
    --config-dir my-configs/ \
    --output-dir my-output/
```

---

## Adding a New Model

1. **Edit `config/models.yaml`**:

```yaml
models:
  my-new-model:
    display_name: "My New Model"
    huggingface_name: org/model-name
    gpu:
      count: 1
      type: RTX-3090
      node_selector: NVIDIA-GeForce-RTX-3090
    vllm:
      dtype: bfloat16
      quantization: null
      max_model_len: 4096
      tensor_parallel_size: 1
    resources:
      memory_request: 16Gi
      memory_limit: 19Gi
      cpu_request: "8"
      cpu_limit: "9600m"
      shm_size: 8Gi
    deployment:
      name: vllm-my-new-model
      service_name: vllm-my-model
```

2. **Regenerate manifests**:

```bash
python3 scripts/generate_manifests.py
```

3. **Done!** All 4 experiments automatically created for new model.

---

## Adding a New Experiment

1. **Edit `config/experiments.yaml`**:

```yaml
experiments:
  my-new-exp:
    name: "My New Experiment"
    description: "Description here"
    script: query_my_exp.py
    input_file: /app/data/my_data.csv
    output_format: json
    study_number: 3
    experiment_number: 1
    parameters:
      temperature: 0.7
      max_tokens: 512
    resources:
      memory_request: 4Gi
      memory_limit: 4800Mi
      cpu_request: "2"
      cpu_limit: "2400m"
```

2. **Regenerate manifests**:

```bash
python3 scripts/generate_manifests.py
```

3. **Done!** Jobs created for all 4 models automatically.

---

## NRP Compliance Validation

The generator automatically validates:

✅ **Memory limits ≤ 120% of requests**
✅ **CPU limits ≤ 120% of requests**
⚠️ **GPU count > 2** (warns, but allows - for exception requests)

Example output:
```
Validating NRP compliance...

⚠️  1 Warning(s):
  - llama-70b-rtx3090: Requires 4 GPUs
    (exceeds NRP default limit of 2 - exception required)
```

---

## Multi-GPU Configuration

### Tensor Parallelism

For models that need multiple GPUs:

```yaml
gpu:
  count: 2  # Split model across 2 GPUs

vllm:
  tensor_parallel_size: 2  # Must match GPU count
  quantization: awq        # Often needed to fit in VRAM
```

**Generated vLLM args:**
```yaml
args:
  - --model=google/gemma-2-27b-it
  - --tensor-parallel-size=2
  - --quantization=awq
```

**Resource allocation:**
```yaml
resources:
  requests:
    nvidia.com/gpu: 2  # Requests 2 GPUs
```

### GPU Limits

| Model | GPUs | Status |
|-------|------|--------|
| Gemma-2B | 1 | ✅ Within default limit |
| Gemma-9B | 1 | ✅ Within default limit |
| Gemma-27B | 2 | ✅ Within default limit |
| Llama-70B | 4 | ⚠️ Requires exception |

---

## Quantization

### AWQ 4-bit Quantization

Reduces memory by ~4x with minimal quality loss (<3%).

**When to use:**
- Model doesn't fit in available VRAM
- Example: Gemma-27B (54GB) → AWQ (27GB) fits in 2×24GB RTX 3090s

**Configuration:**
```yaml
vllm:
  quantization: awq
```

**Generated arg:**
```yaml
args:
  - --quantization=awq
```

### No Quantization

For smaller models that fit comfortably:

**Configuration:**
```yaml
vllm:
  quantization: null
```

---

## Output Files

### Naming Convention

**Deployments:**
- `deployment-{model-key}.yaml`
- Example: `deployment-gemma-2b-rtx3090.yaml`

**Services:**
- `service-{model-key}.yaml`
- Example: `service-gemma-2b-rtx3090.yaml`

**Jobs:**
- `job-{experiment-key}-{model-key}.yaml`
- Example: `job-study1-exp1-gemma-2b-rtx3090.yaml`

### Auto-Generated Headers

All files include generation metadata:

```yaml
---
# Auto-generated Deployment: Gemma 2B
# Generated: 2025-11-19T18:16:12.215200
# Config: gemma-2b-rtx3090
```

---

## Workflow Examples

### Sequential Deployment (Recommended)

Run one model at a time to avoid resource conflicts:

```bash
# 1. Deploy Gemma-2B
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml

# 2. Run all 4 experiments for Gemma-2B
kubectl apply -f kubernetes/generated/job-study1-exp1-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/job-study1-exp2-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/job-study2-exp1-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/job-study2-exp2-gemma-2b-rtx3090.yaml

# 3. Wait for all jobs to complete
kubectl wait --for=condition=complete job -l model=gemma-2b-rtx3090 -n lemn-lab

# 4. Download results
# (see download guide)

# 5. Clean up
kubectl delete deployment vllm-gemma-2b-rtx3090 -n lemn-lab
kubectl delete jobs -l model=gemma-2b-rtx3090 -n lemn-lab

# 6. Repeat for next model (Gemma-9B)
```

### Test with Single Experiment

```bash
# Quick test: Deploy + run one experiment
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l model=gemma-2b-rtx3090 -n lemn-lab

# Run test
kubectl apply -f kubernetes/generated/job-study1-exp1-gemma-2b-rtx3090.yaml

# Monitor
kubectl logs -f job/grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
```

---

## Troubleshooting

### "Model doesn't fit in VRAM"

**Solution 1**: Enable quantization

```yaml
vllm:
  quantization: awq  # Add this
```

**Solution 2**: Increase GPU count

```yaml
gpu:
  count: 2  # Or 4 for very large models

vllm:
  tensor_parallel_size: 2  # Must match
```

### "Resource limits exceed 120%"

**Fix**: Adjust limits in `config/models.yaml`

```yaml
resources:
  memory_request: 16Gi
  memory_limit: 19Gi  # Must be ≤ 19.2Gi (120%)
```

### "Job fails immediately"

**Check:**
1. vLLM deployment is running
2. Service name matches in job
3. Input file path is correct
4. Docker image built and pushed

**Debug:**
```bash
kubectl logs job/grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
kubectl describe job grace-study1-exp1-gemma-2b-rtx3090 -n lemn-lab
```

---

## Best Practices

### 1. Always Regenerate After Config Changes

```bash
# Edit config
vim config/models.yaml

# Regenerate
python3 scripts/generate_manifests.py

# Deploy updated manifests
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
```

### 2. Version Control Configs, Not Generated Files

**Commit:**
- `config/models.yaml` ✅
- `config/experiments.yaml` ✅

**Don't commit:**
- `kubernetes/generated/*.yaml` ❌ (gitignored)

### 3. Validate Before Deploying

```bash
# Generate
python3 scripts/generate_manifests.py

# Check warnings
# ⚠️  Review warning(s) before deploying

# Dry-run validation
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml --dry-run=client
```

### 4. Clean Up Between Models

```bash
# Free GPUs for next model
kubectl delete deployment vllm-gemma-2b-rtx3090 -n lemn-lab
kubectl delete jobs -l model=gemma-2b-rtx3090 -n lemn-lab
```

---

## Migration from Manual YAML

Old manual files in `kubernetes/` are still there for reference:
- `vllm-deployment.yaml` (A100 config)
- `vllm-deployment-rtx3090.yaml`
- `job-exp1-template.yaml`
- `job-exp2-template.yaml`

**New system uses:**
- `kubernetes/generated/` (auto-generated)

**No migration needed** - just start using the generator for new deployments.

---

## Summary

**Before (Manual YAML):**
- Edit 24+ YAML files by hand
- Easy to make mistakes
- Inconsistencies across files

**After (Config-driven):**
- Edit 2 config files
- Generate 24 manifests automatically
- Consistent, validated, compliant

**To get started:**
```bash
source venv-grace/bin/activate
python3 scripts/generate_manifests.py
kubectl apply -f kubernetes/generated/deployment-gemma-2b-rtx3090.yaml
kubectl apply -f kubernetes/generated/service-gemma-2b-rtx3090.yaml
```

---

*Last Updated: 2025-11-19*
*See also: `config/models.yaml`, `config/experiments.yaml`, `scripts/generate_manifests.py`*
