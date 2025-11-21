# Replication Implementation Summary

**Date**: 2025-01-21
**Status**: ✅ Implementation Complete - Ready for Testing

---

## Overview

Successfully implemented a complete multi-replication system for Grace Project experiments. This system enables running multiple independent replications of experiments with proper randomization, seed tracking, and parallel execution using Kubernetes Indexed Jobs.

## Key Features

✅ **Experimental Validity**: Independent draws, counterbalanced trial orders, within-model replication
✅ **Reproducibility**: Full seed tracking, metadata logging, deterministic randomization
✅ **Parallel Execution**: Kubernetes Indexed Jobs with per-model parallelism constraints
✅ **Organized Output**: Structured folder hierarchy (`/results/<model>/replication-<N>/`)
✅ **Safety**: Per-model parallelism prevents GPU contention deadlocks

---

## Implementation Components

### 1. Core Utilities

**File**: `src/utils/replication.py` (NEW)

Functions:
- `add_replication_args()` - Standard CLI arguments for all scripts
- `initialize_replication()` - Seed initialization and metadata context
- `shuffle_trials()` - Deterministic trial randomization with pandas
- `add_replication_metadata()` - Add replication tracking to results

Seed offsets by experiment:
- `study1_exp1`: base_seed + 0
- `study2_exp1`: base_seed + 1000
- `study1_exp2`: base_seed + 2000
- `study2_exp2`: base_seed + 3000

### 2. Modified Experiment Scripts

All 4 scripts updated with full replication support:

**Study 1 Experiment 1** (`src/query_study1_exp1.py`)
- ✅ Replication imports added
- ✅ Accepts `--seed`, `--shuffle`, `--replication-id`
- ✅ Passes seed to vLLM API calls
- ✅ Shuffles trials if requested
- ✅ Adds metadata to results

**Study 1 Experiment 2** (`src/query_study1_exp2.py`)
- ✅ Same pattern as Exp1
- ✅ Uses seed offset 2000

**Study 2 Experiment 1** (`src/query_study2_exp1.py`)
- ✅ Same pattern as Study 1
- ✅ Uses seed offset 1000

**Study 2 Experiment 2** (`src/query_study2_exp2.py`)
- ✅ Same pattern as Study 1
- ✅ Uses seed offset 3000

### 3. API Client Updates

**File**: `src/utils/api_client.py`

Modified methods to accept optional `seed` parameter:
- `_make_request()` - Passes seed to vLLM API payload
- `generate_text()` - Text generation with seed
- `extract_token_probabilities()` - Probability extraction with seed
- `extract_binary_probabilities()` - Binary probabilities with seed

### 4. Model Configuration

**File**: `config/models.yaml`

Added `replication.parallelism` field to all 7 models:

**1-GPU models** (parallelism=3):
- `gemma-2b-rtx3090`
- `llama-3b-rtx3090`
- `gemma-9b-rtx3090`
- `llama-8b-rtx3090`

**4-GPU model** (parallelism=1):
- `gemma-27b-rtx3090`

**8-GPU models** (parallelism=1):
- `llama-70b-rtx3090`
- `deepseek-r1-70b-rtx3090`

Rationale:
- Small models (1 GPU): Can run 3 replications in parallel on different nodes
- Large models (4-8 GPUs): MUST run sequentially to avoid deadlocks

### 5. Kubernetes Job Template

**File**: `templates/replicated-job.yaml.j2` (NEW)

Jinja2 template for generating Kubernetes Indexed Jobs:
- Uses `completionMode: Indexed`
- Reads `parallelism` from model config
- Creates per-replication output folders
- Handles both Exp1 (vLLM API) and Exp2 (direct scoring)
- Automatically configures resources based on experiment type
- Supports reasoning models (DeepSeek-R1)

Template variables:
- `model`: Full model config from models.yaml
- `model_key`: Model identifier (e.g., "gemma-2b-rtx3090")
- `experiment`: Experiment name (e.g., "study1_exp1")
- `num_replications`: Number of replications
- `base_seed`: Base random seed
- `shuffle`: Boolean for trial randomization
- `image`: Docker image tag
- `namespace`: Kubernetes namespace
- `pvcs`: PVC names
- `input_data`: Input CSV path

### 6. Manifest Generation Script

**File**: `scripts/generate_replicated_jobs.py` (NEW)

Python script to generate Kubernetes manifests from template:

Features:
- Loads `models.yaml` and `experiments.yaml`
- Renders Jinja2 template with appropriate values
- Supports single or batch generation (`--all-models`, `--all-experiments`)
- Auto-generates seed if not specified
- Validates model and experiment configurations
- Outputs to `kubernetes/generated/`

Usage examples:
```bash
# Single model/experiment
python scripts/generate_replicated_jobs.py \
    --model gemma-2b-rtx3090 \
    --experiment study1_exp1 \
    --replications 5 \
    --base-seed 42 \
    --shuffle

# All models, single experiment
python scripts/generate_replicated_jobs.py \
    --all-models \
    --experiment study1_exp2 \
    --replications 10

# All models and experiments
python scripts/generate_replicated_jobs.py \
    --all-models \
    --all-experiments \
    --replications 10
```

### 7. Orchestration Script

**File**: `scripts/run_replicated_experiments.sh` (NEW)

Bash script for end-to-end execution:

Workflow:
1. **Generate manifests** - Using generate_replicated_jobs.py
2. **Deploy vLLM server** - If needed for Exp1 experiments
3. **Deploy replicated jobs** - Apply generated manifests
4. **Monitor progress** - Wait for job completion
5. **Cleanup** - Delete completed jobs and optionally vLLM deployment

Features:
- Colored output for readability
- Dry-run mode (`--dry-run`)
- Optional waiting (`--no-wait`)
- Optional cleanup (`--no-cleanup`)
- Skip vLLM deployment (`--no-vllm`)

Usage examples:
```bash
# Run single experiment with replications
./scripts/run_replicated_experiments.sh \
    --model gemma-2b-rtx3090 \
    --experiment study1_exp1 \
    --replications 5 \
    --shuffle

# Run all experiments for a model
./scripts/run_replicated_experiments.sh \
    --model llama-70b-rtx3090 \
    --all-experiments \
    --replications 10 \
    --base-seed 42

# Dry run (generate manifests only)
./scripts/run_replicated_experiments.sh \
    --model gemma-9b-rtx3090 \
    --experiment study1_exp2 \
    --replications 3 \
    --dry-run
```

---

## Output Structure

Results are organized hierarchically:

```
/results/
  <model>/
    replication-0/
      study1_exp1_results.json
      study1_exp1_reasoning.jsonl
      study1_exp2_results.json
      study2_exp1_results.json
      study2_exp2_results.json
    replication-1/
      ...
    replication-N/
      ...
```

Each result file includes replication metadata:
- `replication_id`: 0-indexed replication number
- `replication_seed`: Actual seed used (base_seed + offset + replication_id)
- `trial_order_shuffled`: Whether trials were randomized
- `trial_order_in_replication`: Position in shuffled order
- `run_timestamp`: ISO timestamp of execution
- `model_name`: Model identifier
- `experiment_name`: Experiment identifier

---

## Seed Management

### Seed Calculation

For each replication:
```
actual_seed = base_seed + seed_offset + replication_id
```

Where:
- `base_seed`: User-specified or auto-generated (timestamp-based)
- `seed_offset`: Experiment-specific (0, 1000, 2000, 3000)
- `replication_id`: Replication index (0, 1, 2, ...)

### Example

Running study1_exp2 with base_seed=1000, 3 replications:
- Replication 0: seed = 1000 + 2000 + 0 = 3000
- Replication 1: seed = 1000 + 2000 + 1 = 3001
- Replication 2: seed = 1000 + 2000 + 2 = 3002

This ensures:
- ✅ Each replication gets unique seed
- ✅ Different experiments get different seeds
- ✅ Seeds are deterministic and reproducible
- ✅ Random number generators (Python, NumPy) are seeded
- ✅ vLLM API receives seed for reproducible sampling

---

## Parallelism Management

### Per-Model Parallelism

**Small models (1 GPU)**: `parallelism: 3`
- Can run 3 replications simultaneously
- Each replication uses 1 GPU on different node
- No contention, maximum throughput

**Large models (4-8 GPUs)**: `parallelism: 1`
- Only 1 replication at a time
- Prevents deadlocks when model needs all GPUs on a node
- Sequential execution is safer

### Example: Gemma 2B (1 GPU, parallelism=3)

With 5 replications:
- Replications 0, 1, 2 start immediately (3 pods running)
- Replication 3 starts when any of 0-2 completes
- Replication 4 starts when another completes
- Total time ≈ ceiling(5/3) × time_per_replication

### Example: Llama 70B (8 GPUs, parallelism=1)

With 5 replications:
- Replication 0 starts immediately (8 GPUs allocated)
- Replication 1 waits until 0 completes
- Replication 2 waits until 1 completes
- ...
- Total time = 5 × time_per_replication

---

## Resource Requirements

### Experiment 1 (vLLM API) - CPU-Only Jobs

- **CPU**: 4 cores
- **Memory**: 8Gi
- **GPU**: None (vLLM deployment has GPUs)
- **Storage**: Minimal (just output files)

### Experiment 2 (Direct Scoring) - GPU Jobs

Resources match model requirements:
- **1-GPU models**: 1 GPU, 20Gi memory, 8 CPU
- **4-GPU models**: 4 GPUs, 60Gi memory, 16 CPU
- **8-GPU models**: 8 GPUs, 120Gi memory, 48 CPU

All limits within 120% of requests (NRP compliant ✅)

---

## Testing Plan

### Phase 1: Local Testing
- ✅ All scripts pass syntax checks
- ⏳ Test replication utilities with mock data
- ⏳ Verify seed propagation through pipeline

### Phase 2: Small-Scale Cluster Test
- ⏳ Deploy Gemma 2B on RTX 3090
- ⏳ Run study1_exp1 with 2 replications
- ⏳ Verify:
  - Different seeds per replication
  - Organized output folders
  - Correct metadata in results
  - Parallel execution works

### Phase 3: Large-Scale Test
- ⏳ Test Llama 70B (8 GPUs, parallelism=1)
- ⏳ Run 3 replications sequentially
- ⏳ Verify no deadlocks with 8-GPU model
- ⏳ Confirm sequential execution

### Phase 4: Full Production
- ⏳ Run all models × all experiments × 10 replications
- ⏳ Monitor resource usage and timing
- ⏳ Validate statistical properties of results

---

## Next Steps

1. **Install Dependencies**
   ```bash
   pip install jinja2 pyyaml
   ```

2. **Test Generation Script**
   ```bash
   # Dry run to test manifest generation
   python scripts/generate_replicated_jobs.py \
       --model gemma-2b-rtx3090 \
       --experiment study1_exp1 \
       --replications 2 \
       --dry-run
   ```

3. **Run Small Test**
   ```bash
   # Deploy and run 2 replications on Gemma 2B
   ./scripts/run_replicated_experiments.sh \
       --model gemma-2b-rtx3090 \
       --experiment study1_exp1 \
       --replications 2 \
       --shuffle
   ```

4. **Verify Results**
   ```bash
   # Check output structure
   kubectl exec -n lemn-lab <results-pod> -- ls -la /results/gemma-2b-rtx3090/

   # Inspect result files
   kubectl exec -n lemn-lab <results-pod> -- cat \
       /results/gemma-2b-rtx3090/replication-0/study1_exp1_results.json
   ```

5. **Scale to Production**
   - Run all models × all experiments × 10 replications
   - Analyze cross-replication consistency
   - Generate publication-ready datasets

---

## Files Modified/Created

### Created Files (9 total)

1. `src/utils/replication.py` - Core replication utilities
2. `templates/replicated-job.yaml.j2` - Kubernetes job template
3. `scripts/generate_replicated_jobs.py` - Manifest generator
4. `scripts/run_replicated_experiments.sh` - Orchestration script
5. `docs/REPLICATION_IMPLEMENTATION.md` - Full implementation guide
6. `docs/REPLICATION_IMPLEMENTATION_SUMMARY.md` - This file
7. `kubernetes/pvcs-large-models.yaml` - 400GB PVCs for Llama/DeepSeek
8. `kubernetes/job-cache-warmup-llama-70b.yaml` - Llama cache warmup
9. `kubernetes/job-cache-warmup-deepseek-r1-70b.yaml` - DeepSeek cache warmup

### Modified Files (6 total)

1. `src/query_study1_exp1.py` - Added replication support
2. `src/query_study1_exp2.py` - Added replication support
3. `src/query_study2_exp1.py` - Added replication support
4. `src/query_study2_exp2.py` - Added replication support
5. `src/utils/api_client.py` - Added seed parameter to API calls
6. `config/models.yaml` - Added `replication.parallelism` to all models

---

## Benefits

### Scientific Validity
- ✅ Independent replications for statistical power
- ✅ Counterbalanced trial orders reduce order effects
- ✅ Within-model replication enables consistency analysis
- ✅ Full reproducibility via seed tracking

### Operational Efficiency
- ✅ Parallel execution (where safe) reduces wall-clock time
- ✅ Sequential execution (where needed) prevents deadlocks
- ✅ Automated orchestration reduces manual work
- ✅ Organized output simplifies analysis

### NRP Compliance
- ✅ All resource limits within 120% of requests
- ✅ Uses Kubernetes Jobs (not Deployments) for batch work
- ✅ Indexed Jobs provide built-in parallelism control
- ✅ Per-model parallelism prevents oversubscription

---

## Success Metrics

**Implementation**: ✅ Complete

**Testing**: ⏳ Pending
- [ ] Generate manifests for all model/experiment combinations
- [ ] Deploy small test (Gemma 2B, 2 replications)
- [ ] Verify output structure and metadata
- [ ] Test large model (Llama 70B) with parallelism=1
- [ ] Run full production test (all models, 10 replications)

**Analysis**: ⏳ Pending
- [ ] Verify cross-replication consistency
- [ ] Analyze within-model variance
- [ ] Confirm statistical validity of design

---

## Known Limitations

1. **vLLM API Seed Handling**: vLLM may not fully honor seeds for all models. Test thoroughly.

2. **Node Availability**: Large models (8 GPUs) require nodes with 8 available GPUs. RTX 3090 cluster has limited such nodes.

3. **Storage**: 10 replications × 7 models × 4 experiments = 280 result files. Ensure sufficient PVC space.

4. **Runtime**: Sequential execution for large models means 10 replications × 4 experiments × ~2 hours = ~80 hours per large model.

5. **Monitoring**: Currently manual. Could add automated monitoring/alerting.

---

## Support

For issues or questions:
1. Check logs: `kubectl logs -n lemn-lab -l job-name=<job-name>`
2. Inspect job status: `kubectl describe job -n lemn-lab <job-name>`
3. Review generated manifests in `kubernetes/generated/`
4. Consult implementation guide: `docs/REPLICATION_IMPLEMENTATION.md`

---

**Status**: ✅ Ready for testing
**Next**: Run small-scale test with Gemma 2B
