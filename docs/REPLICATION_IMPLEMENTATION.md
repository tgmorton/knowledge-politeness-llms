# Multi-Replication Implementation Plan

## Overview

Implement parallel experiment replications using Kubernetes Job indexed parallelism, where each pod runs one independent replication with:
- Randomized trial order (different seed per replication)
- Unique output folders per model/study/experiment/replication
- Full reproducibility tracking (seeds, timestamps, deployment info)

## Architecture

### Current Pattern (Single Run)
```
Job "grace-gemma-27b-all-experiments"
└── Pod (single)
    ├── Start vLLM
    ├── Run Study1 Exp1 → /results/study1_exp1_gemma-27b.json
    ├── Run Study2 Exp1 → /results/study2_exp1_gemma-27b.json
    ├── Kill vLLM
    ├── Run Study1 Exp2 → /results/study1_exp2_gemma-27b.json
    └── Run Study2 Exp2 → /results/study2_exp2_gemma-27b.json
```

### New Pattern (Multiple Replications)
```
Job "grace-gemma-27b-all-experiments-replicated"
├── spec.completions: 5           # Total replications to run
├── spec.parallelism: 3            # Run 3 at a time
└── spec.completionMode: Indexed   # Provides JOB_COMPLETION_INDEX

Creates 5 pods in parallel (3 at a time):
├── Pod grace-...-0 (INDEX=0, seed=1001)
│   └── /results/gemma-27b/replication-0/study1_exp1.json
├── Pod grace-...-1 (INDEX=1, seed=1002)
│   └── /results/gemma-27b/replication-1/study1_exp1.json
├── Pod grace-...-2 (INDEX=2, seed=1003)
│   └── /results/gemma-27b/replication-2/study1_exp1.json
├── Pod grace-...-3 (INDEX=3, seed=1004) [waits for slot]
└── Pod grace-...-4 (INDEX=4, seed=1005) [waits for slot]
```

## Output Folder Structure

### New Organization
```
/results/
├── gemma-27b/
│   ├── replication-0/
│   │   ├── metadata.json          # Run info: seed, timestamp, deployment
│   │   ├── study1_exp1.json       # 300 trials
│   │   ├── study1_exp2.json       # 1500 queries (300 × 5)
│   │   ├── study2_exp1.json       # 2424 trials
│   │   └── study2_exp2.json
│   ├── replication-1/
│   │   ├── metadata.json
│   │   ├── study1_exp1.json
│   │   └── ...
│   └── replication-2/
│       └── ...
├── llama-70b/
│   ├── replication-0/
│   └── ...
└── deepseek-r1-70b/
    └── ...
```

### Metadata File Format
```json
{
  "model_key": "gemma-27b-rtx3090",
  "model_name": "google/gemma-2-27b-it",
  "replication_index": 0,
  "replication_id": "gemma-27b-rep0-20250121-143022",
  "seeds": {
    "base_seed": 1001,
    "study1_exp1_seed": 1001,
    "study2_exp1_seed": 2001,
    "study1_exp2_seed": 3001,
    "study2_exp2_seed": 4001
  },
  "timestamps": {
    "start": "2025-01-21T14:30:22Z",
    "study1_exp1_complete": "2025-01-21T14:45:10Z",
    "study2_exp1_complete": "2025-01-21T14:50:33Z",
    "vllm_killed": "2025-01-21T14:50:35Z",
    "study1_exp2_complete": "2025-01-21T15:20:18Z",
    "study2_exp2_complete": "2025-01-21T15:55:44Z",
    "end": "2025-01-21T15:55:44Z"
  },
  "deployment_info": {
    "pod_name": "grace-gemma-27b-all-experiments-replicated-0",
    "node_name": "gpu-node-42.nrp.edu",
    "vllm_version": "0.6.4",
    "cuda_version": "12.1",
    "gpu_count": 4,
    "gpu_type": "NVIDIA GeForce RTX 3090"
  },
  "configuration": {
    "temperature_exp1": 0.7,
    "temperature_exp2": 1.0,
    "max_model_len": 4096,
    "tensor_parallel_size": 4
  }
}
```

## Implementation Components

### 1. Kubernetes Job Manifest Template

**File**: `kubernetes/templates/job-replicated-experiments.yaml.j2`

```yaml
---
# Replicated Experiments Job Template
# Variables: MODEL_KEY, MODEL_NAME, HF_MODEL, GPU_COUNT, TP_SIZE, NUM_REPLICATIONS, PARALLELISM

apiVersion: batch/v1
kind: Job
metadata:
  name: grace-{{ MODEL_KEY }}-replicated
  namespace: lemn-lab
  labels:
    app: grace-experiment
    model: {{ MODEL_KEY }}
    type: replicated-all
spec:
  # Indexed Jobs provide JOB_COMPLETION_INDEX to each pod
  completionMode: Indexed
  completions: {{ NUM_REPLICATIONS }}     # Total replications (e.g., 5)
  parallelism: {{ PARALLELISM }}          # Concurrent pods (e.g., 2)
  backoffLimit: 0
  ttlSecondsAfterFinished: 7200           # 2 hours

  template:
    metadata:
      labels:
        app: grace-experiment
        model: {{ MODEL_KEY }}
    spec:
      restartPolicy: Never

      nodeSelector:
        nvidia.com/gpu.product: {{ GPU_NODE_SELECTOR }}

      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
      - name: replication
        image: vllm/vllm-openai:latest
        imagePullPolicy: Always

        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e

            # Get replication index from Kubernetes (0-indexed)
            REPLICATION_INDEX=${JOB_COMPLETION_INDEX}

            # Generate base seed from index (ensures different seeds per replication)
            BASE_SEED=$((1000 + REPLICATION_INDEX))

            # Create output directory for this replication
            MODEL_DIR="/results/{{ MODEL_KEY }}"
            REP_DIR="${MODEL_DIR}/replication-${REPLICATION_INDEX}"
            mkdir -p "${REP_DIR}"

            echo "===== REPLICATION ${REPLICATION_INDEX} - {{ MODEL_KEY }} ====="
            echo "Base seed: ${BASE_SEED}"
            echo "Output dir: ${REP_DIR}"

            # Capture start time
            START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            # Start vLLM server
            echo "[1/7] Starting vLLM server..."
            python -m vllm.entrypoints.openai.api_server \
              --model {{ HF_MODEL }} \
              --dtype bfloat16 \
              --max-model-len 4096 \
              --tensor-parallel-size {{ TP_SIZE }} \
              --host 0.0.0.0 \
              --port 8000 &

            VLLM_PID=$!

            # Wait for vLLM
            echo "[2/7] Waiting for vLLM..."
            for i in {1..60}; do
              if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "✓ vLLM ready"
                break
              fi
              sleep 5
            done

            # Install dependencies
            echo "[3/7] Installing dependencies..."
            pip install -q httpx pandas numpy scipy tqdm
            export PYTHONPATH=/code

            # Run Study 1 Exp1
            echo "[4/7] Running Study 1 Exp1 (seed=$((BASE_SEED)))..."
            python3 /code/src/query_study1_exp1.py \
              --input=/code/data/study1.csv \
              --output="${REP_DIR}/study1_exp1.json" \
              --endpoint=http://localhost:8000 \
              --model-name={{ MODEL_KEY }} \
              --seed=${BASE_SEED} \
              --shuffle \
              --replication-id=${REPLICATION_INDEX}

            STUDY1_EXP1_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            # Run Study 2 Exp1 (different seed offset)
            echo "[5/7] Running Study 2 Exp1 (seed=$((BASE_SEED + 1000)))..."
            python3 /code/src/query_study2_exp1.py \
              --input=/code/data/study2.csv \
              --output="${REP_DIR}/study2_exp1.json" \
              --endpoint=http://localhost:8000 \
              --model-name={{ MODEL_KEY }} \
              --seed=$((BASE_SEED + 1000)) \
              --shuffle \
              --replication-id=${REPLICATION_INDEX}

            STUDY2_EXP1_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            # Kill vLLM
            echo "[6/7] Stopping vLLM..."
            kill $VLLM_PID
            wait $VLLM_PID || true
            VLLM_KILLED_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            # Run Exp2 studies
            echo "[7/7] Running Exp2 studies..."
            python3 /code/src/query_study1_exp2.py \
              --input=/code/data/study1.csv \
              --output="${REP_DIR}/study1_exp2.json" \
              --model-path={{ HF_MODEL }} \
              --model-name={{ MODEL_KEY }} \
              --seed=$((BASE_SEED + 2000)) \
              --shuffle \
              --replication-id=${REPLICATION_INDEX}

            STUDY1_EXP2_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            python3 /code/src/query_study2_exp2.py \
              --input=/code/data/study2.csv \
              --output="${REP_DIR}/study2_exp2.json" \
              --model-path={{ HF_MODEL }} \
              --model-name={{ MODEL_KEY }} \
              --seed=$((BASE_SEED + 3000)) \
              --shuffle \
              --replication-id=${REPLICATION_INDEX}

            STUDY2_EXP2_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            # Write metadata file
            cat > "${REP_DIR}/metadata.json" <<EOF
            {
              "model_key": "{{ MODEL_KEY }}",
              "model_name": "{{ HF_MODEL }}",
              "replication_index": ${REPLICATION_INDEX},
              "replication_id": "{{ MODEL_KEY }}-rep${REPLICATION_INDEX}-$(date +%Y%m%d-%H%M%S)",
              "seeds": {
                "base_seed": ${BASE_SEED},
                "study1_exp1_seed": ${BASE_SEED},
                "study2_exp1_seed": $((BASE_SEED + 1000)),
                "study1_exp2_seed": $((BASE_SEED + 2000)),
                "study2_exp2_seed": $((BASE_SEED + 3000))
              },
              "timestamps": {
                "start": "${START_TIME}",
                "study1_exp1_complete": "${STUDY1_EXP1_TIME}",
                "study2_exp1_complete": "${STUDY2_EXP1_TIME}",
                "vllm_killed": "${VLLM_KILLED_TIME}",
                "study1_exp2_complete": "${STUDY1_EXP2_TIME}",
                "study2_exp2_complete": "${STUDY2_EXP2_TIME}",
                "end": "${END_TIME}"
              },
              "deployment_info": {
                "pod_name": "${HOSTNAME}",
                "node_name": "${NODE_NAME}",
                "gpu_count": {{ GPU_COUNT }},
                "tensor_parallel_size": {{ TP_SIZE }}
              },
              "configuration": {
                "temperature_exp1": 0.7,
                "temperature_exp2": 1.0,
                "max_model_len": 4096
              }
            }
            EOF

            echo "===== ✅ REPLICATION ${REPLICATION_INDEX} COMPLETE ====="
            ls -lh "${REP_DIR}/"

        env:
          - name: HF_HOME
            value: /models/.cache
          - name: TRANSFORMERS_CACHE
            value: /models/.cache
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token-thomas
                key: HF_TOKEN
          - name: NODE_NAME
            valueFrom:
              fieldRef:
                fieldPath: spec.nodeName

        resources:
          requests:
            nvidia.com/gpu: {{ GPU_COUNT }}
            memory: {{ MEMORY_REQUEST }}
            cpu: "{{ CPU_REQUEST }}"
          limits:
            nvidia.com/gpu: {{ GPU_COUNT }}
            memory: {{ MEMORY_LIMIT }}
            cpu: "{{ CPU_LIMIT }}"

        volumeMounts:
          - name: code
            mountPath: /code
            readOnly: true
          - name: results
            mountPath: /results
          - name: model-cache
            mountPath: /models

      volumes:
        - name: code
          persistentVolumeClaim:
            claimName: grace-code
        - name: results
          persistentVolumeClaim:
            claimName: thomas-grace-results
        - name: model-cache
          persistentVolumeClaim:
            claimName: {{ MODEL_CACHE_PVC }}
```

### 2. Script Modifications

#### A. Add Common Replication Arguments

**File**: `src/utils/replication.py` (NEW)

```python
"""
Replication support utilities for Grace Project
Handles seeding, trial randomization, and metadata tracking
"""

import argparse
import random
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def add_replication_args(parser: argparse.ArgumentParser):
    """Add standard replication arguments to argument parser"""
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: use current time)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Randomize trial order (recommended for multi-replication studies)'
    )
    parser.add_argument(
        '--replication-id',
        type=int,
        default=None,
        help='Replication index (0-indexed, from JOB_COMPLETION_INDEX)'
    )


def initialize_replication(args, default_seed_offset: int = 0) -> Dict[str, Any]:
    """
    Initialize replication context: set seeds, prepare metadata

    Args:
        args: Parsed command-line arguments
        default_seed_offset: Offset to add to seed (for different experiments)

    Returns:
        Replication context dictionary with seed, metadata, etc.
    """
    # Determine seed
    if args.seed is not None:
        seed = args.seed + default_seed_offset
    else:
        # Generate from timestamp if not provided
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
        seed += default_seed_offset

    # Set global random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Build context
    context = {
        'seed': seed,
        'base_seed': args.seed,
        'seed_offset': default_seed_offset,
        'shuffle': args.shuffle,
        'replication_id': args.replication_id,
        'timestamp': datetime.now().isoformat(),
    }

    return context


def shuffle_trials(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle trial order deterministically

    Args:
        df: DataFrame of trials
        seed: Random seed

    Returns:
        Shuffled DataFrame with reset index
    """
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_replication_metadata(
    results: list,
    context: Dict[str, Any],
    model_name: str,
    experiment_name: str
) -> list:
    """
    Add replication metadata to each result record

    Args:
        results: List of result dictionaries
        context: Replication context from initialize_replication()
        model_name: Model identifier
        experiment_name: Experiment identifier (e.g., "study1_exp1")

    Returns:
        Results with added metadata fields
    """
    for result in results:
        result['replication_id'] = context['replication_id']
        result['replication_seed'] = context['seed']
        result['trial_order_shuffled'] = context['shuffle']
        result['run_timestamp'] = context['timestamp']
        result['model_name'] = model_name
        result['experiment_name'] = experiment_name

    return results
```

#### B. Update Query Scripts

**File**: `src/query_study1_exp1.py` (MODIFICATIONS)

```python
# Add at top
from utils.replication import (
    add_replication_args,
    initialize_replication,
    shuffle_trials,
    add_replication_metadata
)

def main():
    parser = argparse.ArgumentParser(description='Study 1 Experiment 1')
    parser.add_argument('--input', required=True, help='Input CSV')
    parser.add_argument('--output', required=True, help='Output JSON')
    parser.add_argument('--endpoint', required=True, help='vLLM endpoint')
    parser.add_argument('--model-name', required=True, help='Model name')

    # Add replication arguments
    add_replication_args(parser)

    args = parser.parse_args()

    # Initialize replication context
    context = initialize_replication(args, default_seed_offset=0)

    logger.info(f"Study 1 Exp1 - Replication {context['replication_id']}")
    logger.info(f"Seed: {context['seed']}, Shuffle: {context['shuffle']}")

    # Load trials
    trials_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(trials_df)} trials")

    # Shuffle if requested
    if context['shuffle']:
        trials_df = shuffle_trials(trials_df, context['seed'])
        logger.info(f"Shuffled trials with seed {context['seed']}")

    # Process trials (existing code)
    client = VLLMClient(args.endpoint)
    results = []

    for idx, trial in tqdm(trials_df.iterrows(), total=len(trials_df)):
        result = process_trial(trial, client, args.model_name)
        result['trial_order_in_replication'] = idx
        results.append(result)

    # Add replication metadata
    results = add_replication_metadata(
        results,
        context,
        args.model_name,
        "study1_exp1"
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} results to {args.output}")
```

**Similar modifications for**:
- `src/query_study1_exp2.py`
- `src/query_study2_exp1.py`
- `src/query_study2_exp2.py`

#### C. Pass Seed to vLLM API

**File**: `src/utils/api_client.py` (MODIFICATIONS)

```python
def _make_request(
    self,
    prompt: str,
    temperature: float,
    max_tokens: int,
    logprobs: Optional[int] = None,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,  # NEW PARAMETER
) -> Dict:
    """Make request to vLLM API with retry logic"""
    endpoint = f"{self.base_url}/v1/completions"

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if logprobs is not None:
        payload["logprobs"] = logprobs

    if stop is not None:
        payload["stop"] = stop

    if seed is not None:  # NEW
        payload["seed"] = seed

    # ... rest of function


def generate_text(
    self,
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,  # NEW PARAMETER
    extract_reasoning: bool = True,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
) -> CompletionResponse:
    """Generate text completion with optional seed"""
    temp = temperature if temperature is not None else self.config.temp_text_generation
    tokens = max_tokens if max_tokens is not None else self.config.max_tokens_text

    response = self._make_request(
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        logprobs=None,
        stop=stop,
        seed=seed,  # NEW
    )

    # ... rest of function
```

### 3. Manifest Generation Script

**File**: `scripts/generate_replicated_jobs.py` (NEW)

```python
#!/usr/bin/env python3
"""
Generate replicated experiment jobs from model configs
Creates Kubernetes Jobs with indexed parallelism for multi-replication studies
"""

import yaml
import sys
from pathlib import Path
from jinja2 import Template

def generate_replicated_job(model_config, num_replications=5, parallelism=2):
    """Generate replicated job manifest for a model"""

    template_path = Path("kubernetes/templates/job-replicated-experiments.yaml.j2")
    with open(template_path) as f:
        template = Template(f.read())

    # Determine model cache PVC based on model
    if "llama-70b" in model_config['key']:
        cache_pvc = "thomas-grace-llama-70b-cache"
    elif "deepseek" in model_config['key']:
        cache_pvc = "thomas-grace-deepseek-r1-70b-cache"
    else:
        cache_pvc = "thomas-grace-model-cache"

    # Render template
    manifest = template.render(
        MODEL_KEY=model_config['key'],
        MODEL_NAME=model_config['display_name'],
        HF_MODEL=model_config['huggingface_name'],
        GPU_COUNT=model_config['gpu']['count'],
        TP_SIZE=model_config.get('vllm', {}).get('tensor_parallel_size', 1),
        GPU_NODE_SELECTOR=model_config['gpu']['node_selector'],
        NUM_REPLICATIONS=num_replications,
        PARALLELISM=parallelism,
        MEMORY_REQUEST=model_config['resources']['memory_request'],
        MEMORY_LIMIT=model_config['resources']['memory_limit'],
        CPU_REQUEST=model_config['resources']['cpu_request'],
        CPU_LIMIT=model_config['resources']['cpu_limit'],
        MODEL_CACHE_PVC=cache_pvc,
    )

    # Write to file
    output_path = Path(f"kubernetes/generated/job-{model_config['key']}-replicated.yaml")
    with open(output_path, 'w') as f:
        f.write(manifest)

    print(f"✅ Generated: {output_path}")

def main():
    # Load model configs
    with open("config/models.yaml") as f:
        config = yaml.safe_load(f)

    # Get parameters
    num_replications = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    parallelism = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"Generating replicated jobs: {num_replications} replications, {parallelism} parallel")

    # Generate for each model
    for model_key, model_config in config['models'].items():
        model_config['key'] = model_key
        generate_replicated_job(model_config, num_replications, parallelism)

if __name__ == '__main__':
    main()
```

### 4. Orchestration Script

**File**: `scripts/run_replicated_experiments.sh` (NEW)

```bash
#!/bin/bash
# Run replicated experiments for all models

set -e

NAMESPACE="lemn-lab"
NUM_REPLICATIONS=${1:-5}
PARALLELISM=${2:-2}

echo "========================================="
echo "Replicated Experiments Launcher"
echo "========================================="
echo "Replications per model: $NUM_REPLICATIONS"
echo "Parallel pods: $PARALLELISM"
echo ""

# Generate manifests
echo "Generating replicated job manifests..."
python3 scripts/generate_replicated_jobs.py $NUM_REPLICATIONS $PARALLELISM

# List of models to run
MODELS=(
    "gemma-2b-rtx3090"
    "llama-3b-rtx3090"
    "gemma-9b-rtx3090"
    "llama-8b-rtx3090"
    "gemma-27b-rtx3090"
    "llama-70b-rtx3090"
    "deepseek-r1-70b-rtx3090"
)

# Run each model sequentially
for MODEL in "${MODELS[@]}"; do
    JOB_NAME="grace-${MODEL}-replicated"

    echo ""
    echo "========================================="
    echo "Launching: $MODEL"
    echo "========================================="

    # Apply job
    kubectl apply -f "kubernetes/generated/job-${MODEL}-replicated.yaml"

    # Wait for completion
    echo "Waiting for all replications to complete..."
    kubectl wait --for=condition=complete \
        --timeout=180m \
        "job/$JOB_NAME" \
        -n "$NAMESPACE"

    echo "✅ $MODEL complete"

    # Show results
    echo ""
    echo "Results:"
    kubectl logs -n "$NAMESPACE" -l "app=grace-experiment,model=$MODEL" --tail=20

    # Cleanup job (keep results in PVC)
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE"

    echo ""
done

echo ""
echo "========================================="
echo "✅ ALL REPLICATED EXPERIMENTS COMPLETE"
echo "========================================="
echo ""
echo "Results location: /results/<model>/replication-<N>/"
```

## Implementation Steps

### Phase 1: Script Modifications (2-3 hours)

1. **Create replication utilities**
   ```bash
   # Create new file
   touch src/utils/replication.py
   # Implement: add_replication_args, initialize_replication, etc.
   ```

2. **Update query scripts**
   ```bash
   # Modify all 4 query scripts to add:
   # - Replication args
   # - Seed initialization
   # - Trial shuffling
   # - Metadata tracking
   ```

3. **Update API client**
   ```bash
   # Modify api_client.py to pass seed to vLLM
   ```

4. **Test locally**
   ```bash
   # Test with different seeds
   python src/query_study1_exp1.py \
       --input data/test_samples/study1_sample.csv \
       --output outputs/test_rep0.json \
       --endpoint http://localhost:8000 \
       --model-name gemma-2b \
       --seed 1001 \
       --shuffle \
       --replication-id 0

   python src/query_study1_exp1.py \
       --input data/test_samples/study1_sample.csv \
       --output outputs/test_rep1.json \
       --endpoint http://localhost:8000 \
       --model-name gemma-2b \
       --seed 1002 \
       --shuffle \
       --replication-id 1

   # Verify different trial orders and results
   ```

### Phase 2: Kubernetes Setup (1-2 hours)

5. **Create Jinja2 template**
   ```bash
   mkdir -p kubernetes/templates
   # Create job-replicated-experiments.yaml.j2
   ```

6. **Create generation script**
   ```bash
   # Install Jinja2
   pip install jinja2

   # Create generate_replicated_jobs.py
   # Test generation
   python scripts/generate_replicated_jobs.py 5 2
   ```

7. **Create orchestration script**
   ```bash
   # Create run_replicated_experiments.sh
   chmod +x scripts/run_replicated_experiments.sh
   ```

### Phase 3: Testing (2-3 hours)

8. **Test single model with 2 replications**
   ```bash
   # Generate manifest for gemma-2b with 2 replications, 2 parallel
   python scripts/generate_replicated_jobs.py 2 2

   # Apply just gemma-2b
   kubectl apply -f kubernetes/generated/job-gemma-2b-rtx3090-replicated.yaml

   # Watch progress
   kubectl get pods -n lemn-lab -w

   # Check results
   kubectl run temp-check --rm -i --restart=Never --image=busybox -n lemn-lab \
       --overrides='{"spec":{"containers":[{"name":"check","image":"busybox",
       "command":["ls","-lhR","/results/gemma-2b-rtx3090/"],"stdin":true,"tty":true,
       "volumeMounts":[{"name":"results","mountPath":"/results"}]}],
       "volumes":[{"name":"results","persistentVolumeClaim":
       {"claimName":"thomas-grace-results"}}]}}'
   ```

9. **Verify outputs**
   ```bash
   # Check folder structure
   # Should have: /results/gemma-2b-rtx3090/replication-0/ and replication-1/

   # Verify metadata files exist
   # Verify different seeds in metadata.json
   # Verify JSON outputs have replication metadata fields
   ```

10. **Run full pipeline**
    ```bash
    # Run all models with 5 replications each
    ./scripts/run_replicated_experiments.sh 5 2
    ```

### Phase 4: Analysis Integration (1-2 hours)

11. **Create data aggregation script**
    ```python
    # scripts/aggregate_replications.py
    # Combines all replications into analysis-ready format
    ```

12. **Update R analysis**
    ```r
    # Add batch/replication as random effect
    # Check ICCs for replication variance
    ```

## Resource Requirements

### Per Model Replication

**Gemma 2B** (5 replications, 2 parallel):
- Peak pods: 2
- GPU-time: 5 × 30min = 2.5 GPU-hours
- Wall time: ~75 min (ceiling(5/2) × 30min)

**Llama 70B** (5 replications, 1 parallel):
- Peak pods: 1 (only 1 at a time due to 4-GPU requirement)
- GPU-time: 5 × 4 GPUs × 2.5h = 50 GPU-hours
- Wall time: ~12.5 hours (5 × 2.5h)

### Total for All Models (5 replications each)

- Total experiments: 7 models × 5 replications = 35 runs
- Total GPU-hours: ~250-300 (depending on parallelism)
- Total wall time: ~3-5 days (sequential models, parallel replications)

## Benefits

1. **Statistical Validity**: Independent samples per model, treat as repeated measures
2. **Batch Effect Detection**: Can quantify within-model variance vs between-model variance
3. **Reproducibility**: Every trial has documented seed and can be replayed
4. **Drift Monitoring**: Timestamps and deployment info track any systematic changes
5. **Flexible Analysis**: Can pool replications or treat separately based on ICC
6. **Efficient Resource Use**: Kubernetes handles parallelism natively, maximizes GPU utilization

## Risks & Mitigations

**Risk**: Replications might be too similar (low temperature, no shuffle)
- **Mitigation**: Use shuffle + seed, verify different trial orders

**Risk**: Job failures waste GPU time
- **Mitigation**: Test with 2 replications first, use backoffLimit=0 to fail fast

**Risk**: PVC write conflicts (parallel writes to same file)
- **Mitigation**: Separate folders per replication, no shared files

**Risk**: Out of storage space (35 runs × 4 experiments × file sizes)
- **Mitigation**: Monitor PVC usage, each replication ~50-100MB, need ~5-10GB total

**Risk**: Deadlock with large models (70B needs 4 GPUs, parallelism: 2 could block)
- **Mitigation**: Parameterize parallelism per model in config/models.yaml
- **Solution**: Add `replication_parallelism` field to model config:
  - Gemma 2B/9B: parallelism=3 (1 GPU each, can run 3 at a time)
  - Llama 70B: parallelism=1 (4 GPUs, only 1 at a time)
  - Generator script reads this value instead of global parallelism
- **Example config addition**:
  ```yaml
  llama-70b-rtx3090:
    # ... existing config ...
    replication_parallelism: 1  # Only 1 at a time (needs 4 GPUs)

  gemma-9b-rtx3090:
    # ... existing config ...
    replication_parallelism: 3  # Can run 3 in parallel (1 GPU each)
  ```

## Success Criteria

- [ ] Scripts accept --seed, --shuffle, --replication-id
- [ ] Trial order differs across replications
- [ ] Metadata files created with complete info
- [ ] Outputs organized in /results/<model>/replication-<N>/
- [ ] Kubernetes Jobs complete successfully with indexed parallelism
- [ ] Analysis can load and combine all replications
- [ ] ICC < 0.10 for most conditions (low batch effects)

## Timeline Estimate

- **Phase 1** (Script mods): 2-3 hours
- **Phase 2** (K8s setup): 1-2 hours
- **Phase 3** (Testing): 2-3 hours
- **Phase 4** (Analysis): 1-2 hours
- **Total**: 6-10 hours development
- **Execution**: 3-5 days for full data collection

## Next Steps

1. Approve this design
2. Implement Phase 1 (script modifications)
3. Test locally with port-forward
4. Implement Phase 2 (K8s manifests)
5. Run pilot with 2 replications on gemma-2b
6. Scale to full 5 replications across all models
