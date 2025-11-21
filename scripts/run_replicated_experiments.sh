#!/bin/bash
#
# Run replicated experiments end-to-end
#
# This script orchestrates:
# 1. Model deployment (for Exp1 which uses vLLM server)
# 2. Job manifest generation (using generate_replicated_jobs.py)
# 3. Job deployment to Kubernetes
# 4. Progress monitoring
# 5. Cleanup after completion
#
# Usage:
#   # Run single experiment with replications
#   ./scripts/run_replicated_experiments.sh \
#       --model gemma-2b-rtx3090 \
#       --experiment study1_exp1 \
#       --replications 5 \
#       --shuffle
#
#   # Run all experiments for a model
#   ./scripts/run_replicated_experiments.sh \
#       --model llama-70b-rtx3090 \
#       --all-experiments \
#       --replications 10 \
#       --base-seed 42
#
#   # Dry run (generate manifests but don't deploy)
#   ./scripts/run_replicated_experiments.sh \
#       --model gemma-9b-rtx3090 \
#       --experiment study1_exp2 \
#       --replications 3 \
#       --dry-run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
NAMESPACE="lemn-lab"
MODELS_CONFIG="config/models.yaml"
EXPERIMENTS_CONFIG="config/experiments.yaml"
OUTPUT_DIR="kubernetes/generated"
DRY_RUN=false
WAIT_FOR_COMPLETION=true
CLEANUP_AFTER=true
DEPLOY_VLLM=true

# Parse arguments
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --model MODEL                 Model key from models.yaml (e.g., gemma-2b-rtx3090)
  --experiment EXPERIMENT       Experiment to run (study1_exp1, study1_exp2, study2_exp1, study2_exp2)
                               OR --all-experiments to run all experiments
  --replications N             Number of replications to run

Optional:
  --base-seed SEED             Base random seed (default: timestamp-based)
  --shuffle                    Shuffle trial order for each replication
  --namespace NAMESPACE        Kubernetes namespace (default: lemn-lab)
  --dry-run                    Generate manifests but don't deploy
  --no-wait                    Don't wait for job completion
  --no-cleanup                 Don't clean up after completion
  --no-vllm                    Don't deploy vLLM server (assume already running)
  -h, --help                   Show this help message

Examples:
  # Run Study 1 Exp 1 with 5 replications, shuffled trials
  $0 --model gemma-2b-rtx3090 --experiment study1_exp1 --replications 5 --shuffle

  # Run all experiments for Llama 70B with 10 replications
  $0 --model llama-70b-rtx3090 --all-experiments --replications 10 --base-seed 42

  # Dry run to generate manifests only
  $0 --model gemma-9b-rtx3090 --experiment study1_exp2 --replications 3 --dry-run
EOF
    exit 1
}

# Parse command line arguments
MODEL=""
EXPERIMENT=""
ALL_EXPERIMENTS=false
REPLICATIONS=""
BASE_SEED=""
SHUFFLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --all-experiments)
            ALL_EXPERIMENTS=true
            shift
            ;;
        --replications)
            REPLICATIONS="$2"
            shift 2
            ;;
        --base-seed)
            BASE_SEED="$2"
            shift 2
            ;;
        --shuffle)
            SHUFFLE=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-wait)
            WAIT_FOR_COMPLETION=false
            shift
            ;;
        --no-cleanup)
            CLEANUP_AFTER=false
            shift
            ;;
        --no-vllm)
            DEPLOY_VLLM=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: --model is required${NC}"
    usage
fi

if [ -z "$REPLICATIONS" ]; then
    echo -e "${RED}Error: --replications is required${NC}"
    usage
fi

if [ "$ALL_EXPERIMENTS" = false ] && [ -z "$EXPERIMENT" ]; then
    echo -e "${RED}Error: --experiment or --all-experiments is required${NC}"
    usage
fi

# Banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Grace Project - Replicated Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model: ${GREEN}$MODEL${NC}"
if [ "$ALL_EXPERIMENTS" = true ]; then
    echo -e "Experiments: ${GREEN}ALL (study1_exp1, study1_exp2, study2_exp1, study2_exp2)${NC}"
else
    echo -e "Experiment: ${GREEN}$EXPERIMENT${NC}"
fi
echo -e "Replications: ${GREEN}$REPLICATIONS${NC}"
[ -n "$BASE_SEED" ] && echo -e "Base seed: ${GREEN}$BASE_SEED${NC}"
[ "$SHUFFLE" = true ] && echo -e "Shuffle: ${GREEN}enabled${NC}"
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
[ "$DRY_RUN" = true ] && echo -e "${YELLOW}DRY RUN - no deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Step 1: Generate job manifests
echo -e "${BLUE}[Step 1/5] Generating job manifests...${NC}"

GEN_ARGS="--model $MODEL --replications $REPLICATIONS"
if [ "$ALL_EXPERIMENTS" = true ]; then
    GEN_ARGS="$GEN_ARGS --all-experiments"
else
    GEN_ARGS="$GEN_ARGS --experiment $EXPERIMENT"
fi
[ -n "$BASE_SEED" ] && GEN_ARGS="$GEN_ARGS --base-seed $BASE_SEED"
[ "$SHUFFLE" = true ] && GEN_ARGS="$GEN_ARGS --shuffle"

python scripts/generate_replicated_jobs.py $GEN_ARGS --output-dir "$OUTPUT_DIR"

# Get list of generated manifests
MANIFESTS=$(find "$OUTPUT_DIR" -name "job-${MODEL}-*.yaml" -type f -mmin -1)
if [ -z "$MANIFESTS" ]; then
    echo -e "${RED}Error: No manifests generated${NC}"
    exit 1
fi

MANIFEST_COUNT=$(echo "$MANIFESTS" | wc -l)
echo -e "${GREEN}✓ Generated $MANIFEST_COUNT manifest(s)${NC}"
echo

# If dry run, stop here
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete. Manifests generated in: $OUTPUT_DIR${NC}"
    echo
    echo "Generated manifests:"
    echo "$MANIFESTS"
    exit 0
fi

# Step 2: Deploy vLLM server (if needed for Exp1)
NEEDS_VLLM=false
if [ "$ALL_EXPERIMENTS" = true ] || [[ "$EXPERIMENT" == "study1_exp1" ]] || [[ "$EXPERIMENT" == "study2_exp1" ]]; then
    NEEDS_VLLM=true
fi

if [ "$NEEDS_VLLM" = true ] && [ "$DEPLOY_VLLM" = true ]; then
    echo -e "${BLUE}[Step 2/5] Deploying vLLM server for $MODEL...${NC}"

    # Check if deployment already exists
    if kubectl get deployment -n "$NAMESPACE" "vllm-${MODEL}" &>/dev/null; then
        echo -e "${YELLOW}vLLM deployment already exists, skipping${NC}"
    else
        # Generate deployment from models.yaml using existing script
        if [ -f "scripts/generate_manifests.py" ]; then
            python scripts/generate_manifests.py --model "$MODEL" --type deployment
            DEPLOYMENT_MANIFEST=$(find kubernetes/deployments -name "deployment-${MODEL}.yaml" -type f -mmin -1 | head -1)

            if [ -f "$DEPLOYMENT_MANIFEST" ]; then
                kubectl apply -f "$DEPLOYMENT_MANIFEST" -n "$NAMESPACE"
                echo -e "${GREEN}✓ vLLM deployment created${NC}"

                # Wait for deployment to be ready
                echo "Waiting for vLLM server to be ready..."
                kubectl wait --for=condition=available --timeout=600s \
                    deployment/"vllm-${MODEL}" -n "$NAMESPACE"
                echo -e "${GREEN}✓ vLLM server ready${NC}"
            else
                echo -e "${RED}Error: Could not generate deployment manifest${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}Warning: generate_manifests.py not found, assuming vLLM is already deployed${NC}"
        fi
    fi
    echo
else
    echo -e "${BLUE}[Step 2/5] Skipping vLLM deployment${NC}"
    [ "$NEEDS_VLLM" = false ] && echo "  (Exp2 uses direct model scoring, no vLLM needed)"
    [ "$DEPLOY_VLLM" = false ] && echo "  (--no-vllm flag set)"
    echo
fi

# Step 3: Deploy replicated jobs
echo -e "${BLUE}[Step 3/5] Deploying replicated jobs...${NC}"

JOB_NAMES=()
for manifest in $MANIFESTS; do
    echo "Deploying: $(basename "$manifest")"
    kubectl apply -f "$manifest" -n "$NAMESPACE"

    # Extract job name from manifest
    JOB_NAME=$(grep "name:" "$manifest" | head -1 | awk '{print $2}')
    JOB_NAMES+=("$JOB_NAME")
done

echo -e "${GREEN}✓ Deployed $MANIFEST_COUNT job(s)${NC}"
echo

# Step 4: Monitor progress
if [ "$WAIT_FOR_COMPLETION" = true ]; then
    echo -e "${BLUE}[Step 4/5] Monitoring job progress...${NC}"
    echo "Jobs running:"
    for job_name in "${JOB_NAMES[@]}"; do
        echo "  - $job_name"
    done
    echo

    # Wait for all jobs to complete
    for job_name in "${JOB_NAMES[@]}"; do
        echo "Waiting for $job_name to complete..."
        kubectl wait --for=condition=complete --timeout=3600s \
            job/"$job_name" -n "$NAMESPACE" || {
            echo -e "${RED}Error: Job $job_name failed or timed out${NC}"
            echo "Check logs with: kubectl logs -n $NAMESPACE -l job-name=$job_name"
            exit 1
        }
        echo -e "${GREEN}✓ $job_name completed${NC}"
    done

    echo
    echo -e "${GREEN}✓ All jobs completed successfully${NC}"
    echo
else
    echo -e "${BLUE}[Step 4/5] Skipping progress monitoring (--no-wait)${NC}"
    echo "Monitor manually with:"
    for job_name in "${JOB_NAMES[@]}"; do
        echo "  kubectl get job -n $NAMESPACE $job_name -w"
    done
    echo
fi

# Step 5: Cleanup
if [ "$CLEANUP_AFTER" = true ] && [ "$WAIT_FOR_COMPLETION" = true ]; then
    echo -e "${BLUE}[Step 5/5] Cleaning up...${NC}"

    # Delete completed jobs
    for job_name in "${JOB_NAMES[@]}"; do
        echo "Deleting job: $job_name"
        kubectl delete job "$job_name" -n "$NAMESPACE"
    done

    # Optionally delete vLLM deployment
    if [ "$NEEDS_VLLM" = true ]; then
        read -p "Delete vLLM deployment for $MODEL? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl delete deployment "vllm-${MODEL}" -n "$NAMESPACE"
            echo -e "${GREEN}✓ vLLM deployment deleted${NC}"
        fi
    fi

    echo -e "${GREEN}✓ Cleanup complete${NC}"
    echo
else
    echo -e "${BLUE}[Step 5/5] Skipping cleanup${NC}"
    [ "$CLEANUP_AFTER" = false ] && echo "  (--no-cleanup flag set)"
    [ "$WAIT_FOR_COMPLETION" = false ] && echo "  (--no-wait flag set)"
    echo
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Replicated experiments complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Results location:"
echo "  /results/$MODEL/replication-{0..$((REPLICATIONS-1))}/"
echo
echo "To retrieve results from cluster:"
echo "  kubectl cp $NAMESPACE/<results-pod>:/results/$MODEL ./results-$MODEL -c grace-results"
echo -e "${BLUE}========================================${NC}"
