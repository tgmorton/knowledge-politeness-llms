#!/bin/bash
# Run Experiment 1 (vLLM-based) for both studies on FULL datasets
# Requires: vLLM running and port-forwarded to localhost:8000
#
# Study 1: 300 trials
# Study 2: 2,424 trials
# Total: 2,724 queries (~15-20 minutes)

set -e  # Exit on error

ENDPOINT="http://localhost:8000"
MODEL="google/gemma-2-2b-it"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================="
echo "Running Experiment 1 (vLLM) on Full Datasets"
echo "======================================================================="
echo ""
echo "Endpoint: $ENDPOINT"
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Study 1: 300 trials (knowledge attribution)"
echo "Study 2: 2,424 trials (politeness judgments)"
echo "Total: 2,724 queries"
echo ""

# Test vLLM connection
echo "Testing vLLM connection..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: vLLM not responding at http://localhost:8000"
    echo "Make sure port-forward is running:"
    echo "  kubectl port-forward -n lemn-lab svc/vllm-gemma-2b 8000:8000"
    exit 1
fi
echo "✅ vLLM is responding"
echo ""

read -p "This will run 2,724 queries (~15-20 min). Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Study 1 Experiment 1 - FULL DATASET (300 trials)
echo "======================================================================="
echo "Study 1 Experiment 1: Knowledge Attribution - Raw Text Responses"
echo "Input: data/study1.csv (300 trials)"
echo "======================================================================="
python3 src/query_study1_exp1.py \
    --input data/study1.csv \
    --output outputs/study1_exp1_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""

# Study 2 Experiment 1 - FULL DATASET (2,424 trials)
echo "======================================================================="
echo "Study 2 Experiment 1: Politeness Judgments - Raw Text Responses"
echo "Input: data/study2.csv (2,424 trials)"
echo "======================================================================="
python3 src/query_study2_exp1.py \
    --input data/study2.csv \
    --output outputs/study2_exp1_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""
echo "======================================================================="
echo "Experiment 1 Complete for Both Studies!"
echo "======================================================================="
echo ""
echo "Output files:"
ls -lh outputs/study*_exp1_gemma2b_${TIMESTAMP}.csv
echo ""
echo "Results:"
wc -l outputs/study*_exp1_gemma2b_${TIMESTAMP}.csv
echo ""
echo "Note: Experiment 2 requires direct model access (--model-path)"
echo "Cannot run via vLLM port-forward. Will run as Kubernetes Jobs later."
