#!/bin/bash
# Run all 4 experiments locally via port-forward
# Requires: vLLM running and port-forwarded to localhost:8000

set -e  # Exit on error

ENDPOINT="http://localhost:8000"
MODEL="google/gemma-2-2b-it"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================="
echo "Running All 4 Grace Project Experiments (Local Test)"
echo "======================================================================="
echo ""
echo "Endpoint: $ENDPOINT"
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
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

# Study 1 Experiment 1
echo "======================================================================="
echo "Study 1 Experiment 1: Knowledge Attribution - Raw Text Responses"
echo "======================================================================="
python3 src/query_study1_exp1.py \
    --input data/test_samples/study1_sample.csv \
    --output outputs/study1_exp1_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""

# Study 1 Experiment 2
echo "======================================================================="
echo "Study 1 Experiment 2: Knowledge Attribution - Probability Distributions"
echo "======================================================================="
python3 src/query_study1_exp2.py \
    --input data/test_samples/study1_sample.csv \
    --output outputs/study1_exp2_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""

# Study 2 Experiment 1
echo "======================================================================="
echo "Study 2 Experiment 1: Politeness Judgments - Raw Text Responses"
echo "======================================================================="
python3 src/query_study2_exp1.py \
    --input data/test_samples/study2_sample.csv \
    --output outputs/study2_exp1_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""

# Study 2 Experiment 2
echo "======================================================================="
echo "Study 2 Experiment 2: Politeness Judgments - Probability Distributions"
echo "======================================================================="
python3 src/query_study2_exp2.py \
    --input data/test_samples/study2_sample.csv \
    --output outputs/study2_exp2_gemma2b_${TIMESTAMP}.csv \
    --endpoint $ENDPOINT \
    --model-name $MODEL

echo ""
echo "======================================================================="
echo "All Experiments Complete!"
echo "======================================================================="
echo ""
echo "Output files:"
ls -lh outputs/study*_${TIMESTAMP}.csv
echo ""
echo "To view results:"
echo "  cat outputs/study1_exp1_gemma2b_${TIMESTAMP}.csv"
echo "  cat outputs/study1_exp2_gemma2b_${TIMESTAMP}.csv"
echo "  cat outputs/study2_exp1_gemma2b_${TIMESTAMP}.csv"
echo "  cat outputs/study2_exp2_gemma2b_${TIMESTAMP}.csv"
