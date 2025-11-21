#!/bin/bash
# Run Experiment 1 (vLLM API-based) locally with K8s-equivalent behavior
#
# Usage:
#   ./scripts/run_local_exp1.sh <model-name> [limit]
#
# Examples:
#   ./scripts/run_local_exp1.sh gemma-2b-rtx3090 5      # Test with 5 trials
#   ./scripts/run_local_exp1.sh llama-3b-rtx3090        # Full dataset
#
# Prerequisites:
#   - vLLM server running on localhost:8000
#   - Test data in data/test_samples/ or full data in data/

set -e  # Exit on error

# Configuration
ENDPOINT="http://localhost:8000"
OUTPUT_DIR="outputs/results"  # Mimics /data/results from K8s
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
MODEL_NAME="${1:-gemma-2b-rtx3090}"
LIMIT="${2:-}"

# Determine input files (use test samples if limit specified, full data otherwise)
if [ -n "$LIMIT" ]; then
    STUDY1_INPUT="data/test_samples/study1_sample.csv"
    STUDY2_INPUT="data/test_samples/study2_sample.csv"
    echo "Using test samples (limited to $LIMIT trials)"
else
    STUDY1_INPUT="data/study1.csv"
    STUDY2_INPUT="data/study2.csv"
    echo "Using full datasets"
fi

echo "======================================================================="
echo "Running Experiment 1 (vLLM API) Locally - K8s-Equivalent Behavior"
echo "======================================================================="
echo ""
echo "Model: $MODEL_NAME"
echo "Endpoint: $ENDPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test vLLM connection
echo "Testing vLLM connection..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: vLLM not responding at http://localhost:8000"
    echo ""
    echo "Start vLLM server first:"
    echo "  vllm serve google/gemma-2-2b-it --port 8000"
    echo ""
    echo "Or use the mock server for Mac:"
    echo "  python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it"
    exit 1
fi
echo "✅ vLLM is responding"
echo ""

# Study 1 Experiment 1
echo "======================================================================="
echo "[1/2] Study 1 Experiment 1: Knowledge Attribution - Raw Text"
echo "======================================================================="

STUDY1_OUTPUT="$OUTPUT_DIR/study1_exp1_${MODEL_NAME}_${TIMESTAMP}.json"

CMD="python3 src/query_study1_exp1.py \
    --input $STUDY1_INPUT \
    --output $STUDY1_OUTPUT \
    --endpoint $ENDPOINT \
    --model-name $MODEL_NAME"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "✅ Study 1 Exp 1 complete"
echo "   Output: $STUDY1_OUTPUT"
echo ""

# Study 2 Experiment 1
echo "======================================================================="
echo "[2/2] Study 2 Experiment 1: Politeness Judgments - Raw Text"
echo "======================================================================="

STUDY2_OUTPUT="$OUTPUT_DIR/study2_exp1_${MODEL_NAME}_${TIMESTAMP}.json"

CMD="python3 src/query_study2_exp1.py \
    --input $STUDY2_INPUT \
    --output $STUDY2_OUTPUT \
    --endpoint $ENDPOINT \
    --model-name $MODEL_NAME"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "✅ Study 2 Exp 1 complete"
echo "   Output: $STUDY2_OUTPUT"
echo ""

# Summary
echo "======================================================================="
echo "Experiment 1 Complete!"
echo "======================================================================="
echo ""
echo "Output files:"
ls -lh "$STUDY1_OUTPUT" "$STUDY2_OUTPUT"
echo ""
echo "Verify JSON format:"
echo "  head -20 $STUDY1_OUTPUT"
echo ""
echo "Next steps:"
echo "  - Review outputs to ensure they match expected format"
echo "  - Compare with K8s Job output format (when available)"
echo "  - Run full datasets by omitting the limit parameter"
echo ""
