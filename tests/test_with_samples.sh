#!/bin/bash
#
# Test script for Grace Project experiments
#
# Runs all 4 experiments on 10-row test samples
# Requires vLLM server running at specified endpoint
#
# Usage:
#   ./tests/test_with_samples.sh [endpoint] [model_name]
#
# Example:
#   ./tests/test_with_samples.sh http://localhost:8000 gemma-2-2b-it
#

set -e  # Exit on error

# Configuration
ENDPOINT="${1:-http://localhost:8000}"
MODEL_NAME="${2:-gemma-2-2b-it}"
OUTPUT_DIR="outputs/test_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Grace Project - Phase 0 Test Suite"
echo "======================================================================"
echo "Endpoint: $ENDPOINT"
echo "Model: $MODEL_NAME"
echo "Timestamp: $TIMESTAMP"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test 1: Study 1 Experiment 1 (Raw Text)
echo -e "${BLUE}[1/4] Running Study 1 Experiment 1 (Raw Text)${NC}"
python3 src/query_study1_exp1.py \
    --input data/test_samples/study1_sample.csv \
    --output "$OUTPUT_DIR/study1_exp1_${TIMESTAMP}.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 1 Exp 1 completed successfully${NC}\n"
else
    echo -e "${RED}❌ Study 1 Exp 1 failed${NC}\n"
    exit 1
fi

# Test 2: Study 1 Experiment 2 (Probability Distributions)
echo -e "${BLUE}[2/4] Running Study 1 Experiment 2 (Probability Distributions)${NC}"
echo "Note: This makes 5 queries per trial (50 total queries)"
python3 src/query_study1_exp2.py \
    --input data/test_samples/study1_sample.csv \
    --output "$OUTPUT_DIR/study1_exp2_${TIMESTAMP}.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 1 Exp 2 completed successfully${NC}\n"
else
    echo -e "${RED}❌ Study 1 Exp 2 failed${NC}\n"
    exit 1
fi

# Test 3: Study 2 Experiment 1 (Raw Text)
echo -e "${BLUE}[3/4] Running Study 2 Experiment 1 (Raw Text)${NC}"
python3 src/query_study2_exp1.py \
    --input data/test_samples/study2_sample.csv \
    --output "$OUTPUT_DIR/study2_exp1_${TIMESTAMP}.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 2 Exp 1 completed successfully${NC}\n"
else
    echo -e "${RED}❌ Study 2 Exp 1 failed${NC}\n"
    exit 1
fi

# Test 4: Study 2 Experiment 2 (Probability Distributions)
echo -e "${BLUE}[4/4] Running Study 2 Experiment 2 (Probability Distributions)${NC}"
echo "Note: This makes 2 queries per trial (20 total queries)"
python3 src/query_study2_exp2.py \
    --input data/test_samples/study2_sample.csv \
    --output "$OUTPUT_DIR/study2_exp2_${TIMESTAMP}.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 2 Exp 2 completed successfully${NC}\n"
else
    echo -e "${RED}❌ Study 2 Exp 2 failed${NC}\n"
    exit 1
fi

# Summary
echo "======================================================================"
echo -e "${GREEN}ALL TESTS PASSED!${NC}"
echo "======================================================================"
echo "Output files saved to: $OUTPUT_DIR/"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*${TIMESTAMP}.csv
echo ""
echo "Next steps:"
echo "1. Review output files for data quality"
echo "2. Check validation reports for any issues"
echo "3. If tests pass, proceed to full dataset runs"
echo "======================================================================"
