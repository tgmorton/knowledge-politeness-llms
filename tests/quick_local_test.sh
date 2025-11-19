#!/bin/bash
#
# Quick Local Test Script for M1 Mac
#
# Tests query scripts with just 2 trials to verify everything works
# before deploying to Kubernetes
#
# Usage:
#   1. Start local mock server in one terminal:
#      python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it
#
#   2. Run this script in another terminal:
#      ./tests/quick_local_test.sh
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENDPOINT="http://localhost:8000"
MODEL_NAME="gemma-2-2b-it"
OUTPUT_DIR="outputs/local_test"

echo "======================================================================"
echo "Grace Project - Quick Local Test (M1 Mac)"
echo "======================================================================"
echo "Testing with 2 trials from each study"
echo "Endpoint: $ENDPOINT"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if server is running
echo -e "${BLUE}Checking if server is running...${NC}"
if curl -s "$ENDPOINT/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running${NC}\n"
else
    echo -e "${YELLOW}⚠️  Server not detected at $ENDPOINT${NC}"
    echo "Please start the local server first:"
    echo "  python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it"
    echo ""
    exit 1
fi

# Test 1: Study 1 Experiment 1 (just 2 trials)
echo -e "${BLUE}[1/4] Testing Study 1 Experiment 1 (2 trials)${NC}"
python3 src/query_study1_exp1.py \
    --input data/test_samples/study1_sample.csv \
    --output "$OUTPUT_DIR/study1_exp1_test.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME" \
    --limit 2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 1 Exp 1 passed${NC}\n"
else
    echo -e "${RED}❌ Study 1 Exp 1 failed${NC}\n"
    exit 1
fi

# Test 2: Study 1 Experiment 2 (2 trials = 10 queries!)
echo -e "${BLUE}[2/4] Testing Study 1 Experiment 2 (2 trials, 10 queries)${NC}"
echo -e "${YELLOW}Note: This makes 5 queries per trial (10 total)${NC}"
python3 src/query_study1_exp2.py \
    --input data/test_samples/study1_sample.csv \
    --output "$OUTPUT_DIR/study1_exp2_test.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME" \
    --limit 2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 1 Exp 2 passed${NC}\n"
else
    echo -e "${RED}❌ Study 1 Exp 2 failed${NC}\n"
    exit 1
fi

# Test 3: Study 2 Experiment 1 (2 trials)
echo -e "${BLUE}[3/4] Testing Study 2 Experiment 1 (2 trials)${NC}"
python3 src/query_study2_exp1.py \
    --input data/test_samples/study2_sample.csv \
    --output "$OUTPUT_DIR/study2_exp1_test.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME" \
    --limit 2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 2 Exp 1 passed${NC}\n"
else
    echo -e "${RED}❌ Study 2 Exp 1 failed${NC}\n"
    exit 1
fi

# Test 4: Study 2 Experiment 2 (2 trials = 4 queries)
echo -e "${BLUE}[4/4] Testing Study 2 Experiment 2 (2 trials, 4 queries)${NC}"
python3 src/query_study2_exp2.py \
    --input data/test_samples/study2_sample.csv \
    --output "$OUTPUT_DIR/study2_exp2_test.csv" \
    --endpoint "$ENDPOINT" \
    --model-name "$MODEL_NAME" \
    --limit 2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Study 2 Exp 2 passed${NC}\n"
else
    echo -e "${RED}❌ Study 2 Exp 2 failed${NC}\n"
    exit 1
fi

# Summary
echo "======================================================================"
echo -e "${GREEN}ALL TESTS PASSED! 🎉${NC}"
echo "======================================================================"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.csv
echo ""
echo "Your scripts work correctly! Next steps:"
echo "1. Review output CSVs to verify data quality"
echo "2. Adjust K8s resource limits (see below)"
echo "3. Deploy to Kubernetes cluster"
echo ""
echo -e "${YELLOW}IMPORTANT: Kubernetes Deployment Adjustments${NC}"
echo "Based on NRP policy review, you should:"
echo ""
echo "Option A (Recommended - No Exception Needed):"
echo "  Edit kubernetes/vllm-deployment.yaml:"
echo "    memory: \"32Gi\"  (request) and \"38Gi\" (limit)"
echo "    cpu: \"16\" (request) and \"19\" (limit)"
echo ""
echo "Option B (Requires Exception Approval):"
echo "  Keep current settings (64Gi RAM, 32 CPU)"
echo "  Request exception via NRP Matrix chat:"
echo "  https://matrix.to/#/#nrp:matrix.org"
echo ""
echo "======================================================================"
