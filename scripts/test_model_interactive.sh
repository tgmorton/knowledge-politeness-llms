#!/bin/bash
# Interactive model testing script
# Guides user through testing each model step by step

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================="
echo "Grace Project - Interactive Model Testing"
echo -e "=======================================================================${NC}"
echo ""

# Model configurations
declare -A MODELS
MODELS["1"]="google/gemma-2-2b-it|gemma-2b-rtx3090|Gemma-2 2B (Smallest, baseline)"
MODELS["2"]="meta-llama/Llama-3.2-3B-Instruct|llama-3b-rtx3090|Llama-3.2 3B (Small Llama)"
MODELS["3"]="google/gemma-2-9b-it|gemma-9b-rtx3090|Gemma-2 9B (Medium)"
MODELS["4"]="meta-llama/Llama-3.1-8B-Instruct|llama-8b-rtx3090|Llama-3.1 8B (Medium)"

echo "Available models for local testing:"
echo ""
for key in 1 2 3 4; do
    IFS='|' read -r hf_name model_key desc <<< "${MODELS[$key]}"
    echo -e "  ${GREEN}[$key]${NC} $desc"
    echo "      Model: $hf_name"
    echo "      Key: $model_key"
    echo ""
done

echo "Multi-GPU models (K8s only):"
echo -e "  ${YELLOW}[5]${NC} Gemma-2 27B (2 GPUs required)"
echo -e "  ${YELLOW}[6]${NC} Llama-3.1 70B (4 GPUs required)"
echo -e "  ${YELLOW}[7]${NC} DeepSeek-R1 70B (4 GPUs required + reasoning)"
echo ""

# Select model
read -p "Select model to test [1-4, or 'q' to quit]: " choice

if [ "$choice" = "q" ]; then
    echo "Exiting."
    exit 0
fi

if [ -z "${MODELS[$choice]}" ]; then
    echo -e "${RED}Invalid choice. Please select 1-4.${NC}"
    exit 1
fi

IFS='|' read -r HF_MODEL_NAME MODEL_KEY MODEL_DESC <<< "${MODELS[$choice]}"

echo ""
echo -e "${BLUE}======================================================================="
echo "Testing: $MODEL_DESC"
echo -e "=======================================================================${NC}"
echo ""
echo "HuggingFace Model: $HF_MODEL_NAME"
echo "Model Key: $MODEL_KEY"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}Warning: vLLM not found.${NC}"
    echo ""
    echo "Options:"
    echo "  1. Install vLLM (Linux/CUDA only): pip install vllm"
    echo "  2. Use mock server (Mac/M1): python3 tests/local_vllm_mock.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Step 1: Start vLLM
echo -e "${BLUE}Step 1: Start vLLM Server${NC}"
echo ""
echo "In a separate terminal, run:"
echo ""
echo -e "${GREEN}source venv-grace/bin/activate"
echo "vllm serve $HF_MODEL_NAME \\"
echo "  --port 8000 \\"
echo "  --dtype bfloat16 \\"
echo "  --max-model-len 4096 \\"
echo -e "  --gpu-memory-utilization 0.9${NC}"
echo ""
echo "Or for Mac/M1:"
echo -e "${GREEN}python3 tests/local_vllm_mock.py --model $HF_MODEL_NAME${NC}"
echo ""
read -p "Press Enter once vLLM server is running..."

# Test connection
echo ""
echo -e "${BLUE}Testing vLLM connection...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ vLLM server is responding${NC}"
else
    echo -e "${RED}❌ vLLM server not responding at http://localhost:8000${NC}"
    echo ""
    echo "Please start the vLLM server first and try again."
    exit 1
fi

# Step 2: Run test
echo ""
echo -e "${BLUE}Step 2: Run Experiment 1 (5 trial test)${NC}"
echo ""
echo "This will run:"
echo "  - Study 1 Experiment 1: 5 trials (knowledge attribution)"
echo "  - Study 2 Experiment 1: 5 trials (politeness judgments)"
echo ""
read -p "Press Enter to start test..."

echo ""
./scripts/run_local_exp1.sh "$MODEL_KEY" 5

# Step 3: Validate output
echo ""
echo -e "${BLUE}Step 3: Validate Output${NC}"
echo ""

# Find the output files (most recent)
STUDY1_OUTPUT=$(ls -t outputs/results/study1_exp1_${MODEL_KEY}_*.json 2>/dev/null | head -1)
STUDY2_OUTPUT=$(ls -t outputs/results/study2_exp1_${MODEL_KEY}_*.json 2>/dev/null | head -1)

if [ -z "$STUDY1_OUTPUT" ] || [ -z "$STUDY2_OUTPUT" ]; then
    echo -e "${RED}❌ Output files not found!${NC}"
    exit 1
fi

echo "Output files:"
echo "  Study 1: $STUDY1_OUTPUT"
echo "  Study 2: $STUDY2_OUTPUT"
echo ""

# Validate JSON
echo "Validating JSON format..."
python3 << EOF
import json
import sys

try:
    # Load Study 1
    with open('$STUDY1_OUTPUT') as f:
        study1 = json.load(f)

    print(f"✅ Study 1: Valid JSON with {len(study1)} results")

    if study1:
        print(f"   Fields: {', '.join(study1[0].keys())}")
        print(f"   Sample response: {study1[0]['response'][:80]}...")

    # Load Study 2
    with open('$STUDY2_OUTPUT') as f:
        study2 = json.load(f)

    print(f"✅ Study 2: Valid JSON with {len(study2)} results")

    if study2:
        print(f"   Fields: {', '.join(study2[0].keys())}")
        print(f"   Sample response: {study2[0]['response'][:80]}...")

    print()
    print("✅ All validations passed!")

except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================================="
    echo "✅ Model Test Complete: $MODEL_DESC"
    echo -e "=======================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review outputs manually:"
    echo "     cat $STUDY1_OUTPUT | head -50"
    echo ""
    echo "  2. Test next model (if desired)"
    echo "     ./scripts/test_model_interactive.sh"
    echo ""
    echo "  3. Run full dataset (300 + 2,424 trials):"
    echo "     ./scripts/run_local_exp1.sh $MODEL_KEY"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Validation failed. Check output files.${NC}"
    exit 1
fi
