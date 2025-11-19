#!/bin/bash
#
# Deploy and Test Single Model on Kubernetes (Phase 1)
#
# This script handles the complete workflow for testing one model:
# 1. Deploy vLLM server (for Experiment 1)
# 2. Wait for server to be ready
# 3. Run all 4 experiments (Study 1 & 2, Exp 1 & 2)
# 4. Download results
# 5. Clean up deployment
#
# Usage:
#   ./scripts/deploy_model_k8s.sh <model-config>
#
# Example:
#   ./scripts/deploy_model_k8s.sh gemma-2b
#
# Available configs: gemma-2b, gemma-9b, gemma-27b, llama-70b
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MODEL_CONFIG=$1
NAMESPACE="grace-experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -z "$MODEL_CONFIG" ]; then
    echo -e "${RED}Error: Model configuration required${NC}"
    echo "Usage: $0 <model-config>"
    echo "Available: gemma-2b, gemma-9b, gemma-27b, llama-70b"
    exit 1
fi

# Model-specific configurations
case $MODEL_CONFIG in
    gemma-2b)
        MODEL_NAME="google/gemma-2-2b-it"
        MODEL_SHORT="gemma2b"
        DEPLOYMENT_FILE="kubernetes/vllm-deployment.yaml"
        GPU_COUNT=1
        ;;
    gemma-9b)
        MODEL_NAME="google/gemma-2-9b-it"
        MODEL_SHORT="gemma9b"
        DEPLOYMENT_FILE="kubernetes/vllm-deployment.yaml"
        GPU_COUNT=1
        ;;
    gemma-27b)
        MODEL_NAME="google/gemma-2-27b-it"
        MODEL_SHORT="gemma27b"
        DEPLOYMENT_FILE="kubernetes/vllm-deployment-large.yaml"
        GPU_COUNT=2
        ;;
    llama-70b)
        MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
        MODEL_SHORT="llama70b"
        DEPLOYMENT_FILE="kubernetes/vllm-deployment-large.yaml"
        GPU_COUNT=2
        ;;
    *)
        echo -e "${RED}Error: Unknown model configuration: $MODEL_CONFIG${NC}"
        exit 1
        ;;
esac

echo "======================================================================"
echo -e "${BLUE}Grace Project - Kubernetes Deployment${NC}"
echo "======================================================================"
echo "Model: $MODEL_NAME"
echo "Short name: $MODEL_SHORT"
echo "GPUs: $GPU_COUNT"
echo "Timestamp: $TIMESTAMP"
echo "======================================================================"
echo ""

# Step 1: Verify cluster access
echo -e "${BLUE}[1/8] Verifying cluster access...${NC}"
if ! kubectl get nodes > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot access Kubernetes cluster${NC}"
    echo "Please check your kubeconfig and cluster connection"
    exit 1
fi
echo -e "${GREEN}✅ Cluster access verified${NC}\n"

# Step 2: Verify namespace exists
echo -e "${BLUE}[2/8] Checking namespace...${NC}"
if ! kubectl get namespace $NAMESPACE > /dev/null 2>&1; then
    echo -e "${YELLOW}Namespace doesn't exist, creating...${NC}"
    kubectl apply -f kubernetes/namespace.yaml
fi
echo -e "${GREEN}✅ Namespace ready${NC}\n"

# Step 3: Deploy vLLM server
echo -e "${BLUE}[3/8] Deploying vLLM server...${NC}"
echo "Using deployment: $DEPLOYMENT_FILE"

# Update deployment with correct model
sed "s|google/gemma-2-2b-it|$MODEL_NAME|g" $DEPLOYMENT_FILE | \
    kubectl apply -f -

echo -e "${YELLOW}Waiting for deployment to be created...${NC}"
sleep 5

DEPLOYMENT_NAME=$(kubectl get deployments -n $NAMESPACE -l app=vllm -o jsonpath='{.items[0].metadata.name}')
echo "Deployment name: $DEPLOYMENT_NAME"

echo -e "${YELLOW}Waiting for pod to be ready (this may take 5-10 minutes)...${NC}"
kubectl wait --for=condition=available \
    deployment/$DEPLOYMENT_NAME \
    -n $NAMESPACE \
    --timeout=600s

echo -e "${GREEN}✅ vLLM server deployed and ready${NC}\n"

# Step 4: Create service if needed
echo -e "${BLUE}[4/8] Ensuring service exists...${NC}"
kubectl apply -f kubernetes/service.yaml
echo -e "${GREEN}✅ Service ready${NC}\n"

# Step 5: Run Experiment 1 (vLLM-based)
echo -e "${BLUE}[5/8] Running Experiment 1 (both studies)...${NC}"

# Study 1 Experiment 1
echo -e "${YELLOW}Running Study 1 Experiment 1...${NC}"
cat kubernetes/job-exp1-template.yaml | \
    sed "s/gemma-2b/$MODEL_SHORT/g" | \
    sed "s/gemma-2-2b-it/$MODEL_NAME/g" | \
    sed "s/20250119_000000/$TIMESTAMP/g" | \
    sed "s/study1-exp1/study1-exp1-$MODEL_SHORT/g" | \
    kubectl apply -f -

kubectl wait --for=condition=complete \
    job/grace-study1-exp1-$MODEL_SHORT \
    -n $NAMESPACE \
    --timeout=3600s

echo -e "${GREEN}✅ Study 1 Experiment 1 complete${NC}"

# Study 2 Experiment 1
echo -e "${YELLOW}Running Study 2 Experiment 1...${NC}"
cat kubernetes/job-exp1-template.yaml | \
    sed "s/gemma-2b/$MODEL_SHORT/g" | \
    sed "s/gemma-2-2b-it/$MODEL_NAME/g" | \
    sed "s/20250119_000000/$TIMESTAMP/g" | \
    sed "s/study1/study2/g" | \
    sed "s|src/query_study1_exp1.py|src/query_study2_exp1.py|g" | \
    kubectl apply -f -

kubectl wait --for=condition=complete \
    job/grace-study2-exp1-$MODEL_SHORT \
    -n $NAMESPACE \
    --timeout=3600s

echo -e "${GREEN}✅ Study 2 Experiment 1 complete${NC}\n"

# Step 6: Run Experiment 2 (direct scoring)
echo -e "${BLUE}[6/8] Running Experiment 2 (both studies)...${NC}"
echo -e "${YELLOW}Note: These jobs load models directly (no vLLM server)${NC}"

# Study 1 Experiment 2
echo -e "${YELLOW}Running Study 1 Experiment 2...${NC}"
cat kubernetes/job-exp2-template.yaml | \
    sed "s/gemma-2b/$MODEL_SHORT/g" | \
    sed "s/google\/gemma-2-2b-it/${MODEL_NAME//\//\\/}/g" | \
    sed "s/20250119_000000/$TIMESTAMP/g" | \
    sed "s/study1-exp2/study1-exp2-$MODEL_SHORT/g" | \
    kubectl apply -f -

kubectl wait --for=condition=complete \
    job/grace-study1-exp2-$MODEL_SHORT \
    -n $NAMESPACE \
    --timeout=7200s

echo -e "${GREEN}✅ Study 1 Experiment 2 complete${NC}"

# Study 2 Experiment 2
echo -e "${YELLOW}Running Study 2 Experiment 2...${NC}"
cat kubernetes/job-exp2-template.yaml | \
    sed "s/gemma-2b/$MODEL_SHORT/g" | \
    sed "s/google\/gemma-2-2b-it/${MODEL_NAME//\//\\/}/g" | \
    sed "s/20250119_000000/$TIMESTAMP/g" | \
    sed "s/study1/study2/g" | \
    sed "s|src/query_study1_exp2.py|src/query_study2_exp2.py|g" | \
    kubectl apply -f -

kubectl wait --for=condition=complete \
    job/grace-study2-exp2-$MODEL_SHORT \
    -n $NAMESPACE \
    --timeout=7200s

echo -e "${GREEN}✅ Study 2 Experiment 2 complete${NC}\n"

# Step 7: Download results
echo -e "${BLUE}[7/8] Downloading results...${NC}"
OUTPUT_DIR="outputs/k8s_${MODEL_SHORT}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Create a temporary pod to access PVC
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: results-downloader
  namespace: $NAMESPACE
spec:
  containers:
  - name: downloader
    image: busybox
    command: ["sleep", "300"]
    volumeMounts:
    - name: results
      mountPath: /data/results
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: grace-results
  restartPolicy: Never
EOF

kubectl wait --for=condition=ready pod/results-downloader -n $NAMESPACE --timeout=60s

# Copy results
kubectl cp $NAMESPACE/results-downloader:/data/results/$TIMESTAMP "$OUTPUT_DIR/"

# Clean up downloader pod
kubectl delete pod results-downloader -n $NAMESPACE

echo -e "${GREEN}✅ Results downloaded to: $OUTPUT_DIR${NC}\n"

# Step 8: Clean up
echo -e "${BLUE}[8/8] Cleaning up...${NC}"

# Delete jobs
kubectl delete job --all -n $NAMESPACE

# Delete deployment (free up GPU)
kubectl delete deployment $DEPLOYMENT_NAME -n $NAMESPACE

echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# Summary
echo "======================================================================"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETE! 🎉${NC}"
echo "======================================================================"
echo "Model: $MODEL_NAME"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review output files"
echo "2. Run validation on results"
echo "3. Deploy next model with: $0 <next-model>"
echo "======================================================================"
