#!/bin/bash
#
# Run all 6 models sequentially on K8s cluster
#
# This script automates the sequential deployment strategy:
# 1. Deploy vLLM for model
# 2. Wait for ready
# 3. Run all experiments for that model
# 4. Wait for completion
# 5. Clean up deployment
# 6. Repeat for next model
#
# Usage:
#   ./scripts/run_all_models_sequential.sh
#

set -e

NAMESPACE="grace-experiments"

# Model configurations
# Format: "model_id|hf_model_name|deployment_file|gpu_count"
MODELS=(
    "gemma-2b|google/gemma-2-2b-it|vllm-deployment.yaml|1"
    "gemma-9b|google/gemma-2-9b-it|vllm-deployment.yaml|1"
    "gemma-27b|google/gemma-2-27b-it|vllm-deployment-large.yaml|2"
    "llama-70b|meta-llama/Llama-3-70b-chat-hf|vllm-deployment-large.yaml|2"
    # GPT-OSS models - uncomment when released
    # "gpt-oss-20b|openai/gpt-oss-20b|vllm-deployment.yaml|1"
    # "gpt-oss-120b|openai/gpt-oss-120b|vllm-deployment-large.yaml|2"
)

echo "======================================================================"
echo "Grace Project - Sequential Model Execution"
echo "======================================================================"
echo "Models to process: ${#MODELS[@]}"
echo "Estimated total time: ~6-8 hours (${#MODELS[@]} models × 1.5hrs avg)"
echo "======================================================================"
echo ""

# Ensure PVCs exist
echo "Checking PVCs..."
if ! kubectl get pvc grace-input-data -n $NAMESPACE &> /dev/null; then
    echo "Creating PVCs..."
    kubectl apply -f kubernetes/pvcs.yaml
fi

# Ensure data is uploaded
echo "Checking if data is uploaded..."
echo "If not, run: ./scripts/upload_data.sh"
echo ""

# Process each model
for MODEL_CONFIG in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_ID HF_MODEL DEPLOYMENT_FILE GPU_COUNT <<< "$MODEL_CONFIG"

    echo "======================================================================"
    echo "Processing Model: $MODEL_ID"
    echo "======================================================================"
    echo "HuggingFace model: $HF_MODEL"
    echo "GPU count: $GPU_COUNT"
    echo "Start time: $(date)"
    echo "======================================================================"
    echo ""

    # 1. Update deployment YAML with correct model
    echo "[1/6] Preparing deployment configuration..."
    cp kubernetes/$DEPLOYMENT_FILE /tmp/vllm-deployment-temp.yaml
    sed -i '' "s|google/gemma-2-2b-it|$HF_MODEL|g" /tmp/vllm-deployment-temp.yaml
    sed -i '' "s|name: vllm-.*|name: vllm-$MODEL_ID|g" /tmp/vllm-deployment-temp.yaml
    sed -i '' "s|model: .*|model: $MODEL_ID|g" /tmp/vllm-deployment-temp.yaml

    # 2. Deploy vLLM
    echo "[2/6] Deploying vLLM server..."
    kubectl apply -f /tmp/vllm-deployment-temp.yaml
    kubectl apply -f kubernetes/service.yaml

    # 3. Wait for deployment to be ready
    echo "[3/6] Waiting for vLLM server to be ready (this may take 5-10 minutes)..."
    kubectl wait --for=condition=available deployment/vllm-$MODEL_ID -n $NAMESPACE --timeout=600s

    echo "✅ vLLM server is ready!"
    echo ""

    # 4. Run experiments
    echo "[4/6] Running experiments..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Create Job for each experiment
    EXPERIMENTS=("study1-exp1" "study1-exp2" "study2-exp1" "study2-exp2")

    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        echo "  Starting $EXPERIMENT..."

        # Create Job YAML
        cat > /tmp/job-$EXPERIMENT-$MODEL_ID.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: grace-$EXPERIMENT-$MODEL_ID-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: experiment-runner
        image: grace-query-generator:latest
        imagePullPolicy: IfNotPresent
        command: ["python3", "/app/src/query_${EXPERIMENT//-/_}.py"]
        args:
          - --input=/data/input/${EXPERIMENT%%-*}.csv
          - --output=/data/results/${EXPERIMENT}_${MODEL_ID}_${TIMESTAMP}.csv
          - --reasoning-output=/data/reasoning/${EXPERIMENT}_${MODEL_ID}_${TIMESTAMP}_reasoning.jsonl
          - --endpoint=http://vllm-$MODEL_ID:8000
          - --model-name=$MODEL_ID
        resources:
          requests: {memory: "4Gi", cpu: "2"}
          limits: {memory: "4800Mi", cpu: "2400m"}
        volumeMounts:
          - {name: input-data, mountPath: /data/input, readOnly: true}
          - {name: results, mountPath: /data/results}
          - {name: reasoning-traces, mountPath: /data/reasoning}
      volumes:
        - {name: input-data, persistentVolumeClaim: {claimName: grace-input-data}}
        - {name: results, persistentVolumeClaim: {claimName: grace-results}}
        - {name: reasoning-traces, persistentVolumeClaim: {claimName: grace-reasoning-traces}}
EOF

        # Apply Job
        kubectl apply -f /tmp/job-$EXPERIMENT-$MODEL_ID.yaml

        # Wait for job to complete
        JOB_NAME="grace-$EXPERIMENT-$MODEL_ID-$TIMESTAMP"
        kubectl wait --for=condition=complete job/$JOB_NAME -n $NAMESPACE --timeout=7200s || {
            echo "⚠️  Job $JOB_NAME did not complete successfully"
            kubectl logs job/$JOB_NAME -n $NAMESPACE || true
        }

        echo "  ✅ $EXPERIMENT complete"
    done

    echo "✅ All experiments complete for $MODEL_ID"
    echo ""

    # 5. Clean up deployment
    echo "[5/6] Cleaning up vLLM deployment..."
    kubectl delete deployment vllm-$MODEL_ID -n $NAMESPACE

    echo "[6/6] Model $MODEL_ID complete!"
    echo "End time: $(date)"
    echo ""
    echo "======================================================================"
    echo ""

    # Brief pause before next model
    sleep 10
done

echo "======================================================================"
echo "✅ ALL MODELS COMPLETE!"
echo "======================================================================"
echo "Total models processed: ${#MODELS[@]}"
echo "Completion time: $(date)"
echo ""
echo "Next steps:"
echo "1. Download results: ./scripts/download_results.sh"
echo "2. Analyze data in outputs/cluster_results/"
echo "======================================================================"
