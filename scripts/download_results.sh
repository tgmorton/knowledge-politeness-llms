#!/bin/bash
#
# Download results from K8s cluster PVC
#
# This script creates a temporary pod to download all results from the
# grace-results and grace-reasoning-traces PVCs.
#
# Usage:
#   ./scripts/download_results.sh [output-dir]
#
# Example:
#   ./scripts/download_results.sh outputs/cluster_results
#

set -e

NAMESPACE="grace-experiments"
OUTPUT_DIR="${1:-outputs/cluster_results}"

echo "======================================================================"
echo "Grace Project - Download Results from Cluster"
echo "======================================================================"
echo "Namespace: $NAMESPACE"
echo "Output directory: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/results"
mkdir -p "$OUTPUT_DIR/reasoning"

# Function to download from a PVC
download_from_pvc() {
    local PVC_NAME=$1
    local MOUNT_PATH=$2
    local LOCAL_DIR=$3

    echo "Downloading from PVC: $PVC_NAME"

    # Check if PVC exists
    if ! kubectl get pvc $PVC_NAME -n $NAMESPACE &> /dev/null; then
        echo "⚠️  PVC $PVC_NAME not found, skipping..."
        return
    fi

    # Create temporary download pod
    kubectl run data-downloader-$(date +%s) \
        --image=busybox \
        --restart=Never \
        --namespace=$NAMESPACE \
        --overrides='{
          "spec": {
            "containers": [{
              "name": "downloader",
              "image": "busybox",
              "command": ["sleep", "3600"],
              "volumeMounts": [{
                "name": "data",
                "mountPath": "'$MOUNT_PATH'"
              }]
            }],
            "volumes": [{
              "name": "data",
              "persistentVolumeClaim": {
                "claimName": "'$PVC_NAME'"
              }
            }]
          }
        }' &> /dev/null

    # Get pod name
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l run=data-downloader --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')

    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/$POD_NAME -n $NAMESPACE --timeout=60s &> /dev/null

    # List files
    echo "Files in $PVC_NAME:"
    kubectl exec $POD_NAME -n $NAMESPACE -- ls -lh $MOUNT_PATH/ || echo "  (empty)"

    # Download all files
    kubectl exec $POD_NAME -n $NAMESPACE -- sh -c "cd $MOUNT_PATH && tar cf - ." | tar xf - -C "$LOCAL_DIR" 2>/dev/null || echo "  No files to download"

    # Clean up
    kubectl delete pod $POD_NAME -n $NAMESPACE &> /dev/null

    echo "✅ Downloaded from $PVC_NAME"
    echo ""
}

# Download results
download_from_pvc "grace-results" "/data" "$OUTPUT_DIR/results"

# Download reasoning traces
download_from_pvc "grace-reasoning-traces" "/data" "$OUTPUT_DIR/reasoning"

# Summary
echo "======================================================================"
echo "✅ Download complete!"
echo "======================================================================"
echo "Results saved to:"
echo "  Main results: $OUTPUT_DIR/results/"
echo "  Reasoning traces: $OUTPUT_DIR/reasoning/"
echo ""

# List downloaded files
echo "Downloaded results:"
find "$OUTPUT_DIR/results" -type f 2>/dev/null | head -20 || echo "  (no results files)"
echo ""

echo "Downloaded reasoning traces:"
find "$OUTPUT_DIR/reasoning" -type f 2>/dev/null | head -20 || echo "  (no reasoning files)"
echo ""

echo "======================================================================"
