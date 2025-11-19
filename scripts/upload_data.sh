#!/bin/bash
#
# Upload data to K8s cluster PVC
#
# This script creates a temporary pod to upload data to the grace-input-data PVC.
# The data is then available to all experiment Jobs.
#
# Usage:
#   ./scripts/upload_data.sh
#

set -e

NAMESPACE="grace-experiments"
PVC_NAME="grace-input-data"

echo "======================================================================"
echo "Grace Project - Upload Data to Cluster"
echo "======================================================================"
echo "Namespace: $NAMESPACE"
echo "PVC: $PVC_NAME"
echo "======================================================================"
echo ""

# Check if PVC exists
if ! kubectl get pvc $PVC_NAME -n $NAMESPACE &> /dev/null; then
    echo "❌ PVC $PVC_NAME not found in namespace $NAMESPACE"
    echo "Please create PVCs first:"
    echo "  kubectl apply -f kubernetes/pvcs.yaml"
    exit 1
fi

echo "✅ PVC $PVC_NAME found"
echo ""

# Create temporary upload pod
echo "Creating temporary upload pod..."
kubectl run data-uploader \
    --image=busybox \
    --restart=Never \
    --namespace=$NAMESPACE \
    --overrides='{
      "spec": {
        "containers": [{
          "name": "uploader",
          "image": "busybox",
          "command": ["sleep", "3600"],
          "volumeMounts": [{
            "name": "data",
            "mountPath": "/data"
          }]
        }],
        "volumes": [{
          "name": "data",
          "persistentVolumeClaim": {
            "claimName": "'$PVC_NAME'"
          }
        }]
      }
    }'

# Wait for pod to be ready
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod/data-uploader -n $NAMESPACE --timeout=60s

# Upload files
echo ""
echo "Uploading data files..."
kubectl cp data/study1.csv $NAMESPACE/data-uploader:/data/study1.csv
kubectl cp data/study2.csv $NAMESPACE/data-uploader:/data/study2.csv

echo "✅ Files uploaded successfully!"

# List files in PVC
echo ""
echo "Files in PVC:"
kubectl exec data-uploader -n $NAMESPACE -- ls -lh /data/

# Clean up
echo ""
echo "Cleaning up temporary pod..."
kubectl delete pod data-uploader -n $NAMESPACE

echo ""
echo "======================================================================"
echo "✅ Data upload complete!"
echo "======================================================================"
echo "study1.csv and study2.csv are now available in the cluster"
echo "Experiments can read from: /data/input/"
echo "======================================================================"
