#!/bin/bash
#
# GPU Cluster Inventory Script
#
# Scans all GPU nodes in the NRP cluster and generates an inventory
# showing GPU type, memory, and count per node.
#
# Usage:
#   ./scripts/inventory_cluster_gpus.sh
#
# Output:
#   CSV-style report of all GPU nodes

set -e

echo "=========================================="
echo "NRP Cluster GPU Inventory"
echo "=========================================="
echo ""
echo "Scanning GPU nodes (printing as found)..."
echo ""

# CSV header
echo "NODE,GPU_PRODUCT,GPU_MEMORY_GB,GPU_COUNT,STATUS"
echo "---"

# Get all GPU nodes and process one at a time (streaming)
kubectl get nodes -l nvidia.com/gpu.product -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | while read node; do
    [ -z "$node" ] && continue
    # Get GPU info from node labels
    GPU_PRODUCT=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.product}' 2>/dev/null || echo "Unknown")
    GPU_MEMORY=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.memory}' 2>/dev/null || echo "0")
    GPU_COUNT=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.count}' 2>/dev/null || echo "0")
    NODE_STATUS=$(kubectl get node "$node" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")

    # Convert memory from MiB to GiB
    if [ "$GPU_MEMORY" != "0" ] && [ "$GPU_MEMORY" != "" ]; then
        GPU_MEMORY_GB=$((GPU_MEMORY / 1024))
    else
        GPU_MEMORY_GB="Unknown"
    fi

    # Map True/False to Ready/NotReady
    if [ "$NODE_STATUS" = "True" ]; then
        STATUS="Ready"
    else
        STATUS="NotReady"
    fi

    # Print immediately (streaming output)
    echo "$node,$GPU_PRODUCT,$GPU_MEMORY_GB,$GPU_COUNT,$STATUS"

    # Track GPU types for summary (append to temp file)
    echo "$GPU_PRODUCT" >> /tmp/gpu_inventory_$$.txt
done

echo ""
echo "=========================================="
echo "Summary by GPU Type"
echo "=========================================="

# Summary from collected data
if [ -f /tmp/gpu_inventory_$$.txt ]; then
    sort /tmp/gpu_inventory_$$.txt | uniq -c | sort -rn | while read count gpu_type; do
        echo "$count nodes with $gpu_type"
    done
fi

echo ""
echo "=========================================="
echo "Recommendations for Grace Project:"
echo "=========================================="

# Count GPU types from collected data
if [ -f /tmp/gpu_inventory_$$.txt ]; then
    A100_80GB_COUNT=$(grep -c "A100.*80GB" /tmp/gpu_inventory_$$.txt 2>/dev/null || echo "0")
    A100_40GB_COUNT=$(grep -c "A100.*40GB" /tmp/gpu_inventory_$$.txt 2>/dev/null || echo "0")
    RTX3090_COUNT=$(grep -c "RTX.*3090" /tmp/gpu_inventory_$$.txt 2>/dev/null || echo "0")
    V100_32GB_COUNT=$(grep -c "V100.*32GB" /tmp/gpu_inventory_$$.txt 2>/dev/null || echo "0")
    V100_16GB_COUNT=$(grep -c "V100.*16GB" /tmp/gpu_inventory_$$.txt 2>/dev/null || echo "0")

    # Clean up temp file
    rm -f /tmp/gpu_inventory_$$.txt
else
    A100_80GB_COUNT=0
    A100_40GB_COUNT=0
    RTX3090_COUNT=0
    V100_32GB_COUNT=0
    V100_16GB_COUNT=0
fi

echo ""
if [ "$A100_80GB_COUNT" -gt 0 ]; then
    echo "✅ BEST: $A100_80GB_COUNT nodes with A100-80GB available"
    echo "   → Can run all models at full precision"
    echo "   → Use nodeSelector: nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB"
elif [ "$A100_40GB_COUNT" -gt 0 ]; then
    echo "✅ EXCELLENT: $A100_40GB_COUNT nodes with A100-40GB available"
    echo "   → Can run Gemma-2B, 9B, 27B at full precision"
    echo "   → May need quantization for Llama-70B"
    echo "   → Use nodeSelector: nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB"
elif [ "$RTX3090_COUNT" -gt 0 ]; then
    echo "✅ GOOD: $RTX3090_COUNT nodes with RTX 3090 available (24GB VRAM)"
    echo "   → Can run all models with AWQ 4-bit quantization"
    echo "   → Use nodeSelector: nvidia.com/gpu.product: NVIDIA-GeForce-RTX-3090"
elif [ "$V100_32GB_COUNT" -gt 0 ]; then
    echo "⚠️  OKAY: $V100_32GB_COUNT nodes with V100-32GB available"
    echo "   → Can run smaller models (Gemma-2B, 9B) at full precision"
    echo "   → Need quantization for larger models"
    echo "   → Use nodeSelector: nvidia.com/gpu.product: Tesla-V100-SXM2-32GB"
elif [ "$V100_16GB_COUNT" -gt 0 ]; then
    echo "⚠️  LIMITED: $V100_16GB_COUNT nodes with V100-16GB available"
    echo "   → Can run Gemma-2B with quantization"
    echo "   → Not recommended for larger models"
    echo "   → Use nodeSelector: nvidia.com/gpu.product: Tesla-V100-SXM2-16GB"
else
    echo "❌ No suitable GPUs found for Grace Project"
fi

echo ""
echo "=========================================="
