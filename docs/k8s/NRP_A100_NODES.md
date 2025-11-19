# NRP A100 GPU Nodes - Analysis and Recommendations

## Overview

Based on the NRP node list analysis, there are **excellent A100 GPU resources** available across multiple sites. This document identifies A100 nodes and provides recommendations for Grace Project deployment.

---

## A100 Nodes Available

### Summary Statistics

| GPU Type | Count | Total GPUs | Sites | Memory Variants |
|----------|-------|------------|-------|-----------------|
| **A100-SXM4-80GB** | ~20 nodes | Variable | Multiple | 80GB HBM2e |
| **A100-80GB-PCIe** | ~10 nodes | Variable | SDSU, Mizzou, SDSC | 80GB |
| **A100-SXM4-40GB** | ~2 nodes | Variable | Mixed | 40GB |
| **A100-PCIE-40GB** | ~5 nodes | Variable | UNL, GP | 40GB |

**Total A100 Nodes**: ~37+ nodes across the cluster

---

## Detailed A100 Node List

### 80GB A100 Nodes (Recommended for Grace Project)

#### Great Plains Network (Missouri)
```
gpn-fiona-mizzou-1.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
gpn-fiona-mizzou-2.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
gpn-fiona-mizzou-3.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
gpn-fiona-mizzou-4.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
gpn-fiona-mizzou-5.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
gpn-fiona-mizzou-6.rnet.missouri.edu       NVIDIA-A100-SXM4-80GB    (Ready)
```
**6 nodes** with A100-80GB SXM4

#### Great Plains Engine Nodes
```
gp-engine.beocat.ksu.edu                   NVIDIA-A100-SXM4-80GB    (Ready)
gp-engine.hpc.okstate.edu                  NVIDIA-A100-SXM4-80GB    (Ready)
gpengine-uams.areon.net                    NVIDIA-A100-SXM4-80GB    (Ready)
gpengine-uark.areon.net                    NVIDIA-A100-SXM4-80GB    (Ready)
hcc-gpengine-shor-c5303.unl.edu            NVIDIA-A100-SXM4-80GB    (Ready)
tu.gp-engine.greatplains.net               NVIDIA-A100-SXM4-80GB    (Ready)
sphinx.sdstate.edu                         NVIDIA-A100-SXM4-80GB    (Ready)
```
**7 nodes** with A100-80GB SXM4

#### SDSC (San Diego Supercomputer Center) - node-*.sdsc.optiputer.net
```
node-1-1.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-1-2.sdsc.optiputer.net                NVIDIA-A100-80GB-PCIe    (Ready)
node-1-3.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-1-4.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-2-1.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-2-2.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-2-3.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-2-4.sdsc.optiputer.net                NVIDIA-A100-SXM4-80GB    (Ready)
node-2-10.sdsc.optiputer.net               NVIDIA-A100-80GB-PCIe    (Ready)
```
**9 nodes** with A100-80GB (SXM4 and PCIe)

#### SDSU (San Diego State University)
```
rci-nrp-gpu-01.sdsu.edu                    NVIDIA-A100-80GB-PCIe-MIG (Ready)
rci-nrp-gpu-02.sdsu.edu                    NVIDIA-A100-80GB-PCIe-MIG (Ready)
rci-nrp-gpu-03.sdsu.edu                    NVIDIA-A100-80GB-PCIe-MIG (Ready)
rci-nrp-gpu-04.sdsu.edu                    NVIDIA-A100-80GB-PCIe-MIG (Ready)
rci-nrp-gpu-05.sdsu.edu                    NVIDIA-A100-80GB-PCIe-MIG (Ready)
rci-nrp-gpu-06.sdsu.edu                    NVIDIA-A100-80GB-PCIe    (Ready)
rci-nrp-gpu-07.sdsu.edu                    NVIDIA-A100-80GB-PCIe    (Ready)
rci-nrp-gpu-08.sdsu.edu                    NVIDIA-A100-80GB-PCIe    (Ready)
```
**8 nodes** with A100-80GB PCIe (some with MIG)

**Note**: MIG (Multi-Instance GPU) nodes are partitioned and may have limited resources per partition

**Total 80GB A100 Nodes**: ~30 nodes

### 40GB A100 Nodes

#### Great Plains / University Nodes
```
hcc-gpn-argo-1.unl.edu                     NVIDIA-A100-PCIE-40GB    (Ready)
k8s-a100-01.suncorridor.org                NVIDIA-A100-PCIE-40GB    (Ready)
oru.gp-argo.greatplains.net                NVIDIA-A100-PCIE-40GB    (Ready)
ren-gp-argo-01.madren.org                  NVIDIA-A100-PCIE-40GB    (Ready)
sdsmt.gp-argo.greatplains.net              NVIDIA-A100-PCIE-40GB    (Ready)
sdsu.gp-argo.greatplains.net               NVIDIA-A100-PCIE-40GB    (Ready)
```
**6 nodes** with A100-40GB PCIe

---

## Recommendations for Grace Project

### Option 1: SDSC Nodes (Recommended)

**Advantages**:
- ✅ Co-located in San Diego (low inter-node latency)
- ✅ High-quality infrastructure (supercomputer center)
- ✅ 9 A100-80GB nodes available
- ✅ Mix of SXM4 (faster) and PCIe
- ✅ Same domain simplifies networking

**Node Selector**:
```yaml
nodeSelector:
  kubernetes.io/hostname: node-1-1.sdsc.optiputer.net
  # Or use label-based selection if available
```

**Recommended Nodes for Grace Project**:
1. `node-1-1.sdsc.optiputer.net` - A100-80GB SXM4
2. `node-1-3.sdsc.optiputer.net` - A100-80GB SXM4
3. `node-1-4.sdsc.optiputer.net` - A100-80GB SXM4
4. `node-2-1.sdsc.optiputer.net` - A100-80GB SXM4
5. `node-2-2.sdsc.optiputer.net` - A100-80GB SXM4
6. `node-2-3.sdsc.optiputer.net` - A100-80GB SXM4
7. `node-2-4.sdsc.optiputer.net` - A100-80GB SXM4

**Coverage**: 7 nodes × 4 GPUs each (if DGX A100) = potential for 28+ GPUs

### Option 2: Great Plains Network (Missouri)

**Advantages**:
- ✅ 6+ dedicated A100-80GB nodes
- ✅ Clustered in same region
- ✅ Recent Kubernetes versions

**Node Selector**:
```yaml
nodeSelector:
  kubernetes.io/hostname: gpn-fiona-mizzou-1.rnet.missouri.edu
```

**Recommended Nodes**:
1. `gpn-fiona-mizzou-1.rnet.missouri.edu`
2. `gpn-fiona-mizzou-2.rnet.missouri.edu`
3. `gpn-fiona-mizzou-3.rnet.missouri.edu`
4. `gpn-fiona-mizzou-4.rnet.missouri.edu`
5. `gpn-fiona-mizzou-5.rnet.missouri.edu`
6. `gpn-fiona-mizzou-6.rnet.missouri.edu`

### Option 3: SDSU Nodes (Local to San Diego)

**Advantages**:
- ✅ 8 A100-80GB PCIe nodes
- ✅ Also in San Diego area
- ✅ Good for backup/overflow

**Caution**:
- ⚠️ Some nodes have MIG enabled (partitioned GPUs)
- ⚠️ MIG nodes (gpu-01 through gpu-05) have limited memory per instance

**Recommended**: Use non-MIG nodes (gpu-06, gpu-07, gpu-08)

---

## Node Selector Strategies

### Strategy 1: Specific Node Assignment (Most Control)

**Use when**: You want guaranteed placement on specific nodes

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm-gemma-2b
spec:
  template:
    spec:
      nodeSelector:
        kubernetes.io/hostname: node-1-1.sdsc.optiputer.net
      # ... rest of spec
```

**Pros**: Predictable placement, known node characteristics
**Cons**: Less flexible if node goes down

### Strategy 2: Label-Based Selection (Flexible)

**Use when**: You want automatic scheduling across A100 nodes

```yaml
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
  # Or if labels exist:
  # gpu.nvidia.com/class: A100
  # nvidia.com/gpu.memory: 80GB
```

**Pros**: Automatic failover, better resource utilization
**Cons**: May schedule across different sites (higher latency)

### Strategy 3: Node Affinity (Recommended for Grace)

**Use when**: You want preferred nodes with fallback options

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - node-1-1.sdsc.optiputer.net
          - node-1-3.sdsc.optiputer.net
          - node-1-4.sdsc.optiputer.net
          - node-2-1.sdsc.optiputer.net
          - node-2-2.sdsc.optiputer.net
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - NVIDIA-A100-SXM4-80GB
          - NVIDIA-A100-80GB-PCIe
```

**Pros**: Prefers SDSC nodes, falls back to any A100-80GB
**Cons**: More complex YAML

---

## GPU Resource Names

Based on node list analysis, use standard Kubernetes GPU resource:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1  # Standard resource name
  limits:
    nvidia.com/gpu: 1
```

**Confirmed**: NRP uses standard NVIDIA GPU resource naming

---

## Multi-GPU Pod Considerations

For pods requiring multiple GPUs (Gemma-27B, Llama-70B, GPT-OSS-120B):

### Single-Node Multi-GPU (Preferred)

**Best for**: Tensor parallelism (same node, fast NVLink)

```yaml
resources:
  requests:
    nvidia.com/gpu: 4  # All 4 GPUs from same node
```

**Requirements**:
- Node must have ≥4 GPUs available
- SDSC DGX nodes likely have 8 GPUs each (perfect fit)

### Multi-Node Multi-GPU (If Needed)

**Best for**: When single node doesn't have enough GPUs

**Note**: vLLM supports multi-node tensor parallelism but adds network overhead

---

## Deployment Plan for Grace Project

### Phase 1: Test Deployment (1 model)

**Target**: SDSC node-1-1

```yaml
# Deploy Gemma-2B on single A100-80GB
nodeSelector:
  kubernetes.io/hostname: node-1-1.sdsc.optiputer.net
resources:
  requests:
    nvidia.com/gpu: 1
```

**Purpose**: Validate deployment, test vLLM, verify node access

### Phase 2: Multi-GPU Test (Gemma-27B)

**Target**: SDSC node (any with 2+ GPUs free)

```yaml
resources:
  requests:
    nvidia.com/gpu: 2
nodeSelector:
  kubernetes.io/hostname: node-1-3.sdsc.optiputer.net
```

**Purpose**: Test multi-GPU tensor parallelism

### Phase 3: Full Deployment (All 6 models)

**Resource Allocation**:

| Model | GPUs | Target Node(s) |
|-------|------|----------------|
| Gemma-2B | 1 | node-1-1.sdsc.optiputer.net |
| Gemma-9B | 1 | node-1-1.sdsc.optiputer.net (shared) |
| Gemma-27B | 2 | node-1-3.sdsc.optiputer.net |
| Llama-70B | 4 | node-1-4.sdsc.optiputer.net |
| GPT-OSS-20B | 1 | node-2-1.sdsc.optiputer.net |
| GPT-OSS-120B | 2 | node-2-2.sdsc.optiputer.net |

**Total**: 11 GPUs across 6 nodes (well within capacity)

**Alternative**: Use node affinity to let Kubernetes schedule automatically across SDSC nodes

---

## Checking Node Availability

Before deployment, verify node status:

```bash
# Check specific node
kubectl describe node node-1-1.sdsc.optiputer.net

# Check GPU allocation
kubectl describe node node-1-1.sdsc.optiputer.net | grep -A 5 "Allocated resources"

# Check all A100-80GB nodes
kubectl get nodes -o wide | grep "A100-SXM4-80GB"

# Check node labels
kubectl get nodes node-1-1.sdsc.optiputer.net -o json | jq '.metadata.labels'
```

---

## Avoiding Restricted Nodes

**Do NOT use these nodes** (restricted or disabled):

### Haosu Cluster (Restricted)
```
k8s-haosu-*.sdsc.optiputer.net   # Restricted to Hao Su's group
```

### Tide Cluster (SDSU, Restricted)
```
rci-tide-*.sdsu.edu              # Restricted to SDSU Tide cluster
```

### Scheduled for Maintenance
```
Any node with: Ready,SchedulingDisabled
Any node with: NotReady
```

---

## Important Considerations

### 1. Node Locations

**SDSC Nodes**: US West (San Diego) - Low latency to each other
**Mizzou Nodes**: US Central (Missouri) - Higher latency to SDSC
**GP Engine Nodes**: Distributed across Great Plains

**Recommendation**: Keep all models on SDSC nodes for lowest inter-pod latency

### 2. GPU Types

**SXM4 vs PCIe**:
- **SXM4**: Faster GPU-to-GPU (NVLink), better for multi-GPU
- **PCIe**: Slightly slower, but sufficient for single-GPU models

**Recommendation**: 
- Use SXM4 nodes for Llama-70B (4 GPUs, benefits from NVLink)
- PCIe is fine for all other models

### 3. Memory

**80GB vs 40GB**:
- All Grace models fit comfortably in 80GB
- Use 80GB nodes (more headroom for KV cache)
- 40GB nodes unnecessary

### 4. Shared Nodes

**DGX A100 nodes** typically have 8 GPUs:
- Can run multiple models on same node
- Example: Gemma-2B (1 GPU) + Gemma-9B (1 GPU) on node-1-1

---

## Summary & Next Steps

### ✅ Excellent A100 Availability

- **30+ A100-80GB nodes** available
- **SDSC has 9 nodes** - perfect for Grace Project
- **All Ready** and not restricted

### 📋 Recommended Approach

1. **Request access** to SDSC A100 nodes (if not automatic)
2. **Start with node-1-1.sdsc.optiputer.net** for testing
3. **Use node affinity** to prefer SDSC nodes
4. **Deploy incrementally**: 1 model → 2 models → all 6 models
5. **Monitor usage**: Check GPU allocation before adding models

### 🎯 Node Selector for Grace Project

```yaml
# Recommended node affinity (put in StatefulSet template)
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - node-1-1.sdsc.optiputer.net
          - node-1-3.sdsc.optiputer.net
          - node-1-4.sdsc.optiputer.net
          - node-2-1.sdsc.optiputer.net
          - node-2-2.sdsc.optiputer.net
          - node-2-3.sdsc.optiputer.net
          - node-2-4.sdsc.optiputer.net
```

### 🚀 Ready to Deploy

With 30+ A100-80GB nodes available, Grace Project's requirement of **11 GPUs is easily achievable**!

**Next**: Update Kubernetes manifests with node selectors and proceed to Phase 0.
