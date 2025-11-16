# National Research Platform (NRP) Storage Guide

## Overview

This document provides essential information about storage on the NRP Kubernetes cluster, including CephFS and RBD storage classes, best practices, and restrictions that must be followed for the Grace Project.

---

## Storage Architecture

NRP uses **Ceph** for persistent storage with two primary types:

1. **CephFS** (Ceph Filesystem) - Shared filesystem, ReadWriteMany
2. **RBD** (Rados Block Device) - Block storage, ReadWriteOnce

---

## Available Storage Classes

### CephFS (Shared Filesystem) - ReadWriteMany

| StorageClass | Region | Restrictions | Storage Type | Use Case |
|--------------|--------|--------------|--------------|----------|
| `rook-cephfs` | US West | None | Spinning + NVME meta | **Recommended for Grace input data** |
| `rook-cephfs-ucsd` | US West (local) | **See policy below** | NVME only | Fast, temporary computation data only |
| `rook-cephfs-central` | US Central | None | Spinning + NVME meta | Alternative region |
| `rook-cephfs-east` | US East | None | Mixed | Alternative region |
| `rook-cephfs-south-east` | US South East | None | Spinning + NVME meta | Alternative region |
| `rook-cephfs-pacific` | Hawaii+Guam | None | Spinning + NVME meta | Alternative region |
| `rook-cephfs-haosu` | US West (local) | Hao Su and Ravi cluster only | Spinning + NVME meta | Restricted access |
| `rook-cephfs-tide` | US West (local) | SDSU Tide cluster only | Spinning + NVME meta | Restricted access |
| `rook-cephfs-fullerton` | US West (local) | None | Spinning + NVME meta | Alternative region |

### RBD (Block Storage) - ReadWriteOnce

| StorageClass | Region | Restrictions | Storage Type | Use Case |
|--------------|--------|--------------|--------------|----------|
| `rook-ceph-block` (**default**) | US West | None | Spinning + NVME meta | **Recommended for model weights** |
| `rook-ceph-block-central` | US Central | None | Spinning + NVME meta | Alternative region |
| `rook-ceph-block-east` | US East | None | Mixed | Alternative region |
| `rook-ceph-block-south-east` | US South East | None | Spinning + NVME meta | Alternative region |
| `rook-ceph-block-pacific` | Hawaii+Guam | None | Spinning + NVME meta | Alternative region |
| `rook-ceph-block-tide` | US West (local) | SDSU Tide cluster only | Spinning + NVME meta | Restricted access |
| `rook-ceph-block-fullerton` | US West (local) | None | Spinning + NVME meta | Alternative region |

---

## Storage Selection for Grace Project

### Recommended Storage Strategy

| Component | Storage Class | Access Mode | Rationale |
|-----------|---------------|-------------|-----------|
| **Input Data** (study1.csv, study2.csv) | `rook-cephfs` | ReadWriteMany | Small files, need shared read access across all jobs |
| **Model Weights** (per model) | `rook-ceph-block` | ReadWriteOnce | Large files (10-150GB), single pod access, fastest performance |
| **Output Data** (results CSVs) | `rook-cephfs` | ReadWriteMany | Need write access from multiple jobs in parallel |
| **Logs** | `rook-cephfs` | ReadWriteMany | Multiple jobs writing logs simultaneously |

### Why Not Use rook-cephfs-ucsd?

**DO NOT USE `rook-cephfs-ucsd` for Grace Project**, despite being faster (NVME):

⚠️ **UCSD NVMe CephFS Policy**:
> "The filesystem is very fast and small. We expect all data on it to be used for currently running computation and then promptly deleted. We reserve the right to purge any data that's staying there longer than needed at admin's discretion **without any notifications**."

**Risks**:
- Data may be deleted without warning
- Not suitable for experiments that run over days/weeks
- Input data and model weights would be at risk
- Output data could be lost before retrieval

**Use Case for rook-cephfs-ucsd**: Only for truly temporary scratch space that can be regenerated

---

## Critical Restrictions

### 🚫 NO Conda or PIP on CephFS

**STRICTLY PROHIBITED**:
```bash
# ❌ DO NOT DO THIS on CephFS volumes
pip install --target /cephfs/my-packages package-name
conda install --prefix /cephfs/my-env package-name
```

**Why**: Installing Python packages on shared filesystems causes:
- File locking issues
- Performance degradation for all users
- Potential corruption

**Solution for Grace Project**:
- ✅ Install all dependencies in Docker images (build time)
- ✅ Use Python virtual environments on local pod storage (EmptyDir)
- ✅ Never install packages on mounted CephFS volumes

### 🚫 Avoiding Write Conflicts on CephFS

**DO NOT** open the same file for write from multiple parallel clients:

```python
# ❌ BAD: Multiple jobs writing to same file
# Job 1, 2, 3 all doing:
with open('/cephfs/output/results.csv', 'a') as f:  # CONFLICT!
    f.write(data)
```

**Why This Causes Problems**:
- File locking issues (jobs block each other)
- Data corruption
- Race conditions

**Solutions**:

#### Option 1: Unique Files Per Job (RECOMMENDED)
```python
# ✅ GOOD: Each job writes to unique file
import os
job_id = os.environ.get('JOB_NAME', 'unknown')
output_file = f'/cephfs/output/{job_id}_results.csv'

with open(output_file, 'w') as f:
    f.write(data)
```

#### Option 2: Write to Local, Copy to CephFS
```python
# ✅ GOOD: Write locally first, then copy
# Write to pod's local storage
with open('/tmp/results.csv', 'w') as f:
    f.write(data)

# Copy to CephFS after complete
shutil.copy('/tmp/results.csv', f'/cephfs/output/{job_id}_results.csv')
```

#### Option 3: Coordination/Locking (COMPLEX)
```python
# ✅ GOOD but complex: Use file locking
import fcntl

with open('/cephfs/output/results.csv', 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
    f.write(data)
    fcntl.flock(f, fcntl.LOCK_UN)  # Release
```

---

## Best Practices for Grace Project

### 1. Input Data (ReadWriteMany CephFS)

**Use Case**: study1.csv, study2.csv need to be read by all jobs

**Strategy**:
```yaml
# PVC for input data
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-input-data
spec:
  accessModes:
    - ReadOnlyMany  # Read-only after initial upload
  storageClassName: rook-cephfs
  resources:
    requests:
      storage: 5Gi
```

**Upload Once**:
```bash
# Create temporary pod to upload data
kubectl run data-uploader --image=busybox \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"grace-input-data"}}],"containers":[{"name":"uploader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}'

# Copy files
kubectl cp data/study1.csv data-uploader:/data/study1.csv
kubectl cp data/study2.csv data-uploader:/data/study2.csv

# Verify
kubectl exec data-uploader -- ls -lh /data/

# Cleanup
kubectl delete pod data-uploader
```

**Mount as ReadOnly in Jobs**:
```yaml
volumeMounts:
  - name: input-data
    mountPath: /data
    readOnly: true  # Prevent accidental writes
```

### 2. Model Weights (ReadWriteOnce RBD)

**Use Case**: Each model needs dedicated storage for weights (10-150GB)

**Strategy**: One PVC per model
```yaml
# Separate PVC for each model
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gemma-2b
spec:
  accessModes:
    - ReadWriteOnce  # Only one pod at a time
  storageClassName: rook-ceph-block  # Fastest access
  resources:
    requests:
      storage: 20Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-llama-8b
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: rook-ceph-block
  resources:
    requests:
      storage: 50Gi
```

**Why RBD for Model Weights**:
- ✅ Fastest I/O (block device)
- ✅ Only one pod (the model server) needs access
- ✅ No write conflicts (single writer)
- ✅ Better performance for large sequential reads

**Download Models in Init Container**:
```yaml
initContainers:
  - name: download-model
    image: python:3.11-slim
    command: ["/bin/bash", "-c"]
    args:
      - |
        pip install huggingface_hub
        python -c "
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='google/gemma-2-2b',
            local_dir='/model-weights',
            cache_dir='/tmp/hf-cache'  # Use local temp, not CephFS!
        )
        "
    volumeMounts:
      - name: model-weights
        mountPath: /model-weights
```

### 3. Output Data (ReadWriteMany CephFS)

**Use Case**: Multiple jobs writing results in parallel

**Strategy**: Unique files per job
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-output-data
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: rook-cephfs
  resources:
    requests:
      storage: 100Gi
```

**Organized Directory Structure**:
```
/cephfs/grace-output/
├── exp1/
│   ├── gemma-2b/
│   │   ├── study1_responses_20250115_120000.csv
│   │   └── study2_responses_20250115_120000.csv
│   ├── gemma-9b/
│   │   ├── study1_responses_20250115_120500.csv
│   │   └── study2_responses_20250115_120500.csv
│   └── ...
├── exp2/
│   ├── gemma-2b/
│   │   ├── study1_probabilities_20250116_140000.csv
│   │   └── study2_probabilities_20250116_140000.csv
│   └── ...
└── structured/
    ├── gemma-2b/
    │   ├── study1_structured_20250117_100000.csv
    │   └── study2_structured_20250117_100000.csv
    └── ...
```

**Python Implementation**:
```python
import os
from datetime import datetime

# Get unique identifier from job
job_name = os.environ.get('JOB_NAME', 'unknown')
model_name = os.environ.get('MODEL_NAME', 'unknown')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create unique output path
output_dir = f'/output/exp1/{model_name}'
os.makedirs(output_dir, exist_ok=True)

output_file = f'{output_dir}/study1_responses_{timestamp}.csv'

# Write to unique file (no conflicts)
df.to_csv(output_file, index=False)
```

### 4. Logs (ReadWriteMany CephFS)

**Strategy**: Unique log files per job
```python
import logging
import os

job_name = os.environ.get('JOB_NAME', 'unknown')
log_file = f'/logs/{job_name}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to stdout for kubectl logs
    ]
)
```

---

## File Access Best Practices

### ✅ DO

1. **Use Unique Files**: Each job writes to its own file
   ```python
   output_file = f'/output/{job_id}_{timestamp}.csv'
   ```

2. **Close Files Promptly**: Don't hold file handles open
   ```python
   with open(file, 'w') as f:
       f.write(data)
   # File automatically closed
   ```

3. **Favor Copy Operations**: Copy to CephFS after local write
   ```python
   # Write locally
   with open('/tmp/data.csv', 'w') as f:
       f.write(data)
   
   # Copy to CephFS
   shutil.copy('/tmp/data.csv', '/cephfs/output/data.csv')
   ```

4. **Install Packages in Docker Images**:
   ```dockerfile
   # Dockerfile
   FROM python:3.11-slim
   RUN pip install pandas numpy httpx pydantic
   # Packages baked into image, not on CephFS
   ```

5. **Use Local Temp Space for Scratch**:
   ```yaml
   volumes:
     - name: tmp
       emptyDir: {}  # Local to pod, fast, auto-cleaned
   ```

### ❌ DON'T

1. **Don't Install Packages on CephFS**:
   ```bash
   ❌ pip install --target /cephfs/packages package
   ❌ conda install --prefix /cephfs/env package
   ```

2. **Don't Write to Same File from Multiple Jobs**:
   ```python
   ❌ # Job 1, 2, 3 all writing to same file
   with open('/cephfs/results.csv', 'a') as f:
       f.write(data)
   ```

3. **Don't Keep Files Open Long-Term**:
   ```python
   ❌ f = open('/cephfs/data.csv', 'w')
   # ... do lots of work ...
   f.write(data)  # File open for too long
   f.close()
   ```

4. **Don't Use rook-cephfs-ucsd for Persistent Data**:
   ```yaml
   ❌ storageClassName: rook-cephfs-ucsd  # Data may be purged!
   ```

---

## Performance Optimization

### Storage Performance Characteristics

| Storage Type | Sequential Read | Random Read | Write | Use Case |
|--------------|----------------|-------------|-------|----------|
| RBD (block) | Fastest | Fast | Fast | Model weights, databases |
| CephFS (spinning+NVME meta) | Good | Moderate | Good | Shared data, outputs |
| CephFS (NVME - ucsd) | Fastest | Fastest | Fastest | Temp computation only |
| EmptyDir (local) | Fastest | Fastest | Fastest | Temporary scratch |

### Recommendations for Grace Project

1. **Model Loading** (large sequential reads):
   - Use RBD (`rook-ceph-block`) for model weights
   - Pre-download models in init containers
   - Don't re-download on every pod restart

2. **CSV Processing** (small random reads/writes):
   - CephFS (`rook-cephfs`) is sufficient
   - Use unique files to avoid locking
   - Batch writes to reduce I/O operations

3. **Intermediate Processing**:
   - Use EmptyDir for temporary files
   - Only write final results to CephFS
   ```yaml
   volumes:
     - name: scratch
       emptyDir:
         sizeLimit: 10Gi
   ```

---

## Monitoring Storage Usage

### Check PVC Usage
```bash
# List all PVCs
kubectl get pvc -n grace-experiments

# Check specific PVC details
kubectl describe pvc grace-output-data -n grace-experiments

# See actual usage (requires metrics server)
kubectl exec -it <pod-name> -- df -h /output
```

### Ceph Dashboard

NRP provides Grafana dashboards for monitoring Ceph usage:
- General Ceph dashboard: https://grafana.nrp-nautilus.io/

Current pool usage:
- `rook`: 73.1 TB / 188.6 TB used
- `rook-haosu`: 808.6 TB / 1.4 PB used
- `rook-central`: 206.9 TB / 591.3 TB used
- Other pools available

---

## Cleanup Strategy

### During Development
```bash
# Delete completed jobs but keep PVCs
kubectl delete jobs -l project=grace -n grace-experiments

# Keep model weights and input data
# Only delete output data if needed
```

### After Experiments Complete
```bash
# 1. Download all output data first!
kubectl cp grace-experiments/<pod>:/output ./local-backup/

# 2. Delete output PVC (can recreate if needed)
kubectl delete pvc grace-output-data -n grace-experiments

# 3. Keep model weights PVCs for future runs
# (or delete to free space and re-download later)

# 4. Keep input data PVC (small, reusable)
```

### Complete Cleanup
```bash
# Nuclear option: delete everything
kubectl delete namespace grace-experiments

# This deletes all PVCs, pods, jobs, services
# Make sure you've backed up all data first!
```

---

## Troubleshooting

### Issue: "Volume is already mounted"

**Symptom**: RBD volume fails to mount in new pod

**Cause**: Previous pod still has exclusive lock

**Solution**:
```bash
# Delete old pod first
kubectl delete pod <old-pod-name>

# Wait for graceful shutdown (up to 30s)
kubectl wait --for=delete pod/<old-pod-name> --timeout=60s

# Then start new pod
kubectl apply -f new-pod.yaml
```

### Issue: "No space left on device"

**Symptom**: Writes to PVC fail

**Cause**: PVC is full

**Solution**:
```bash
# Check current usage
kubectl exec <pod> -- df -h /mount-path

# Option 1: Clean up old files
kubectl exec <pod> -- rm /mount-path/old-data/*

# Option 2: Resize PVC (if storage class supports it)
kubectl edit pvc <pvc-name>
# Change storage request, save

# Option 3: Create new larger PVC, copy data
```

### Issue: Slow writes to CephFS

**Symptom**: CSV writes are very slow

**Cause**: Many small writes, file contention, or metadata overhead

**Solutions**:
1. Batch writes:
   ```python
   # ❌ Slow: Write row by row
   for row in data:
       df.to_csv(file, mode='a')
   
   # ✅ Fast: Write all at once
   df.to_csv(file, index=False)
   ```

2. Use buffering:
   ```python
   with open(file, 'w', buffering=8192) as f:
       f.write(data)
   ```

3. Write locally first:
   ```python
   # Write to fast local storage
   df.to_csv('/tmp/data.csv')
   # Copy to CephFS
   shutil.copy('/tmp/data.csv', '/cephfs/output/data.csv')
   ```

---

## Summary for Grace Project

### Storage Allocation Plan

| PVC Name | Storage Class | Size | Access Mode | Purpose |
|----------|---------------|------|-------------|---------|
| `grace-input-data` | `rook-cephfs` | 5Gi | ReadOnlyMany | study1.csv, study2.csv |
| `grace-model-weights-gemma-2b` | `rook-ceph-block` | 20Gi | ReadWriteOnce | Gemma-2B weights |
| `grace-model-weights-gemma-9b` | `rook-ceph-block` | 30Gi | ReadWriteOnce | Gemma-9B weights |
| `grace-model-weights-llama-8b` | `rook-ceph-block` | 50Gi | ReadWriteOnce | Llama-8B weights |
| `grace-model-weights-llama-70b` | `rook-ceph-block` | 150Gi | ReadWriteOnce | Llama-70B weights |
| `grace-model-weights-deepseek` | `rook-ceph-block` | 100Gi | ReadWriteOnce | DeepSeek weights |
| `grace-output-data` | `rook-cephfs` | 100Gi | ReadWriteMany | All experiment outputs |
| `grace-logs` | `rook-cephfs` | 10Gi | ReadWriteMany | Job logs |

**Total**: ~465Gi (~0.5TB)

### Key Takeaways

1. ✅ Use `rook-ceph-block` for model weights (fastest, single-pod access)
2. ✅ Use `rook-cephfs` for input/output data (shared access needed)
3. ✅ Each job writes to unique files (no conflicts)
4. ✅ Install Python packages in Docker images, not on CephFS
5. ✅ Close files promptly, use local temp space for scratch
6. ❌ Never use `rook-cephfs-ucsd` for persistent data
7. ❌ Never install conda/pip on CephFS volumes
8. ❌ Never write to same file from multiple jobs

Following these practices ensures reliable, performant storage for the Grace Project experiments.
