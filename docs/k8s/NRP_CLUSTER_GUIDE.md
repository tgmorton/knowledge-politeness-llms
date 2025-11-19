# National Research Platform (NRP) Nautilus Cluster Guide

## Overview

NRP Nautilus is a **distributed Kubernetes cluster** spanning over 75 sites across the United States, with additional sites in Asia and Europe. This guide covers cluster architecture, access methods, resource allocation, and best practices for the Grace Project.

**Official Resources**:
- Dashboard: https://nautilus.optiputer.net/
- Grafana: Real-time cluster monitoring
- Matrix Chat: Community support and admin contact

---

## Cluster Architecture

### Geographic Distribution

NRP Nautilus is a **geographically distributed** cluster with nodes at 75+ sites:
- **US Coverage**: Sites across all regions (West, Central, East, South East)
- **International**: Sites in Hawaii, Guam, Asia, and Europe
- **Latency Considerations**: Pod placement can affect network performance

**View all sites**: Check the [NRP Dashboard](https://nautilus.optiputer.net/)

### Resource Summary

| Resource | Range per Node | Notes |
|----------|----------------|-------|
| **CPUs** | 16 - 384 cores | Varies by node type |
| **GPUs** | Up to 16 per node | A100, V100, T4, etc. |
| **FPGAs** | Up to 4 per node | Specialized workloads |
| **Memory (RAM)** | 16 GB - 1.6 TB | Varies by node |
| **Local Disk** | 107.3 GB - 17.8 TB | Ephemeral storage |

**Real-time availability**: 
- Portal: [Resources page](https://nautilus.optiputer.net/resources)
- Grafana: [Cluster summary dashboard](https://grafana.nrp-nautilus.io/)

---

## Access Methods

NRP provides **three ways** to use the cluster, ordered from easiest to most flexible:

### 1. JupyterHub Platform (Easiest)

**Best for**: Interactive data science, exploratory analysis, prototyping

**Access**: https://jupyter.nrp-nautilus.io/

**Features**:
- Web-based Jupyter notebooks
- No Kubernetes knowledge required
- Choose hardware specs when spawning
- Run notebooks as usual

**Requirements**:
- ✅ Must be part of a namespace
- ✅ Institution must be part of CILogon
- ✅ Browser-based access

**Documentation**: [JupyterHub docs](https://ucsd-prp.gitlab.io/userdocs/running/jupyter/)

**Use Case for Grace Project**: 
- ❌ Not recommended for Grace Project
- Limited to interactive workloads
- Cannot run distributed model serving
- Better for post-experiment analysis

### 2. Coder Platform (Medium Difficulty)

**Best for**: Development environments, persistent workspaces, code editing

**Access**: https://coder.nrp-nautilus.io/

**Features**:
- JupyterHub-like web experience
- Full development environment (VS Code in browser)
- No Kubernetes knowledge required
- Persistent workspaces

**Requirements**:
- ✅ Must be part of a namespace
- ✅ Cluster admin approval required (ask in Matrix)
- ✅ Institution must be part of CILogon

**Documentation**: [Coder docs](https://ucsd-prp.gitlab.io/userdocs/running/coder/)

**Use Case for Grace Project**:
- ⚠️ Possible for development/testing
- Not ideal for production experiments
- Limited control over pod specifications

### 3. Kubernetes (kubectl) ✅ **RECOMMENDED FOR GRACE PROJECT**

**Best for**: Production workloads, custom deployments, fine-grained resource control

**Access**: Command-line via `kubectl`

**Features**:
- Full control over resources
- Custom software stacks
- Specific resource requirements
- Parallel job execution
- Model serving deployments

**Requirements**:
- ✅ Must be part of a namespace
- ✅ Institution must be part of CILogon
- ✅ Basic Kubernetes knowledge
- ✅ Complete [Basic Kubernetes tutorial](https://ucsd-prp.gitlab.io/userdocs/tutorial/basic/)

**Use Case for Grace Project**:
- ✅ **HIGHLY RECOMMENDED**
- Perfect for model serving (StatefulSets)
- Ideal for batch processing (Jobs)
- Enables parallel experiment execution
- Full control over GPU allocation

---

## Getting Started with Kubernetes

### Prerequisites

1. **Namespace Membership**: You must belong to at least one namespace
2. **CILogon Access**: Your institution must be part of CILogon
3. **kubectl Setup**: Install and configure kubectl
4. **Basic Knowledge**: Complete the [Kubernetes tutorial](https://ucsd-prp.gitlab.io/userdocs/tutorial/basic/)

**How to get a namespace**:
- Follow instructions on [Getting Started page](https://ucsd-prp.gitlab.io/userdocs/start/get-access/)
- Faculty, researchers, or postdocs can request namespace admin role
- Students need supervisor approval (via email or Matrix)

### Key Kubernetes Concepts for NRP

#### 1. Namespace

**What**: An isolated environment for running workloads

**Key Points**:
- Creates logical separation between projects
- Can invite other users to your namespace
- Can access multiple namespaces
- Each namespace has ≥1 admin

**For Grace Project**:
```bash
# Create namespace (if admin)
kubectl create namespace grace-experiments

# Set default namespace
kubectl config set-context --current --namespace=grace-experiments

# Verify
kubectl config view --minify | grep namespace:
```

**Who Can Be Namespace Admin**:
- Faculty members, researchers, postdocs (contact via Matrix)
- PhD/Masters students (with supervisor's request via email/Matrix)

#### 2. Container Images

**What**: Standalone package with code, runtime, libraries, and dependencies

**Where to Store**:
- Docker Hub (public)
- NRP-hosted GitLab Container Registry
- GitHub Container Registry
- Private registries

**For Grace Project**:
```bash
# Build image
docker build -t myusername/grace-query-generator:latest .

# Push to Docker Hub
docker push myusername/grace-query-generator:latest

# Use in Kubernetes
# Image: docker.io/myusername/grace-query-generator:latest
```

**Tip**: Store images in NRP-hosted GitLab for faster pulls within cluster

#### 3. Pods, Jobs, and Deployments

##### Pods

**What**: Smallest deployable unit, one or more containers

**Characteristics**:
- Ephemeral (can be deleted/evicted)
- Destroyed after **6 hours** if not managed by controller
- No automatic restart on failure

**Use Case for Grace Project**:
- ❌ Do NOT use standalone pods for Grace Project
- Too unreliable for long-running model servers

**Example**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: busybox
    command: ["sleep", "3600"]
```

⚠️ **Caution**: Standalone pods are automatically destroyed after 6 hours!

##### Jobs

**What**: Run tasks to completion, with retry logic

**Characteristics**:
- Runs until successful completion
- Retries on failure
- Tracks completed pods
- Supports parallelism
- **Preferred for batch processing**

**Use Case for Grace Project**:
- ✅ **IDEAL for query generation jobs**
- ✅ **IDEAL for probability extraction jobs**
- ✅ **IDEAL for structured output jobs**
- Run to completion with automatic retries

**Resource Limits**:
- No specific GPU limit for jobs
- No specific CPU/memory limit for jobs
- Can request resources as needed

**Restrictions**:
- ❌ Do not run long `sleep` inside jobs
- ❌ Do not submit >400 jobs at once
- ✅ Delete failed jobs to stop pod creation

**Example**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: query-study1-gemma2b
spec:
  parallelism: 1
  completions: 1
  backoffLimit: 3
  template:
    spec:
      containers:
      - name: query
        image: grace-project/query-generator:latest
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
      restartPolicy: OnFailure
```

##### Deployments

**What**: Manages long-running applications, ensures desired replicas

**Characteristics**:
- Maintains desired number of replicas
- Rolling updates and rollbacks
- Auto-restarts failed pods
- **Automatically deleted after 2 weeks** unless exception granted

**Use Case for Grace Project**:
- ✅ **IDEAL for model serving (vLLM servers)**
- ✅ Can use StatefulSets (deployment variant) for persistent identity
- ⚠️ Request exception to keep running >2 weeks

**Resource Limits per Pod**:
- Max 2 GPUs per pod (without exception)
- Max 32 GB RAM per pod
- Max 16 CPU cores per pod

**Note**: For model serving, you may need to request exceptions for:
- More than 2 GPUs per pod (e.g., Llama-70B needs 4 GPUs)
- Running longer than 2 weeks

**Example**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gemma-2b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: 32Gi
```

### Comparison: Job vs Pod vs Deployment

| Feature | Job | Pod | Deployment |
|---------|-----|-----|------------|
| **Purpose** | Task to completion | Single instance | Long-running service |
| **Failure Handling** | Retries automatically | No retry | Restarts pods |
| **Parallelism** | Yes (array/parallel jobs) | No | Yes (replicas) |
| **Max GPU** | No limit (request as needed) | 2 GPUs | 2 GPUs per pod |
| **Max RAM** | No limit | 32 GB | 32 GB per pod |
| **Max CPU** | No limit | 16 cores | 16 cores per pod |
| **Max Runtime** | Until completion | 6 hours | 2 weeks (then deleted) |
| **Best For** | Batch processing | Quick tests | Model serving |
| **Grace Project Use** | ✅ Query/extraction jobs | ❌ Not recommended | ✅ vLLM servers |

**Recommendation for Grace Project**:
- Use **Deployments** (or StatefulSets) for model serving
- Use **Jobs** for all data processing tasks

---

## Resource Allocation

### Requests vs Limits

**Request**: Minimum guaranteed resources (used for scheduling)
```yaml
resources:
  requests:
    cpu: "4"
    memory: 8Gi
    nvidia.com/gpu: 1
```

**Limit**: Maximum allowed resources (enforced at runtime)
```yaml
resources:
  limits:
    cpu: "8"
    memory: 16Gi
    nvidia.com/gpu: 1
```

### Important Behaviors

| Scenario | What Happens |
|----------|--------------|
| Node can't satisfy request | Pod won't be scheduled (stays Pending) |
| Container exceeds memory limit | Pod is killed (OOMKilled) |
| Container exceeds CPU limit | CPU is throttled (slowed down) |
| GPU limit exceeded | Not applicable (can't over-provision GPU) |

### Best Practices for Grace Project

```yaml
# Model Serving Pod (Gemma-2B)
resources:
  requests:
    nvidia.com/gpu: 1
    cpu: "8"
    memory: 32Gi
  limits:
    nvidia.com/gpu: 1  # Same as request
    cpu: "16"          # 2x request for bursts
    memory: 64Gi       # 2x request for safety

# Query Generation Job
resources:
  requests:
    cpu: "4"
    memory: 8Gi
  limits:
    cpu: "8"
    memory: 16Gi
```

**Tips**:
- Always set GPU request = GPU limit (no oversubscription)
- Set memory limit 1.5-2x request for safety margin
- Set CPU limit 1.5-2x request for burst capacity
- Monitor actual usage and adjust

---

## Storage on NRP

See `NRP_STORAGE_GUIDE.md` for complete details.

**Quick Reference**:
- **Ephemeral Storage**: Local disk on nodes (107.3 GB - 17.8 TB)
- **Persistent Storage**: Ceph (CephFS for shared, RBD for block)
- **Temporary**: `emptyDir` volumes (pod-local, auto-cleaned)

---

## System Load and Performance

### Understanding CPU Load

When you run `top`, you'll see:
```
%Cpu(s): 26.9 us,  1.5 sy,  0.0 ni, 71.5 id,  0.0 wa,  0.0 hi,  0.1 si,  0.0 st
```

**Key Metrics**:
- `us` (user): 26.9% - Time spent on user computations
- `sy` (system): 1.5% - Time spent on kernel tasks
- `id` (idle): 71.5% - CPU idle time
- Other metrics: wait, hardware interrupt, software interrupt, steal

### Red Flags 🚩

**High System Load (sy > 15%)**:
- Indicates inefficient code
- Common causes:
  - Too many threads (excessive context switching)
  - Inefficient file I/O (many small files)
  - Excessive system calls

**If Your Pod is Causing High Load**:
1. Check thread count: Reduce parallelism
2. Check I/O patterns: Batch file operations
3. Use profiling tools: `htop`, `perf`, `strace`
4. Optimize code to reduce kernel time

**Example of Good vs Bad**:
```python
# ❌ BAD: Too many threads on small tasks
from multiprocessing import Pool
with Pool(processes=128) as pool:  # Too many!
    results = pool.map(small_task, data)

# ✅ GOOD: Reasonable thread count
with Pool(processes=8) as pool:  # Matches CPU cores
    results = pool.map(small_task, data)

# ❌ BAD: Many small file writes
for item in data:
    with open(f'/output/{item}.txt', 'w') as f:
        f.write(str(item))

# ✅ GOOD: Batch writes
with open('/output/all_data.txt', 'w') as f:
    for item in data:
        f.write(f"{item}\n")
```

### Monitoring Your Pods

```bash
# Check pod resource usage
kubectl top pod <pod-name> -n grace-experiments

# Check node resource usage
kubectl top node

# Get detailed pod metrics
kubectl describe pod <pod-name> -n grace-experiments

# View pod logs
kubectl logs <pod-name> -n grace-experiments

# Execute commands in pod
kubectl exec -it <pod-name> -n grace-experiments -- top
```

---

## Cluster Policies and Restrictions

### Automatic Deletions

| Resource | Policy | Reason |
|----------|--------|--------|
| **Standalone Pods** | Deleted after 6 hours | Prevent abandoned pods |
| **Deployments** | Deleted after 2 weeks | Prevent long-running idle resources |
| **Jobs** | Kept until manually deleted | Allow batch processing |

**For Grace Project**:
- ✅ Request exception for model server Deployments (>2 weeks)
- ✅ Use Jobs for all processing tasks (no time limit)
- ❌ Don't use standalone pods

### Job Restrictions

- ❌ No long `sleep` commands in jobs
- ❌ Don't submit >400 jobs at once
- ✅ Delete failed jobs promptly (stops pod creation loop)

### Resource Limits (Default, Can Request Exceptions)

**Per Deployment Pod**:
- Max 2 GPUs
- Max 32 GB RAM
- Max 16 CPU cores

**For Jobs**:
- No inherent GPU/CPU/memory limits
- Request what you need

**For Grace Project**:
- Request exception for Llama-70B (needs 4 GPUs)
- May need exception for >32GB RAM for some models

### How to Request Exceptions

1. Join NRP Matrix chat
2. Contact cluster admins
3. Explain use case (model serving for research)
4. Specify requirements (e.g., "4 GPUs for Llama-70B, 2 weeks runtime")
5. Wait for approval

**Matrix**: https://matrix.to/#/#nrp:matrix.org

---

## Best Practices for Grace Project

### 1. Use the Right Workload Type

| Task | Use | Why |
|------|-----|-----|
| Model serving | Deployment/StatefulSet | Long-running, needs restarts |
| Query generation | Job | Batch processing, runs to completion |
| Probability extraction | Job | Batch processing |
| Structured output | Job | Batch processing |

### 2. Resource Allocation Strategy

**Model Servers** (Deployments):
```yaml
# Request what you need, limit slightly higher
resources:
  requests:
    nvidia.com/gpu: 1
    cpu: "8"
    memory: 32Gi
  limits:
    nvidia.com/gpu: 1
    cpu: "16"
    memory: 64Gi
```

**Processing Jobs**:
```yaml
# No GPU needed, moderate CPU/memory
resources:
  requests:
    cpu: "4"
    memory: 8Gi
  limits:
    cpu: "8"
    memory: 16Gi
```

### 3. Efficient I/O

```python
# ✅ GOOD: Batch operations
df = pd.read_csv('/input/study1.csv')
# ... process all rows ...
df.to_csv('/output/results.csv', index=False)

# ❌ BAD: Row-by-row I/O
for row in data:
    df = pd.DataFrame([row])
    df.to_csv(f'/output/row_{i}.csv')
```

### 4. Proper Error Handling

```python
# In job scripts
import sys
try:
    # Main processing
    process_data()
except Exception as e:
    logging.error(f"Job failed: {e}")
    sys.exit(1)  # Signal failure to Kubernetes
```

### 5. Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/job.log'),
        logging.StreamHandler()  # Also to stdout for kubectl logs
    ]
)

# Log progress
logging.info(f"Processed {i}/{total} rows")
```

---

## Troubleshooting Common Issues

### Pod Stuck in Pending

**Symptom**: `kubectl get pods` shows pod in `Pending` state

**Causes**:
1. Insufficient resources (GPU, CPU, memory)
2. No nodes match nodeSelector
3. PVC not available

**Solutions**:
```bash
# Check why pod is pending
kubectl describe pod <pod-name> -n grace-experiments

# Look for events like:
# - "Insufficient nvidia.com/gpu"
# - "Insufficient memory"
# - "No nodes match nodeSelector"

# Reduce resource requests or wait for resources
```

### Pod OOMKilled

**Symptom**: Pod shows `OOMKilled` status

**Cause**: Container exceeded memory limit

**Solutions**:
1. Increase memory limit
2. Optimize code to use less memory
3. Process data in smaller batches

```yaml
# Increase memory
resources:
  limits:
    memory: 64Gi  # Was 32Gi
```

### Job Not Completing

**Symptom**: Job creates pods repeatedly, never completes

**Causes**:
1. Code has bugs causing crashes
2. `backoffLimit` reached
3. Wrong `completions` setting

**Solutions**:
```bash
# Check job status
kubectl describe job <job-name> -n grace-experiments

# Check pod logs
kubectl logs <pod-name> -n grace-experiments

# Delete job to stop pod creation
kubectl delete job <job-name> -n grace-experiments

# Fix code and redeploy
```

### High System Load Warning

**Symptom**: Cluster admin contacts you about high system load

**Causes**: See "System Load and Performance" section above

**Solutions**:
1. Reduce thread count
2. Batch file operations
3. Profile code with `perf` or `htop`
4. Optimize hot paths

---

## Grace Project Deployment Strategy

### Phase 1: Setup

```bash
# 1. Verify namespace access
kubectl get ns | grep grace

# 2. Create namespace (if admin)
kubectl create namespace grace-experiments

# 3. Set as default
kubectl config set-context --current --namespace=grace-experiments

# 4. Create PVCs (see NRP_STORAGE_GUIDE.md)
kubectl apply -f kubernetes/storage/
```

### Phase 2: Model Deployment

```bash
# Deploy model servers as Deployments/StatefulSets
kubectl apply -f kubernetes/models/

# Request exception for >2 week runtime
# Contact admins in Matrix

# Monitor deployment
kubectl get pods -w
kubectl logs -f vllm-gemma-2b-0
```

### Phase 3: Run Experiments

```bash
# Submit jobs for experiment execution
kubectl apply -f kubernetes/jobs/exp1/
kubectl apply -f kubernetes/jobs/exp2/

# Monitor jobs
kubectl get jobs -w

# Check specific job
kubectl describe job study1-exp1-gemma2b
kubectl logs job/study1-exp1-gemma2b
```

### Phase 4: Data Retrieval

```bash
# Download results from PVC
kubectl cp <pod-name>:/output ./local-results/

# Or use a dedicated data-retrieval pod
kubectl apply -f kubernetes/data-retrieval-pod.yaml
kubectl cp data-retrieval:/output ./local-results/
```

---

## Resource Requirements for Grace Project

### Estimated Cluster Usage

| Component | Pods | GPUs | CPU | Memory | Duration |
|-----------|------|------|-----|--------|----------|
| Model Servers (5) | 5 | 10 | 80 | 320GB | 2-4 weeks |
| Query Jobs (concurrent) | 10 | 0 | 40 | 80GB | 1-2 days |
| Extraction Jobs | 10 | 0 | 40 | 80GB | 1-2 days |

**Total Peak**:
- ~15 pods running
- 10 GPUs
- ~120 CPU cores
- ~400GB memory

**Storage**: See NRP_STORAGE_GUIDE.md (~465GB total)

---

## Getting Help

### Official Resources

- **Documentation**: https://ucsd-prp.gitlab.io/userdocs/
- **Matrix Chat**: https://matrix.to/#/#nrp:matrix.org
- **Dashboard**: https://nautilus.optiputer.net/
- **Grafana**: https://grafana.nrp-nautilus.io/

### Community Support

**Matrix Channels**:
- `#general`: General questions
- `#kubernetes`: Kubernetes-specific help
- `#storage`: Storage issues
- `#gpu`: GPU-related questions

**Best Practices for Asking**:
1. Provide namespace name
2. Include relevant pod/job names
3. Share error messages and logs
4. Describe what you've already tried

---

## Summary for Grace Project

### ✅ Recommended Approach

1. **Access Method**: Kubernetes (kubectl)
2. **Model Serving**: Deployments/StatefulSets with exception for >2 weeks
3. **Data Processing**: Jobs (no time limit)
4. **Storage**: See NRP_STORAGE_GUIDE.md
5. **Resources**: Request exceptions where needed (>2 GPUs, >2 weeks)

### 📋 Pre-Deployment Checklist

- [ ] Namespace access confirmed
- [ ] kubectl configured
- [ ] Kubernetes tutorial completed
- [ ] Exception requested for long-running deployments
- [ ] Exception requested for >2 GPU pods (if needed)
- [ ] Storage strategy understood (see NRP_STORAGE_GUIDE.md)
- [ ] Container images built and pushed
- [ ] Resource requests/limits defined
- [ ] Monitoring plan in place

### 🎯 Key Takeaways

1. ✅ Use **Jobs** for batch processing (query generation, extraction)
2. ✅ Use **Deployments** for model serving (with exception request)
3. ✅ Request exceptions early (>2 weeks, >2 GPUs)
4. ❌ Never use standalone pods (6 hour limit)
5. ✅ Monitor system load (keep sy <15%)
6. ✅ Use Matrix for questions and exceptions

Following these guidelines ensures smooth execution of Grace Project experiments on NRP Nautilus!
