# Kubernetes Infrastructure Plan

## Overview

This document provides detailed Kubernetes specifications and configurations for deploying the Grace Project experiments on the National Research Platform.

## Namespace & Resource Quotas

### Namespace Definition
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: grace-experiments
  labels:
    project: grace
    environment: research
```

### Resource Quota
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: grace-compute-quota
  namespace: grace-experiments
spec:
  hard:
    requests.cpu: "200"
    requests.memory: 800Gi
    requests.nvidia.com/gpu: "15"
    persistentvolumeclaims: "20"
    pods: "50"
```

## Storage Configuration

### PVC: Input Data
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-input-data
  namespace: grace-experiments
spec:
  accessModes:
    - ReadOnlyMany  # Read-only after initial upload
  resources:
    requests:
      storage: 5Gi
  storageClassName: rook-cephfs  # CephFS for shared read access
```

### PVC: Model Weights (One per model)

**Important**: Use RBD (block storage) for model weights for best performance. Each model gets its own PVC since only one pod (the model server) needs access.

**Final Model Lineup**: Gemma-2 (2B, 9B, 27B), Llama-3 70B, GPT-NeoX 20B, BLOOM 176B

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gemma-2b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: rook-ceph-block
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gemma-9b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 25Gi
  storageClassName: rook-ceph-block
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gemma-27b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 70Gi
  storageClassName: rook-ceph-block
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-llama-70b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 160Gi
  storageClassName: rook-ceph-block
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gpt-oss-20b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 15Gi
  storageClassName: rook-ceph-block
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-model-weights-gpt-oss-120b
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 70Gi
  storageClassName: rook-ceph-block
```

**Total Model Weights Storage**: 350Gi (~0.35TB)

**Note**: GPT-OSS MoE models have much smaller checkpoint sizes (12.8GiB and 60.8GiB) than equivalent dense models

### PVC: Output Data

**Important**: Use CephFS for output data since multiple jobs write in parallel. Each job must write to unique files to avoid conflicts.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-output-data
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteMany  # Multiple jobs writing simultaneously
  resources:
    requests:
      storage: 100Gi  # Increased for all experiment outputs
  storageClassName: rook-cephfs  # CephFS for shared write access
```

### PVC: Logs

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grace-logs
  namespace: grace-experiments
spec:
  accessModes:
    - ReadWriteMany  # Multiple jobs writing logs
  resources:
    requests:
      storage: 10Gi
  storageClassName: rook-cephfs  # CephFS for shared access
```

## Model Serving Deployments

### Important NRP Cluster Policies

**Deployment Auto-Deletion**: NRP automatically deletes Deployments after **2 weeks**

**Action Required**: Request exception from cluster admins for Grace Project model servers
- Join Matrix: https://matrix.to/#/#nrp:matrix.org
- Request: "Exception for Deployments to run >2 weeks for research model serving"
- Specify: Namespace, deployment names, expected duration (2-4 weeks)

**Default Resource Limits per Pod** (can request exceptions):
- Max 2 GPUs per pod
- Max 32 GB RAM per pod  
- Max 16 CPU cores per pod

**For Grace Project Models** (6 models total):
- **Gemma-2 2B**: ✅ Within limits (1 GPU, ~10GB RAM)
- **Gemma-2 9B**: ✅ Within limits (1 GPU, ~20GB RAM)
- **Gemma-2 27B**: ⚠️ Needs exception (2 GPUs, ~60GB RAM)
- **Llama-3 70B**: ⚠️ Needs exception (4 GPUs, ~150GB RAM)
- **GPT-OSS 20B**: ✅ Within limits (1 GPU, ~15GB RAM, MoE 3.6B active)
- **GPT-OSS 120B**: ⚠️ Needs exception (2 GPUs, ~35GB RAM, MoE 5.1B active)

**Required Exceptions**:
- Multi-GPU: Gemma-27B (2), GPT-OSS-120B (2), Llama-70B (4)
- High Memory: Gemma-27B (64GB), GPT-OSS-120B (50GB), Llama-70B (160GB)
- Runtime: All deployments (>2 weeks)

**Total Resources**: 11 GPUs, ~290GB RAM (much less than dense models)

**See**: `docs/NRP_CLUSTER_GUIDE.md` for complete cluster policies

### Template: vLLM Model Server StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm-gemma-2b
  namespace: grace-experiments
  labels:
    app: vllm-server
    model: gemma-2b
spec:
  serviceName: vllm-gemma-2b
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
      model: gemma-2b
  template:
    metadata:
      labels:
        app: vllm-server
        model: gemma-2b
    spec:
      # GPU node selection
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
      
      # Service account
      serviceAccountName: grace-model-server
      
      # Init container to download model weights
      # IMPORTANT: Install packages in Docker image, not on CephFS!
      initContainers:
      - name: download-model
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          # Install huggingface_hub to local pip cache (NOT on mounted volume)
          pip install --no-cache-dir huggingface_hub
          
          # Download model to RBD volume (fast block storage)
          # Use /tmp for HF cache (local to pod, not CephFS)
          python -c "
          from huggingface_hub import snapshot_download
          snapshot_download(
              repo_id='google/gemma-2-2b',
              local_dir='/model-weights',
              cache_dir='/tmp/hf-cache'  # Local temp, not shared storage
          )
          "
        volumeMounts:
        - name: model-weights
          mountPath: /model-weights  # RBD volume, not CephFS
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-token
              key: token
        resources:
          requests:
            memory: 4Gi
            cpu: 2
          limits:
            memory: 8Gi
            cpu: 4
      
      # Main vLLM container
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=/model-weights
        - --host=0.0.0.0
        - --port=8000
        - --tensor-parallel-size=1
        - --dtype=float16
        - --max-model-len=4096
        - --gpu-memory-utilization=0.95
        - --enable-chunked-prefill
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: 32Gi
          limits:
            nvidia.com/gpu: 1
            cpu: "16"
            memory: 64Gi
        volumeMounts:
        - name: model-weights
          mountPath: /model-weights
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
      
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: grace-model-weights-gemma-2b
```

### Service for Model Server
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-gemma-2b
  namespace: grace-experiments
  labels:
    app: vllm-server
    model: gemma-2b
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: vllm-server
    model: gemma-2b
```

## Query Generation Jobs

### ConfigMap: Input Data
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grace-study-data
  namespace: grace-experiments
data:
  # Can mount study1.csv and study2.csv as files
  # Or reference PVC instead
```

### Job Template: Study 1 Experiment 1 Query Generation

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: study1-exp1-gemma-2b
  namespace: grace-experiments
  labels:
    study: study1
    experiment: exp1
    model: gemma-2b
spec:
  parallelism: 1
  completions: 1
  backoffLimit: 3
  template:
    metadata:
      labels:
        study: study1
        experiment: exp1
        model: gemma-2b
    spec:
      restartPolicy: OnFailure
      serviceAccountName: grace-query-job
      
      containers:
      - name: query-generator
        image: grace-project/query-generator:latest  # Custom image
        command:
        - python
        - /app/query_study1_exp1.py
        args:
        - --input=/data/study1.csv
        - --output=/output/exp1/gemma-2b/study1_responses.csv
        - --model-endpoint=http://vllm-gemma-2b:8000/v1
        - --model-name=gemma-2b
        - --batch-size=10
        - --timeout=60
        - --max-retries=3
        env:
        - name: LOG_LEVEL
          value: INFO
        - name: EXPERIMENT_ID
          value: exp1-study1-gemma2b
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
          limits:
            cpu: "8"
            memory: 16Gi
        volumeMounts:
        - name: input-data
          mountPath: /data
          readOnly: true
        - name: output-data
          mountPath: /output
        - name: logs
          mountPath: /logs
      
      volumes:
      - name: input-data
        persistentVolumeClaim:
          claimName: grace-input-data
      - name: output-data
        persistentVolumeClaim:
          claimName: grace-output-data
      - name: logs
        persistentVolumeClaim:
          claimName: grace-logs
```

### Job Template: Study 1 Experiment 2 Probability Extraction

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: study1-exp2-gemma-2b
  namespace: grace-experiments
spec:
  parallelism: 1
  completions: 1
  backoffLimit: 3
  template:
    metadata:
      labels:
        study: study1
        experiment: exp2
        model: gemma-2b
    spec:
      restartPolicy: OnFailure
      serviceAccountName: grace-query-job
      
      containers:
      - name: probability-extractor
        image: grace-project/probability-extractor:latest
        command:
        - python
        - /app/extract_probabilities_study1.py
        args:
        - --input=/data/study1.csv
        - --output=/output/exp2/gemma-2b/study1_probabilities.csv
        - --model-endpoint=http://vllm-gemma-2b:8000/v1
        - --model-name=gemma-2b
        - --tokens=0,1,2,3,yes,no
        - --logprobs=5
        - --temperature=1.0
        env:
        - name: LOG_LEVEL
          value: INFO
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
          limits:
            cpu: "8"
            memory: 16Gi
        volumeMounts:
        - name: input-data
          mountPath: /data
          readOnly: true
        - name: output-data
          mountPath: /output
        - name: logs
          mountPath: /logs
      
      volumes:
      - name: input-data
        persistentVolumeClaim:
          claimName: grace-input-data
      - name: output-data
        persistentVolumeClaim:
          claimName: grace-output-data
      - name: logs
        persistentVolumeClaim:
          claimName: grace-logs
```

## Structured Output Extraction Jobs

### Job: DeepSeek Structured Output Processing

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: structured-output-study1-gemma-2b
  namespace: grace-experiments
spec:
  parallelism: 1
  completions: 1
  backoffLimit: 3
  template:
    metadata:
      labels:
        task: structured-output
        study: study1
        source-model: gemma-2b
    spec:
      restartPolicy: OnFailure
      serviceAccountName: grace-extraction-job
      
      containers:
      - name: deepseek-extractor
        image: grace-project/structured-extractor:latest
        command:
        - python
        - /app/extract_structured_output.py
        args:
        - --input=/output/exp1/gemma-2b/study1_responses.csv
        - --output=/output/structured/gemma-2b/study1_structured.csv
        - --schema=/app/schemas/study1_schema.json
        - --deepseek-endpoint=http://vllm-deepseek:8000/v1
        - --max-retries=5
        - --validate
        env:
        - name: LOG_LEVEL
          value: INFO
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: deepseek-api-key
              key: api-key
              optional: true
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
          limits:
            cpu: "8"
            memory: 16Gi
        volumeMounts:
        - name: output-data
          mountPath: /output
        - name: logs
          mountPath: /logs
      
      volumes:
      - name: output-data
        persistentVolumeClaim:
          claimName: grace-output-data
      - name: logs
        persistentVolumeClaim:
          claimName: grace-logs
```

## Service Accounts & RBAC

### Service Account: Model Server
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grace-model-server
  namespace: grace-experiments
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: grace-model-server-role
  namespace: grace-experiments
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: grace-model-server-binding
  namespace: grace-experiments
subjects:
- kind: ServiceAccount
  name: grace-model-server
  namespace: grace-experiments
roleRef:
  kind: Role
  name: grace-model-server-role
  apiGroup: rbac.authorization.k8s.io
```

### Service Account: Query Jobs
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grace-query-job
  namespace: grace-experiments
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: grace-query-job-role
  namespace: grace-experiments
rules:
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: grace-query-job-binding
  namespace: grace-experiments
subjects:
- kind: ServiceAccount
  name: grace-query-job
  namespace: grace-experiments
roleRef:
  kind: Role
  name: grace-query-job-role
  apiGroup: rbac.authorization.k8s.io
```

## Secrets Management

### HuggingFace Token (for downloading models)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: huggingface-token
  namespace: grace-experiments
type: Opaque
stringData:
  token: "YOUR_HUGGINGFACE_TOKEN_HERE"
```

### DeepSeek API Key (if using external API)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: deepseek-api-key
  namespace: grace-experiments
type: Opaque
stringData:
  api-key: "YOUR_DEEPSEEK_API_KEY_HERE"
```

## Network Policies

### Restrict Inter-Pod Communication
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: grace-network-policy
  namespace: grace-experiments
spec:
  podSelector:
    matchLabels:
      project: grace
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          project: grace
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: vllm-server
    ports:
    - protocol: TCP
      port: 8000
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53  # DNS
    - protocol: UDP
      port: 53
```

## Monitoring & Observability

### ServiceMonitor for Prometheus (if using Prometheus Operator)
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: grace-vllm-metrics
  namespace: grace-experiments
spec:
  selector:
    matchLabels:
      app: vllm-server
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### PodMonitor for Job Metrics
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: grace-job-metrics
  namespace: grace-experiments
spec:
  selector:
    matchLabels:
      project: grace
  podMetricsEndpoints:
  - port: metrics
    interval: 30s
```

## Autoscaling (Optional)

### HorizontalPodAutoscaler for Query Jobs
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: grace-query-hpa
  namespace: grace-experiments
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: grace-query-workers
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Deployment Order & Dependencies

### Phase 1: Foundation
```bash
kubectl apply -f namespace.yaml
kubectl apply -f resource-quota.yaml
kubectl apply -f secrets.yaml
kubectl apply -f service-accounts.yaml
kubectl apply -f network-policies.yaml
```

### Phase 2: Storage
```bash
kubectl apply -f pvc-input-data.yaml
kubectl apply -f pvc-model-weights-*.yaml
kubectl apply -f pvc-output-data.yaml
kubectl apply -f pvc-logs.yaml
```

### Phase 3: Model Servers
```bash
kubectl apply -f statefulset-gemma-2b.yaml
kubectl apply -f service-gemma-2b.yaml
# Wait for readiness
kubectl wait --for=condition=ready pod -l app=vllm-server,model=gemma-2b -n grace-experiments --timeout=600s

# Repeat for all models...
```

### Phase 4: Verify Model Endpoints
```bash
# Port-forward and test
kubectl port-forward svc/vllm-gemma-2b 8000:8000 -n grace-experiments &
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-2b","prompt":"Hello","max_tokens":10}'
```

### Phase 5: Execute Jobs
```bash
kubectl apply -f job-study1-exp1-*.yaml
kubectl apply -f job-study2-exp1-*.yaml
# Monitor
kubectl get jobs -n grace-experiments -w
```

### Phase 6: Structured Output
```bash
# Wait for raw response jobs to complete
kubectl wait --for=condition=complete job -l experiment=exp1 -n grace-experiments --timeout=3600s

kubectl apply -f job-structured-output-*.yaml
```

## Cleanup

```bash
# Delete jobs
kubectl delete jobs -l project=grace -n grace-experiments

# Delete model servers
kubectl delete statefulsets -l app=vllm-server -n grace-experiments

# Keep PVCs for data persistence
# kubectl delete pvc -l project=grace -n grace-experiments  # Only if starting fresh

# Delete namespace (nuclear option)
# kubectl delete namespace grace-experiments
```

## NRP-Specific Configuration ✅

### Storage Classes (Configured)

All PVCs use correct NRP Ceph storage classes:
- **Input Data**: `rook-cephfs` (CephFS, ReadOnlyMany)
- **Model Weights**: `rook-ceph-block` (RBD, ReadWriteOnce) - One per model
- **Output Data**: `rook-cephfs` (CephFS, ReadWriteMany)
- **Logs**: `rook-cephfs` (CephFS, ReadWriteMany)

See `docs/NRP_STORAGE_GUIDE.md` for detailed storage best practices.

### Critical NRP Storage Rules

1. ✅ **Never install pip/conda packages on CephFS** - All Python dependencies must be in Docker images
2. ✅ **Each job writes to unique files** - Prevents write conflicts on CephFS
3. ✅ **Use RBD for model weights** - Faster than CephFS for large sequential reads
4. ✅ **Use local /tmp for HuggingFace cache** - Not shared storage
5. ❌ **Do NOT use `rook-cephfs-ucsd`** - Data may be purged without notice

### GPU Node Selectors (To Be Verified)

```yaml
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB  # Verify actual label format
```

**Action Required**: Check actual GPU node labels with:
```bash
kubectl get nodes -o json | jq '.items[].metadata.labels' | grep gpu
```

### GPU Resource Names (Standard)

Using standard Kubernetes GPU resource:
```yaml
resources:
  requests:
    nvidia.com/gpu: 1  # Standard resource name
```

### Network Configuration

NRP uses standard Kubernetes networking. No special CNI configuration needed for Grace Project.

### Image Registry

**Options**:
1. Use Docker Hub (public images): `docker.io/vllm/vllm-openai:latest`
2. Build custom images and push to Docker Hub
3. Use NRP Harbor registry if available (check with NRP admin)

**Recommendation**: Use Docker Hub for vLLM base images, build custom images for query scripts

### Resource Limits

Stay within NRP quotas:
- Total GPUs: 10-15 (request allocation if needed)
- Total CPU: ~200 cores
- Total Memory: ~800GB
- Total Storage: ~465GB (see storage allocation table)

### No Ingress Required

All access is internal to cluster (pod-to-pod). No external ingress needed.

## Estimated Deployment Timeline

- **Infrastructure Setup**: 1 day
- **Model Deployment & Testing**: 2-3 days
- **Job Development & Testing**: 2-3 days
- **Full Experiment Run**: 1-2 days (depending on dataset size)
- **Total**: 6-9 days for complete system

## Cost Estimation (NRP GPU Hours)

- **Model Deployment (Idle)**: ~15 GPUs × 24 hours × 5 days = 1,800 GPU-hours
- **Query Generation**: ~15 GPUs × 4 hours = 60 GPU-hours
- **Total**: ~1,860 GPU-hours (adjust based on actual experiment duration)
