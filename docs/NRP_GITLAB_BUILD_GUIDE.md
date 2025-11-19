# Building Docker Images with NRP GitLab CI/CD

This guide explains how to use the National Research Platform's GitLab installation to automatically build and deploy Docker images for the Grace Project.

## Why Use NRP GitLab?

**Benefits:**
- ✅ Built-in container registry (faster than DockerHub from cluster)
- ✅ Automatic CI/CD with cluster resources
- ✅ Free builds with GPUs available
- ✅ Nightly backups to Google storage
- ✅ Direct integration with Kubernetes cluster

**Our Use Case:**
- Build `grace-query-generator` image with all dependencies
- Automatically rebuild on code changes
- Store images in NRP registry for fast cluster access

---

## Step 1: Create GitLab Account and Repository

### 1.1 Register
Visit: https://gitlab.nrp-nautilus.io

Use your institutional credentials or create account.

### 1.2 Create New Project

1. Click **New Project** → **Create blank project**
2. Project name: `grace-project`
3. Visibility: Private (recommended for research)
4. Initialize with README: No (we'll push existing code)

### 1.3 Push Existing Code

```bash
# Add GitLab as remote
cd /path/to/grace-project
git remote add nrp https://gitlab.nrp-nautilus.io/<your-username>/grace-project.git

# Push code
git push -u nrp main
```

---

## Step 2: Configure Container Registry

### 2.1 View Registry URL

In your GitLab project:
1. Go to **Deploy** → **Container Registry**
2. Note your registry URL:
   ```
   gitlab-registry.nrp-nautilus.io/<your-username>/grace-project
   ```

### 2.2 Login to Registry (Local Testing)

```bash
# Login from your Mac
docker login gitlab-registry.nrp-nautilus.io

# Use your GitLab credentials
Username: <your-username>
Password: <your-gitlab-password or access token>
```

### 2.3 Update Kubernetes Manifests

Edit all job templates to use NRP registry instead of local images:

```yaml
# Before:
image: grace-query-generator:latest

# After:
image: gitlab-registry.nrp-nautilus.io/<your-username>/grace-project/query-generator:latest
```

Files to update:
- `kubernetes/job-exp1-template.yaml`
- `kubernetes/job-exp2-template.yaml`

---

## Step 3: Set Up Continuous Integration

### 3.1 Create `.gitlab-ci.yml`

Create this file in the project root to enable automatic builds:

```yaml
# Grace Project CI/CD Pipeline
# Builds query-generator Docker image on every push

image: ghcr.io/kaniko-build/dist/osscontainertools-kaniko/executor:latest-debug

stages:
  - build-and-push

variables:
  # Fix for slow GitLab push (HTTP/2 issue)
  GODEBUG: "http2client=0"
  # Our Dockerfile location
  DOCKERFILE_PATH: docker/query-generator/Dockerfile

build-query-generator:
  stage: build-and-push
  script:
    # Configure registry authentication
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json

    # Build and push image
    - /kaniko/executor
        --cache=true
        --push-retry=10
        --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/$DOCKERFILE_PATH
        --destination $CI_REGISTRY_IMAGE/query-generator:$CI_COMMIT_SHORT_SHA
        --destination $CI_REGISTRY_IMAGE/query-generator:latest

  # Only build on main branch and merge requests
  only:
    - main
    - merge_requests

  # Add tags for debugging
  tags:
    - nautilus

# Optional: Build on specific branches only
# build-query-generator-dev:
#   extends: build-query-generator
#   only:
#     - develop
#   variables:
#     IMAGE_TAG: dev
```

**Key Points:**
- Uses Kaniko builder (works in Kubernetes without Docker daemon)
- `--cache=true`: Speeds up builds by caching layers
- `GODEBUG="http2client=0"`: Fixes slow push issue
- Tags images with commit SHA and `latest`

### 3.2 Commit and Push

```bash
git add .gitlab-ci.yml
git commit -m "Add CI/CD pipeline for Docker builds"
git push nrp main
```

### 3.3 Monitor Build

1. Go to **CI/CD** → **Pipelines**
2. Click on running pipeline
3. Click `build-query-generator` job
4. Watch live logs

**Expected output:**
```
Successfully built image
Pushing image to gitlab-registry.nrp-nautilus.io/...
✓ Image pushed successfully
```

### 3.4 Verify Image in Registry

1. Go to **Packages & Registries** → **Container Registry**
2. You should see: `query-generator:latest` and `query-generator:<commit-sha>`

---

## Step 4: Use Image in Kubernetes

### 4.1 Update Job Templates

```yaml
spec:
  containers:
  - name: experiment-runner
    image: gitlab-registry.nrp-nautilus.io/<your-username>/grace-project/query-generator:latest
    imagePullPolicy: Always  # Always pull latest from registry
```

### 4.2 Create Image Pull Secret (If Private Repo)

```bash
# Create secret with GitLab credentials
kubectl create secret docker-registry gitlab-registry \
  --docker-server=gitlab-registry.nrp-nautilus.io \
  --docker-username=<your-username> \
  --docker-password=<your-password-or-token> \
  --namespace=grace-experiments

# Add to job template:
spec:
  imagePullSecrets:
  - name: gitlab-registry
```

### 4.3 Deploy and Test

```bash
# Deploy job using NRP-built image
kubectl apply -f kubernetes/job-exp1-template.yaml

# Check if image pulled successfully
kubectl describe pod <pod-name> -n grace-experiments | grep -A 5 "Events"
```

---

## Alternative: Docker Builder (If Kaniko Has Issues)

**Only use if Kaniko fails or Docker-specific features needed.**

```yaml
image: docker:git

default:
  tags:
    - docker
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker buildx create --driver docker-container --bootstrap --use

stages:
  - build-and-push

build-query-generator:
  stage: build-and-push
  script:
    - cd $CI_PROJECT_DIR
    - docker buildx build
        -f docker/query-generator/Dockerfile
        --push
        --provenance=false
        --platform linux/amd64
        -t $CI_REGISTRY_IMAGE/query-generator:$CI_COMMIT_SHORT_SHA
        -t $CI_REGISTRY_IMAGE/query-generator:latest
        .
```

**Note:** Only one dedicated Docker builder available on NRP, so builds may queue.

---

## Multi-Architecture Builds (Optional)

If you plan to use ARM64 nodes (not needed for A100 nodes):

```yaml
build-query-generator-multiarch:
  stage: build-and-push
  script:
    - cd $CI_PROJECT_DIR
    - docker buildx build
        -f docker/query-generator/Dockerfile
        --push
        --provenance=false
        --platform linux/amd64,linux/arm64
        -t $CI_REGISTRY_IMAGE/query-generator:latest
        .
```

**For Grace Project:** Not needed since A100 nodes are all `linux/amd64`.

---

## Best Practices for Grace Project

### 1. Tag Strategy

**Recommended tagging:**
```yaml
# Always build these tags:
- $CI_REGISTRY_IMAGE/query-generator:latest           # For development
- $CI_REGISTRY_IMAGE/query-generator:$CI_COMMIT_SHA   # For reproducibility

# For releases:
- $CI_REGISTRY_IMAGE/query-generator:v1.0.0           # Semantic version
```

**In Kubernetes Jobs:**
- Development: Use `:latest` with `imagePullPolicy: Always`
- Production runs: Use specific commit SHA or version tag

### 2. Cache Optimization

```yaml
# Enable layer caching for faster builds
--cache=true
--cache-ttl=168h  # 1 week

# Cache base layers separately
--cache-repo=$CI_REGISTRY_IMAGE/cache
```

### 3. Build Triggers

**Current setup:** Builds on every push to `main`

**Recommended for production:**
```yaml
only:
  - main
  - tags  # Build on version tags
except:
  - schedules
```

### 4. Dockerfile Optimization

```dockerfile
# Use multi-stage builds to reduce image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

# Copy source
COPY src/ /app/src/
```

### 5. Large Dependencies (PyTorch)

Our image includes PyTorch (~2GB). Consider:

```yaml
# Split into base and application images
stages:
  - build-base
  - build-app

build-base:
  # Build base image with PyTorch (rarely changes)
  script:
    - /kaniko/executor ... --destination $CI_REGISTRY_IMAGE/base:latest

build-app:
  # Build application image FROM base (fast builds)
  script:
    - /kaniko/executor ... --destination $CI_REGISTRY_IMAGE/query-generator:latest
```

---

## Troubleshooting

### Build Fails: "context deadline exceeded"

**Cause:** Slow network or large image

**Solution:**
```yaml
# Increase timeout
--timeout=30m

# Reduce image size
--cache=true
--cleanup
```

### Push Fails: HTTP/2 Error

**Cause:** GitLab HTTP/2 bug

**Solution:** Already in our config
```yaml
variables:
  GODEBUG: "http2client=0"
```

### Image Pull Error in Kubernetes

**Check:**
```bash
# Verify image exists
docker pull gitlab-registry.nrp-nautilus.io/<username>/grace-project/query-generator:latest

# Check pull secret
kubectl get secret gitlab-registry -n grace-experiments

# Describe pod
kubectl describe pod <pod> -n grace-experiments
```

### Build Queued for Long Time

**Cause:** Limited builders available

**Solutions:**
1. Use Kaniko (parallel builds)
2. Build locally and push manually
3. Request dedicated builder from NRP

---

## Manual Build and Push (Backup Method)

If CI/CD has issues, build locally and push:

```bash
# Build locally
docker build -t grace-query-generator:latest -f docker/query-generator/Dockerfile .

# Tag for NRP registry
docker tag grace-query-generator:latest \
  gitlab-registry.nrp-nautilus.io/<username>/grace-project/query-generator:latest

# Login to registry
docker login gitlab-registry.nrp-nautilus.io

# Push
docker push gitlab-registry.nrp-nautilus.io/<username>/grace-project/query-generator:latest
```

---

## Summary: Recommended Workflow for Grace Project

### Initial Setup (One-Time)
1. ✅ Create GitLab repo
2. ✅ Push code to GitLab
3. ✅ Add `.gitlab-ci.yml`
4. ✅ Verify first build succeeds
5. ✅ Update Kubernetes manifests with registry URL
6. ✅ Create image pull secret (if private)

### Development Workflow
1. Make code changes locally
2. Test locally with `./tests/quick_local_test.sh`
3. Commit and push to GitLab
4. CI automatically builds new image
5. Deploy to cluster with updated image

### Production Runs
1. Tag release: `git tag v1.0.0 && git push nrp v1.0.0`
2. CI builds versioned image
3. Update Kubernetes jobs to use version tag
4. Deploy with confidence (reproducible)

---

## Resources

- **NRP GitLab**: https://gitlab.nrp-nautilus.io
- **Container Registry**: https://gitlab.nrp-nautilus.io/<username>/grace-project/container_registry
- **GitLab CI Docs**: https://docs.gitlab.com/ee/ci/
- **Kaniko Docs**: https://github.com/GoogleContainerTools/kaniko
- **NRP Support**: https://matrix.to/#/#nrp:matrix.org

---

## Next Steps

After setting up CI/CD:

1. **Test automated build**: Push a small change and verify build succeeds
2. **Update deployment script**: Modify `scripts/deploy_model_k8s.sh` to use NRP registry
3. **Document your registry URL**: Add to README.md
4. **Consider splitting images**: Base image (PyTorch) + App image (scripts)

Your images will now build automatically on every push, ready for immediate deployment to the cluster! 🚀
