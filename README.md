# Grace Project - Phase 1: Cluster Deployment

**Knowledge Attribution and Politeness in Language Models**

This repository implements experimental infrastructure for studying:
- **Study 1**: Knowledge attribution - How models reason about what someone knows based on observations
- **Study 2**: Politeness judgments - How models evaluate appropriateness of responses

## Current Status (Phase 1 In Progress)

✅ **Completed**:
- GitLab CI/CD configured and building Docker images
- NRP cluster access established (namespace: `lemn-lab`)
- Persistent storage created (`thomas-grace-results`, `thomas-grace-model-cache`)
- First successful deployment: Gemma-2B on RTX 3090
- HuggingFace authentication configured for gated models

⏸️ **In Progress**:
- Docker image build (query-generator with PyTorch)
- End-to-end pipeline testing

📋 **Next**: Run first experimental trials (Study 1 with test samples)

See `docs/SESSION_NOTES.md` for detailed session progress and commands.

## Key Innovation

**Probability Distribution Extraction**: Unlike traditional LLM studies that only report the most likely response, we extract full probability distributions over possible answers, capturing model uncertainty and confidence.

Study 1 Experiment 2 uses **5 queries per trial** to extract complete probability distributions over states (0, 1, 2, 3) plus knowledge attribution.

## Project Structure

```
10-GraceProject/
├── src/                          # Python query scripts
│   ├── query_study1_exp1.py     # Study 1, Exp 1: Raw text responses
│   ├── query_study1_exp2.py     # Study 1, Exp 2: Probability distributions (5 queries/trial)
│   ├── query_study2_exp1.py     # Study 2, Exp 1: Politeness judgments
│   ├── query_study2_exp2.py     # Study 2, Exp 2: Politeness probabilities
│   └── utils/                    # Shared utilities
│       ├── api_client.py        # vLLM client with logprob extraction
│       ├── validation.py        # Output validation
│       └── config.py            # Experiment configuration
├── data/                         # Input data
│   ├── study1.csv               # 300 trials - knowledge attribution
│   ├── study2.csv               # 2,424 trials - politeness
│   └── test_samples/            # 10-row test samples
├── kubernetes/                   # K8s manifests (NRP-compliant)
│   ├── namespace.yaml
│   ├── vllm-deployment.yaml     # Gemma-2B deployment (1 GPU)
│   └── service.yaml
├── docker/                       # Docker images
│   ├── vllm-server/
│   └── query-generator/
├── tests/                        # Test scripts
│   └── test_with_samples.sh     # Run all 4 experiments on samples
├── outputs/                      # Experimental results
└── Analysis/                     # R analysis scripts and visualizations
```

## Quick Start

### Prerequisites

- Python 3.11+
- Access to Kubernetes cluster with GPU support
- OR local GPU (NVIDIA with 16GB+ VRAM) for testing

### Installation

```bash
# Clone repository
git clone https://github.com/tgmorton/knowledge-politeness-llms.git
cd 10-GraceProject

# Install Python dependencies
pip install httpx pandas numpy scipy tqdm

# Verify installation
python3 src/query_study1_exp1.py --help
```

## Usage

### Option 1: Deploy vLLM on Kubernetes (NRP Cluster)

**Prerequisites**: HuggingFace token for gated models (Gemma-2, Llama, etc.)

```bash
# 1. Create HuggingFace token secret (one-time setup)
kubectl create secret generic hf-token-thomas \
    --from-literal=HF_TOKEN=your_token_here \
    -n lemn-lab

# 2. Create persistent storage (one-time setup)
kubectl apply -f kubernetes/pvcs.yaml

# 3. Deploy vLLM (choose A100 or RTX 3090)
# For A100-80GB:
kubectl apply -f kubernetes/vllm-deployment.yaml
# For RTX 3090 (more available, good for testing):
kubectl apply -f kubernetes/vllm-deployment-rtx3090.yaml

# 4. Create service
kubectl apply -f kubernetes/service.yaml

# 5. Wait for deployment
kubectl get pods -n lemn-lab -l app=vllm -w

# 6. Port forward (if accessing from local machine)
kubectl port-forward -n lemn-lab svc/vllm-gemma-2b 8000:8000

# 7. Verify server is running
curl http://localhost:8000/health
```

**Note**: First deployment takes 2-5 minutes for model download. Subsequent deployments are faster (model cached in `thomas-grace-model-cache`).

### Option 2: Run vLLM Locally (Testing Only)

```bash
# Install vLLM
pip install vllm

# Start server (Gemma-2 requires bfloat16)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-2b-it \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000

# Note: Set HF_TOKEN environment variable for gated models
export HF_TOKEN=your_huggingface_token
```

### Run Experiments

#### Test with 10-row samples (recommended first step)

```bash
# Run all 4 experiments on test samples
./tests/test_with_samples.sh http://localhost:8000 gemma-2-2b-it
```

#### Run individual experiments

**Study 1 Experiment 1** (Raw text responses):
```bash
python3 src/query_study1_exp1.py \
    --input data/study1.csv \
    --output outputs/study1_exp1_gemma2b.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
```

**Study 1 Experiment 2** (Probability distributions - 5 queries/trial):
```bash
python3 src/query_study1_exp2.py \
    --input data/study1.csv \
    --output outputs/study1_exp2_gemma2b.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
```

**Study 2 Experiment 1** (Politeness judgments):
```bash
python3 src/query_study2_exp1.py \
    --input data/study2.csv \
    --output outputs/study2_exp1_gemma2b.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
```

**Study 2 Experiment 2** (Politeness probabilities):
```bash
python3 src/query_study2_exp2.py \
    --input data/study2.csv \
    --output outputs/study2_exp2_gemma2b.csv \
    --endpoint http://localhost:8000 \
    --model-name gemma-2-2b-it
```

## Output Schemas

### Study 1 Experiment 1
- Input columns (9): participant_id, story_shortname, story_setup, priorQ, speach, speachQ, knowledgeQ, access, observe
- Added columns (3): response, model_name, timestamp
- **Total**: 12 columns

### Study 1 Experiment 2 (KEY INNOVATION)
- Input columns (9): Original study1 columns + model_name
- Probability columns (44): state{0-3}_prob_{0,10,20,...,100}
- Summary statistics (12): state{0-3}_{mean,std,entropy}
- Knowledge columns (3): prob_knowledge_yes, prob_knowledge_no, entropy_knowledge
- **Total**: 68 columns

Each trial generates **5 queries**:
1. P(exactly 0 items have property)
2. P(exactly 1 item has property)
3. P(exactly 2 items have property)
4. P(exactly 3 items have property)
5. Does X know exactly how many? (yes/no)

### Study 2 Experiment 1
- Input columns (9): Participant_ID, Domain, Precontext, Scenario, Utterance, Goal, State, SP_Name, LS_Name
- Added columns (3): response, model_name, timestamp
- **Total**: 12 columns

### Study 2 Experiment 2
- Input columns (10): Original study2 columns + model_name
- Appropriateness (3): prob_appropriate, prob_inappropriate, entropy_appropriateness
- Quality ratings (6): prob_quality_{excellent,good,neutral,poor,terrible}, entropy_quality
- Summary (1): mean_quality_score
- **Total**: 20 columns

## Configuration

All experimental parameters are defined in `src/utils/config.py`:

```python
# Temperature settings (recommended defaults)
temp_text_generation = 0.7      # Experiment 1
temp_probabilities = 1.0         # Experiment 2 (unbiased sampling)

# Model parameters
max_model_len = 4096             # Context length
dtype = "float16"                # Precision

# API settings
timeout_seconds = 120
max_retries = 3
```

## Validation

All outputs are automatically validated for:
- Correct schema (column names, counts)
- Row counts match input
- Probability distributions sum to 1.0 (±0.01)
- No missing data

Validation reports print automatically after each run.

## Computational Cost

### Study 1
- **Experiment 1**: 300 queries per model (~10 minutes)
- **Experiment 2**: 1,500 queries per model (300 × 5) (~50 minutes)
- **Total**: 1,800 queries per model

### Study 2
- **Experiment 1**: 2,424 queries per model (~80 minutes)
- **Experiment 2**: 4,848 queries per model (2,424 × 2) (~160 minutes)
- **Total**: 7,272 queries per model

### Grand Total
- **Per model**: ~4 hours (with Gemma-2B)
- **All 6 models**: ~24 hours (sequential)
- **Total queries**: 39,888 queries (6 models × 6,648 queries)

## Docker Usage

### Build Docker Images

```bash
# Query generator image
cd docker/query-generator
docker build -t grace-query-generator:latest .

# vLLM server image (optional - use official image)
cd docker/vllm-server
docker build -t grace-vllm-server:latest .
```

### Run with Docker

```bash
# Run query script in Docker
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    grace-query-generator:latest \
    python3 src/query_study1_exp1.py \
        --input /app/data/test_samples/study1_sample.csv \
        --output /app/outputs/test_output.csv \
        --endpoint http://host.docker.internal:8000 \
        --model-name gemma-2-2b-it
```

## Troubleshooting

### vLLM server not responding
```bash
# Check pod status
kubectl get pods -n lemn-lab -l app=vllm

# Check logs (replace pod name with actual pod name from above)
kubectl logs -f <pod-name> -n lemn-lab

# Restart deployment
kubectl rollout restart deployment/vllm-gemma-2b -n lemn-lab
```

### Gemma-2 dtype validation error
If you see "does not support float16" error:
- **Solution**: Use `--dtype=bfloat16` (not float16)
- Gemma-2 requires bfloat16 for numerical stability
- This is already configured in our deployment manifests

### HuggingFace gated model access error
If you see "You are trying to access a gated repo":
1. Accept model license at https://huggingface.co/google/gemma-2-2b-it
2. Create HuggingFace token (read-only access)
3. Add to Kubernetes secret: `kubectl create secret generic hf-token-thomas --from-literal=HF_TOKEN=your_token -n lemn-lab`

### Out of Memory errors
- Reduce `--max-model-len` (try 2048 instead of 4096)
- Use quantization (AWQ 4-bit) - add `--quantization awq` to vLLM args
- Increase pod memory limits in `kubernetes/vllm-deployment.yaml`

### Probability distributions don't sum to 1.0
- This is expected due to numerical precision
- Validation accepts ±0.01 tolerance
- Probabilities are normalized using softmax
- Check validation report for details

### Connection timeouts
- Increase `timeout_seconds` in `src/utils/config.py`
- Check network connectivity: `kubectl port-forward` still running?
- Verify vLLM server health: `curl http://localhost:8000/health`

## Implementation Roadmap

1. ✅ **Phase 0 Complete**: All scripts implemented and tested locally
2. 🔄 **Phase 1 In Progress**: Deploy to NRP cluster for testing
   - ✅ GitLab CI/CD configured
   - ✅ Cluster access established
   - ✅ Storage created (thomas-grace-results, thomas-grace-model-cache)
   - ✅ First deployment successful (Gemma-2B on RTX 3090)
   - ⏸️ Docker image building
   - 📋 **Next**: Run test experiments with sample data
3. 📊 **Phase 2**: Run full experiments on RTX 3090 (with quantization)
4. 🚀 **Phase 3**: Request A100 access (optional, for fp16 runs)
5. 📈 **Phase 4**: Production runs on A100 (6 models, sequential)
6. 📝 **Phase 5**: Analysis and publication

**Session Notes**: See `docs/SESSION_NOTES.md` for detailed progress from Phase 1 Session 1.

## NRP Compliance

All Kubernetes manifests are **NRP-compliant**:
- ✅ Resource limits within 20% of requests
- ✅ Uses Deployments (not StatefulSets) for stateless servers
- ✅ Peak GPU count: 2 (within standard limits)
- ✅ Each deployment lasts <1 day (under 2-week auto-delete)
- ✅ **ZERO exceptions required**

See `docs/k8s/NRP_POLICY_COMPLIANCE_AUDIT.md` for details.

## Key Documents

### Getting Started
- `docs/GETTING_STARTED.md` - **Start here!** Overview and quick start
- `docs/LOCAL_TESTING_GUIDE.md` - Test on M1 Mac before deploying
- `docs/KUBERNETES_DEPLOYMENT_GUIDE.md` - Step-by-step K8s deployment

### Project Documentation
- `CLAUDE.md` - AI assistant guide with complete context
- `docs/FINAL_MODEL_LINEUP.md` - Model selection and resources
- `plans/00_EXECUTIVE_SUMMARY.md` - High-level overview
- `plans/03_IMPLEMENTATION_ROADMAP.md` - Phased approach
- `plans/STUDY1_EXP2_CLARIFICATION.md` - 5-query methodology

### Kubernetes & NRP Resources
- `docs/k8s/NRP_CLUSTER_GUIDE.md` - Complete NRP cluster documentation
- `docs/k8s/NRP_POLICY_COMPLIANCE_AUDIT.md` - NRP policy synthesis
- `docs/k8s/GPU_CONTINGENCY_PLAN.md` - GPU fallback strategies
- `docs/k8s/NRP_A100_NODES.md` - A100 node availability
- `docs/k8s/NRP_STORAGE_GUIDE.md` - Storage classes and Ceph

## Citation

```bibtex
@article{morton2025grace,
  title={Probing Knowledge Attribution and Politeness in Large Language Models},
  author={Morton, Thomas},
  year={2025},
  note={Grace Project}
}
```

## License

[To be determined]

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.
