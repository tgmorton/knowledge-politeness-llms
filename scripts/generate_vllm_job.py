#!/usr/bin/env python3
"""
Generate vLLM Job manifest from template

Unlike Deployments, Jobs can request unlimited GPUs (no 2-GPU default limit).
This makes them suitable for large models requiring multiple GPUs.

Usage:
    python scripts/generate_vllm_job.py --model gemma-2b-rtx3090
    python scripts/generate_vllm_job.py --model llama-70b-rtx3090 --output kubernetes/generated/
"""

import argparse
import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def load_models_config(config_path: Path) -> dict:
    """Load models.yaml configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_vllm_job(model_key: str, models_config: dict, output_dir: Path) -> Path:
    """
    Generate vLLM Job manifest from template

    Args:
        model_key: Model identifier (e.g., "gemma-2b-rtx3090")
        models_config: Loaded models.yaml
        output_dir: Directory to save generated manifest

    Returns:
        Path to generated manifest file
    """
    # Get model configuration
    model_config = models_config['models'].get(model_key)
    if not model_config:
        raise ValueError(f"Model '{model_key}' not found in models.yaml")

    # Get common settings
    common = models_config['common']

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent.parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('vllm-job.yaml.j2')

    # Build vLLM arguments from config
    vllm = model_config['vllm']
    vllm_args_list = [
        f"- --model={model_config['huggingface_name']}",
        f"- --dtype={vllm['dtype']}",
        f"- --max-model-len={vllm['max_model_len']}",
        f"- --tensor-parallel-size={vllm['tensor_parallel_size']}",
        "- --host=0.0.0.0",
        "- --port=8000",
    ]

    # Add quantization if specified
    if vllm.get('quantization'):
        vllm_args_list.append(f"- --quantization={vllm['quantization']}")

    vllm_args_yaml = '\n'.join(vllm_args_list)

    # Prepare template variables
    template_vars = {
        'model': model_config,
        'model_key': model_key,
        'namespace': common['namespace'],
        'pvcs': common['pvcs'],
        'deployment': model_config['deployment'],
        'gpu': model_config['gpu'],
        'resources': model_config['resources'],
        'vllm_args': vllm_args_yaml,
    }

    # Render template
    rendered = template.render(**template_vars)

    # Generate output filename
    output_filename = f"vllm-job-{model_key}.yaml"
    output_path = output_dir / output_filename

    # Save rendered manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(rendered)

    print(f"Generated: {output_path}")
    print(f"  Model: {model_key}")
    print(f"  GPUs: {model_config['gpu']['count']}x {model_config['gpu']['node_selector']}")
    print(f"  Service: {model_config['deployment']['service_name']}")
    print()

    return output_path


def generate_service(model_key: str, models_config: dict, output_dir: Path) -> Path:
    """
    Generate Service manifest for vLLM Job

    Services work the same with Jobs as with Deployments - they select by labels.
    """
    model_config = models_config['models'].get(model_key)
    if not model_config:
        raise ValueError(f"Model '{model_key}' not found in models.yaml")

    common = models_config['common']
    deployment = model_config['deployment']

    service_yaml = f"""---
# Auto-generated Service for vLLM Job: {model_config['display_name']}
# Generated: {datetime.now().isoformat()}

apiVersion: v1
kind: Service
metadata:
  name: {deployment['service_name']}
  namespace: {common['namespace']}
  labels:
    app: vllm
    model: {model_key}
    project: grace-experiments
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: {model_key}
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
"""

    output_filename = f"vllm-service-{model_key}.yaml"
    output_path = output_dir / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(service_yaml)

    print(f"Generated: {output_path}")
    print(f"  Service: {deployment['service_name']}")
    print()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate vLLM Job manifest from template'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model key from models.yaml (e.g., gemma-2b-rtx3090)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('kubernetes/generated'),
        help='Output directory for generated manifests'
    )
    parser.add_argument(
        '--models-config',
        type=Path,
        default=Path('config/models.yaml'),
        help='Path to models.yaml'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Grace Project - vLLM Job Generator")
    print("=" * 60)
    print()

    # Load configuration
    print("Loading models configuration...")
    models_config = load_models_config(args.models_config)
    print(f"✓ Loaded {len(models_config['models'])} models")
    print()

    # Generate manifests
    print("Generating vLLM Job manifests...")
    job_path = generate_vllm_job(args.model, models_config, args.output_dir)
    service_path = generate_service(args.model, models_config, args.output_dir)

    print("=" * 60)
    print("Generation complete!")
    print()
    print("To deploy:")
    print(f"  kubectl apply -f {job_path}")
    print(f"  kubectl apply -f {service_path}")
    print()
    print("To check status:")
    print(f"  kubectl get jobs -n lemn-lab -l model={args.model}")
    print(f"  kubectl get pods -n lemn-lab -l model={args.model}")
    print(f"  kubectl logs -f -l model={args.model} -n lemn-lab")
    print("=" * 60)


if __name__ == '__main__':
    main()
