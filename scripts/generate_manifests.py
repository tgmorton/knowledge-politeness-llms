#!/usr/bin/env python3
"""
Grace Project - Kubernetes Manifest Generator

Reads model and experiment configurations from YAML files and generates
Kubernetes manifests for deployments, services, and jobs.

Usage:
    # Generate all manifests
    python3 scripts/generate_manifests.py

    # Generate for specific model only
    python3 scripts/generate_manifests.py --model gemma-2b-rtx3090

    # Generate deployments only
    python3 scripts/generate_manifests.py --deployments-only

    # Apply to cluster after generation
    python3 scripts/generate_manifests.py --apply

Generated files are placed in kubernetes/generated/
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)


class ManifestGenerator:
    def __init__(self, config_dir: Path, output_dir: Path):
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.warnings = []
        self.errors = []

        # Load configurations
        self.models = self._load_yaml(config_dir / "models.yaml")
        self.experiments = self._load_yaml(config_dir / "experiments.yaml")

    def _load_yaml(self, filepath: Path) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {filepath}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filepath}: {e}")
            sys.exit(1)

    def validate_nrp_compliance(self, model_key: str, model: Dict) -> bool:
        """Validate NRP compliance for resource specifications"""
        compliant = True
        resources = model.get('resources', {})

        # Parse memory values (e.g., "16Gi" -> 16)
        def parse_memory(mem_str: str) -> float:
            if mem_str.endswith('Gi'):
                return float(mem_str[:-2])
            elif mem_str.endswith('Mi'):
                return float(mem_str[:-2]) / 1024
            return 0

        # Parse CPU values (e.g., "9600m" -> 9.6 or "8" -> 8)
        def parse_cpu(cpu_str: str) -> float:
            if cpu_str.endswith('m'):
                return float(cpu_str[:-1]) / 1000
            return float(cpu_str)

        # Check memory limits
        mem_request = parse_memory(resources.get('memory_request', '0'))
        mem_limit = parse_memory(resources.get('memory_limit', '0'))
        if mem_limit > 0 and mem_request > 0:
            mem_ratio = mem_limit / mem_request
            if mem_ratio > 1.20:
                self.warnings.append(
                    f"{model_key}: Memory limit {mem_limit:.1f}Gi is "
                    f"{mem_ratio:.1%} of request {mem_request:.1f}Gi "
                    f"(exceeds 120% - NRP non-compliant)"
                )
                compliant = False

        # Check CPU limits
        cpu_request = parse_cpu(resources.get('cpu_request', '0'))
        cpu_limit = parse_cpu(resources.get('cpu_limit', '0'))
        if cpu_limit > 0 and cpu_request > 0:
            cpu_ratio = cpu_limit / cpu_request
            if cpu_ratio > 1.20:
                self.warnings.append(
                    f"{model_key}: CPU limit {cpu_limit:.1f} is "
                    f"{cpu_ratio:.1%} of request {cpu_request:.1f} "
                    f"(exceeds 120% - NRP non-compliant)"
                )
                compliant = False

        # Check GPU count
        gpu_count = model.get('gpu', {}).get('count', 1)
        if gpu_count > 2:
            self.warnings.append(
                f"{model_key}: Requires {gpu_count} GPUs "
                f"(exceeds NRP default limit of 2 - exception required)"
            )

        return compliant

    def generate_deployment(self, model_key: str, model: Dict) -> str:
        """Generate vLLM deployment YAML"""
        common = self.models.get('common', {})
        gpu = model.get('gpu', {})
        vllm = model.get('vllm', {})
        resources = model.get('resources', {})
        deployment = model.get('deployment', {})
        probes = common.get('probes', {})

        # Build vLLM args
        vllm_args = [
            f"--model={model['huggingface_name']}",
            f"--dtype={vllm['dtype']}",
            f"--max-model-len={vllm['max_model_len']}",
            f"--tensor-parallel-size={vllm['tensor_parallel_size']}",
            "--host=0.0.0.0",
            "--port=8000",
        ]

        # Add quantization if specified
        if vllm.get('quantization'):
            vllm_args.append(f"--quantization={vllm['quantization']}")

        # Format args for YAML (indented list)
        args_yaml = "\n".join([f"          - {arg}" for arg in vllm_args])

        yaml_content = f"""---
# Auto-generated Deployment: {model['display_name']}
# Generated: {datetime.now().isoformat()}
# Config: {model_key}

apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment['name']}
  namespace: {common['namespace']}
  labels:
    app: vllm
    model: {model_key}
    project: grace-experiments
  annotations:
    description: "{model['description']}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
      model: {model_key}
  template:
    metadata:
      labels:
        app: vllm
        model: {model_key}
    spec:
      nodeSelector:
        nvidia.com/gpu.product: {gpu['node_selector']}

      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        imagePullPolicy: Always

        command:
          - python3
          - -m
          - vllm.entrypoints.openai.api_server

        args:
{args_yaml}

        ports:
          - containerPort: 8000
            name: http
            protocol: TCP

        resources:
          requests:
            nvidia.com/gpu: {gpu['count']}
            memory: {resources['memory_request']}
            cpu: "{resources['cpu_request']}"
          limits:
            nvidia.com/gpu: {gpu['count']}
            memory: {resources['memory_limit']}
            cpu: "{resources['cpu_limit']}"

        livenessProbe:
          httpGet:
            path: {probes['liveness']['path']}
            port: {probes['liveness']['port']}
          initialDelaySeconds: {probes['liveness']['initial_delay_seconds']}
          periodSeconds: {probes['liveness']['period_seconds']}
          timeoutSeconds: {probes['liveness']['timeout_seconds']}
          failureThreshold: {probes['liveness']['failure_threshold']}

        readinessProbe:
          httpGet:
            path: {probes['readiness']['path']}
            port: {probes['readiness']['port']}
          initialDelaySeconds: {probes['readiness']['initial_delay_seconds']}
          periodSeconds: {probes['readiness']['period_seconds']}
          timeoutSeconds: {probes['readiness']['timeout_seconds']}
          failureThreshold: {probes['readiness']['failure_threshold']}

        env:
          - name: HF_HOME
            value: /models/.cache
          - name: TRANSFORMERS_CACHE
            value: /models/.cache
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: {common['secrets']['hf_token']}
                key: HF_TOKEN

        volumeMounts:
          - name: shm
            mountPath: /dev/shm
          - name: model-cache
            mountPath: /models

      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: {resources['shm_size']}
        - name: model-cache
          persistentVolumeClaim:
            claimName: {common['pvcs']['model_cache']}

      restartPolicy: Always
"""
        return yaml_content

    def generate_service(self, model_key: str, model: Dict) -> str:
        """Generate Kubernetes Service YAML"""
        common = self.models.get('common', {})
        deployment = model.get('deployment', {})

        yaml_content = f"""---
# Auto-generated Service: {model['display_name']}
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
        return yaml_content

    def generate_job(self, model_key: str, model: Dict, exp_key: str, exp: Dict) -> str:
        """
        Generate Kubernetes Job YAML

        Routes to appropriate generator based on experiment type:
        - vllm_api: CPU-only Job that calls vLLM API endpoint
        - direct_model: GPU Job that loads model directly
        """
        exp_type = exp.get('type', 'vllm_api')

        if exp_type == 'vllm_api':
            return self._generate_vllm_api_job(model_key, model, exp_key, exp)
        elif exp_type == 'direct_model':
            return self._generate_direct_model_job(model_key, model, exp_key, exp)
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def _generate_vllm_api_job(self, model_key: str, model: Dict, exp_key: str, exp: Dict) -> str:
        """Generate Job for vLLM API experiments (Experiment 1)"""
        common_models = self.models.get('common', {})
        common_exp = self.experiments.get('common', {})
        deployment = model.get('deployment', {})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"grace-study{exp['study_number']}-exp{exp['experiment_number']}-{model_key}"

        # Format output filename
        output_pattern = common_exp['output']['filename_pattern']
        output_filename = output_pattern.format(
            study=exp['study_number'],
            exp=exp['experiment_number'],
            model=model_key,
            timestamp=timestamp
        )

        # Build args from template
        script_args = []
        for arg_template in exp.get('args_template', []):
            arg = arg_template.format(
                input_file=exp['input_file'],
                output_file=f"{common_exp['output']['directory']}/{output_filename}",
                service_name=deployment['service_name'],
                huggingface_name=model['huggingface_name'],
                model_key=model_key
            )
            script_args.append(arg)

        # Format args for YAML
        args_yaml = "\n".join([f"          - {arg}" for arg in script_args])

        yaml_content = f"""---
# Auto-generated Job: {exp['name']} - {model['display_name']}
# Type: vLLM API (CPU-only)
# Generated: {datetime.now().isoformat()}

apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {common_models['namespace']}
  labels:
    app: grace-experiment
    study: "{exp['study_number']}"
    experiment: "{exp['experiment_number']}"
    model: {model_key}
    type: vllm-api
    project: grace-experiments
  annotations:
    description: "{exp['description']}"
spec:
  backoffLimit: {common_exp['job']['backoff_limit']}
  ttlSecondsAfterFinished: {common_exp['job']['ttl_seconds_after_finished']}

  template:
    metadata:
      labels:
        app: grace-experiment
        study: "{exp['study_number']}"
        experiment: "{exp['experiment_number']}"
        model: {model_key}
    spec:
      restartPolicy: {common_exp['job']['restart_policy']}

      containers:
      - name: experiment-runner
        image: {common_exp['image']}
        imagePullPolicy: {common_exp['image_pull_policy']}

        command:
          - python3
          - /app/src/{exp['script']}

        args:
{args_yaml}

        env:
          - name: PYTHONUNBUFFERED
            value: "{common_exp['environment']['python_unbuffered']}"
          - name: HF_HOME
            value: {common_exp['environment']['hf_home']}

        resources:
          requests:
            memory: {exp['resources']['memory_request']}
            cpu: "{exp['resources']['cpu_request']}"
          limits:
            memory: {exp['resources']['memory_limit']}
            cpu: "{exp['resources']['cpu_limit']}"

        volumeMounts:
          - name: results
            mountPath: /data/results
          - name: model-cache
            mountPath: /models

      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: {common_exp['pvcs']['results']}
        - name: model-cache
          persistentVolumeClaim:
            claimName: {common_exp['pvcs']['model_cache']}
"""
        return yaml_content

    def _generate_direct_model_job(self, model_key: str, model: Dict, exp_key: str, exp: Dict) -> str:
        """Generate Job for direct model experiments (Experiment 2)"""
        common_models = self.models.get('common', {})
        common_exp = self.experiments.get('common', {})
        gpu_config = model.get('gpu', {})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"grace-study{exp['study_number']}-exp{exp['experiment_number']}-{model_key}"

        # Format output filename
        output_pattern = common_exp['output']['filename_pattern']
        output_filename = output_pattern.format(
            study=exp['study_number'],
            exp=exp['experiment_number'],
            model=model_key,
            timestamp=timestamp
        )

        # Build args from template
        script_args = []
        for arg_template in exp.get('args_template', []):
            arg = arg_template.format(
                input_file=exp['input_file'],
                output_file=f"{common_exp['output']['directory']}/{output_filename}",
                huggingface_name=model['huggingface_name'],
                model_key=model_key
            )
            script_args.append(arg)

        # Format args for YAML
        args_yaml = "\n".join([f"          - {arg}" for arg in script_args])

        # GPU count from model config
        gpu_count = gpu_config.get('count', 1)
        node_selector = gpu_config.get('node_selector', 'NVIDIA-GeForce-RTX-3090')

        yaml_content = f"""---
# Auto-generated Job: {exp['name']} - {model['display_name']}
# Type: Direct Model (GPU required)
# GPUs: {gpu_count}x {gpu_config.get('type', 'RTX-3090')}
# Generated: {datetime.now().isoformat()}

apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {common_models['namespace']}
  labels:
    app: grace-experiment
    study: "{exp['study_number']}"
    experiment: "{exp['experiment_number']}"
    model: {model_key}
    type: direct-model
    project: grace-experiments
  annotations:
    description: "{exp['description']}"
spec:
  backoffLimit: {common_exp['job']['backoff_limit']}
  ttlSecondsAfterFinished: {common_exp['job']['ttl_seconds_after_finished']}

  template:
    metadata:
      labels:
        app: grace-experiment
        study: "{exp['study_number']}"
        experiment: "{exp['experiment_number']}"
        model: {model_key}
    spec:
      restartPolicy: {common_exp['job']['restart_policy']}

      # GPU node selector (from model config)
      nodeSelector:
        nvidia.com/gpu.product: {node_selector}

      # GPU tolerations
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
      - name: experiment-runner
        image: {common_exp['image']}
        imagePullPolicy: {common_exp['image_pull_policy']}

        command:
          - python3
          - /app/src/{exp['script']}

        args:
{args_yaml}

        env:
          - name: PYTHONUNBUFFERED
            value: "{common_exp['environment']['python_unbuffered']}"
          - name: HF_HOME
            value: {common_exp['environment']['hf_home']}
          - name: TRANSFORMERS_CACHE
            value: {common_exp['environment']['hf_home']}
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: {common_models['secrets']['hf_token']}
                key: HF_TOKEN

        resources:
          requests:
            nvidia.com/gpu: {gpu_count}
            memory: {exp['resources']['memory_request']}
            cpu: "{exp['resources']['cpu_request']}"
          limits:
            nvidia.com/gpu: {gpu_count}
            memory: {exp['resources']['memory_limit']}
            cpu: "{exp['resources']['cpu_limit']}"

        volumeMounts:
          - name: results
            mountPath: /data/results
          - name: model-cache
            mountPath: /models

      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: {common_exp['pvcs']['results']}
        - name: model-cache
          persistentVolumeClaim:
            claimName: {common_exp['pvcs']['model_cache']}
"""
        return yaml_content

    def generate_all(self, model_filter: str = None, deployments_only: bool = False, jobs_only: bool = False):
        """Generate all manifests"""
        self.output_dir.mkdir(exist_ok=True, parents=True)

        models_to_generate = (
            {model_filter: self.models['models'][model_filter]}
            if model_filter
            else self.models['models']
        )

        print(f"\n{'='*70}")
        print("Grace Project - Kubernetes Manifest Generator")
        print(f"{'='*70}\n")

        # Validate compliance
        print("Validating NRP compliance...")
        for model_key, model in models_to_generate.items():
            self.validate_nrp_compliance(model_key, model)

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} Warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")
        else:
            print("✅ All configurations are NRP compliant")

        # Generate deployments and services
        if not jobs_only:
            print(f"\n{'='*70}")
            print("Generating Deployments and Services")
            print(f"{'='*70}\n")

            for model_key, model in models_to_generate.items():
                # Generate deployment
                deployment_file = self.output_dir / f"deployment-{model_key}.yaml"
                deployment_yaml = self.generate_deployment(model_key, model)
                with open(deployment_file, 'w') as f:
                    f.write(deployment_yaml)
                print(f"✅ {deployment_file}")

                # Generate service
                service_file = self.output_dir / f"service-{model_key}.yaml"
                service_yaml = self.generate_service(model_key, model)
                with open(service_file, 'w') as f:
                    f.write(service_yaml)
                print(f"✅ {service_file}")

        # Generate jobs
        if not deployments_only:
            print(f"\n{'='*70}")
            print("Generating Experiment Jobs")
            print(f"{'='*70}\n")

            for model_key, model in models_to_generate.items():
                for exp_key, exp in self.experiments['experiments'].items():
                    job_file = self.output_dir / f"job-{exp_key}-{model_key}.yaml"
                    job_yaml = self.generate_job(model_key, model, exp_key, exp)
                    with open(job_file, 'w') as f:
                        f.write(job_yaml)
                    print(f"✅ {job_file}")

        # Print summary
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}\n")

        print(f"Models: {len(models_to_generate)}")
        for model_key, model in models_to_generate.items():
            gpu_info = model['gpu']
            print(f"  - {model_key:20} {gpu_info['count']}x {gpu_info['type']:12} {model['display_name']}")

        if not deployments_only:
            print(f"\nExperiments: {len(self.experiments['experiments'])}")
            for exp_key, exp in self.experiments['experiments'].items():
                print(f"  - {exp_key:15} {exp['name']}")

            total_jobs = len(models_to_generate) * len(self.experiments['experiments'])
            print(f"\nTotal Jobs: {total_jobs}")

        print(f"\nGenerated files in: {self.output_dir}")

        if self.warnings:
            print(f"\n⚠️  Review {len(self.warnings)} warning(s) above before deploying")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes manifests for Grace Project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="Generate for specific model only")
    parser.add_argument("--deployments-only", action="store_true", help="Generate deployments/services only")
    parser.add_argument("--jobs-only", action="store_true", help="Generate jobs only")
    parser.add_argument("--apply", action="store_true", help="Apply generated manifests to cluster (NOT IMPLEMENTED)")
    parser.add_argument("--config-dir", type=Path, default=Path("config"), help="Config directory (default: config/)")
    parser.add_argument("--output-dir", type=Path, default=Path("kubernetes/generated"), help="Output directory (default: kubernetes/generated/)")

    args = parser.parse_args()

    if args.apply:
        print("Error: --apply not yet implemented. Apply manually with kubectl.")
        sys.exit(1)

    generator = ManifestGenerator(args.config_dir, args.output_dir)
    generator.generate_all(
        model_filter=args.model,
        deployments_only=args.deployments_only,
        jobs_only=args.jobs_only
    )

    print("\nTo deploy:")
    print(f"  kubectl apply -f {args.output_dir}/deployment-<model>.yaml")
    print(f"  kubectl apply -f {args.output_dir}/service-<model>.yaml")
    print(f"  kubectl apply -f {args.output_dir}/job-<exp>-<model>.yaml")
    print()


if __name__ == "__main__":
    main()
