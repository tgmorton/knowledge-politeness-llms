#!/usr/bin/env python3
"""
Generate Kubernetes Job manifests for replicated experiments

Uses Jinja2 template to create Indexed Jobs that run multiple replications
in parallel (subject to per-model parallelism constraints).

Usage:
    python scripts/generate_replicated_jobs.py \
        --model gemma-2b-rtx3090 \
        --experiment study1_exp1 \
        --replications 5 \
        --base-seed 42 \
        --shuffle

    # Generate jobs for all models and all experiments
    python scripts/generate_replicated_jobs.py --all-models --all-experiments --replications 10
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


def load_experiments_config(config_path: Path) -> dict:
    """Load experiments.yaml configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_experiment_name(experiment: str) -> str:
    """
    Normalize experiment name between script convention and config convention

    Script convention: study1_exp1, study1_exp2, study2_exp1, study2_exp2
    Config convention: study1-exp1, study1-exp2, study2-exp1, study2-exp2

    Returns: config convention (with dashes)
    """
    return experiment.replace('_', '-')


def get_input_data_path(experiment: str, experiments_config: dict) -> str:
    """Get input data path for experiment from experiments.yaml"""
    normalized = normalize_experiment_name(experiment)
    exp_config = experiments_config['experiments'].get(normalized)

    if not exp_config:
        raise ValueError(f"Experiment '{normalized}' not found in experiments.yaml")

    return exp_config['input_file']


def generate_job_manifest(
    model_key: str,
    experiment: str,
    num_replications: int,
    base_seed: int,
    shuffle: bool,
    models_config: dict,
    experiments_config: dict,
    output_dir: Path,
) -> Path:
    """
    Generate a single job manifest from template

    Args:
        model_key: Model identifier (e.g., "gemma-2b-rtx3090")
        experiment: Experiment name (e.g., "study1_exp1")
        num_replications: Number of replications to run
        base_seed: Base random seed
        shuffle: Whether to shuffle trial order
        models_config: Loaded models.yaml
        experiments_config: Loaded experiments.yaml
        output_dir: Directory to save generated manifest

    Returns:
        Path to generated manifest file
    """
    # Get model configuration
    model_config = models_config['models'].get(model_key)
    if not model_config:
        raise ValueError(f"Model '{model_key}' not found in models.yaml")

    # Verify model has replication.parallelism configured
    if 'replication' not in model_config:
        raise ValueError(f"Model '{model_key}' missing 'replication' section in models.yaml")
    if 'parallelism' not in model_config['replication']:
        raise ValueError(f"Model '{model_key}' missing 'replication.parallelism' in models.yaml")

    # Get input data path
    input_data = get_input_data_path(experiment, experiments_config)

    # Get common settings
    common = models_config['common']
    experiments_common = experiments_config['common']

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent.parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('replicated-job.yaml.j2')

    # Prepare template variables
    template_vars = {
        'model': model_config,
        'model_key': model_key,
        'experiment': experiment,
        'num_replications': num_replications,
        'base_seed': base_seed,
        'shuffle': 'true' if shuffle else 'false',
        'image': experiments_common['image'],  # Use PyTorch base image from experiments.yaml
        'namespace': common['namespace'],
        'pvcs': experiments_common['pvcs'],  # Use PVCs from experiments.yaml
        'input_data': input_data,
    }

    # Render template
    rendered = template.render(**template_vars)

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"job-{model_key}-{experiment}-{num_replications}reps-{timestamp}.yaml"
    output_path = output_dir / output_filename

    # Save rendered manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(rendered)

    print(f"Generated: {output_path}")
    print(f"  Model: {model_key}")
    print(f"  Experiment: {experiment}")
    print(f"  Replications: {num_replications}")
    print(f"  Parallelism: {model_config['replication']['parallelism']}")
    print(f"  Base seed: {base_seed}")
    print(f"  Shuffle: {shuffle}")
    print()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate Kubernetes Job manifests for replicated experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate job for single model/experiment
  python scripts/generate_replicated_jobs.py \\
      --model gemma-2b-rtx3090 \\
      --experiment study1_exp1 \\
      --replications 5 \\
      --base-seed 42 \\
      --shuffle

  # Generate jobs for all models, single experiment
  python scripts/generate_replicated_jobs.py \\
      --all-models \\
      --experiment study1_exp2 \\
      --replications 10

  # Generate jobs for all models and all experiments
  python scripts/generate_replicated_jobs.py \\
      --all-models \\
      --all-experiments \\
      --replications 10 \\
      --base-seed 12345
        """
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model',
        type=str,
        help='Model key from models.yaml (e.g., gemma-2b-rtx3090)'
    )
    model_group.add_argument(
        '--all-models',
        action='store_true',
        help='Generate jobs for all models'
    )

    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        '--experiment',
        type=str,
        choices=['study1_exp1', 'study1_exp2', 'study2_exp1', 'study2_exp2'],
        help='Experiment to run'
    )
    exp_group.add_argument(
        '--all-experiments',
        action='store_true',
        help='Generate jobs for all experiments'
    )

    # Replication settings
    parser.add_argument(
        '--replications',
        type=int,
        required=True,
        help='Number of replications to run'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=None,
        help='Base random seed (default: timestamp-based)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle trial order for each replication'
    )

    # Paths
    parser.add_argument(
        '--models-config',
        type=Path,
        default=Path('config/models.yaml'),
        help='Path to models.yaml (default: config/models.yaml)'
    )
    parser.add_argument(
        '--experiments-config',
        type=Path,
        default=Path('config/experiments.yaml'),
        help='Path to experiments.yaml (default: config/experiments.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('kubernetes/generated'),
        help='Output directory for generated manifests (default: kubernetes/generated)'
    )

    args = parser.parse_args()

    # Set base seed if not specified
    if args.base_seed is None:
        args.base_seed = int(datetime.now().timestamp() * 1000) % (2**31)
        print(f"Using auto-generated base seed: {args.base_seed}")

    # Load configurations
    print("Loading configurations...")
    models_config = load_models_config(args.models_config)
    experiments_config = load_experiments_config(args.experiments_config)
    print()

    # Determine which models and experiments to generate
    if args.all_models:
        model_keys = list(models_config['models'].keys())
    else:
        model_keys = [args.model]

    if args.all_experiments:
        experiments = ['study1_exp1', 'study1_exp2', 'study2_exp1', 'study2_exp2']
    else:
        experiments = [args.experiment]

    # Generate manifests
    print(f"Generating {len(model_keys)} × {len(experiments)} = {len(model_keys) * len(experiments)} job manifests...")
    print()

    generated_files = []
    for model_key in model_keys:
        for experiment in experiments:
            try:
                output_path = generate_job_manifest(
                    model_key=model_key,
                    experiment=experiment,
                    num_replications=args.replications,
                    base_seed=args.base_seed,
                    shuffle=args.shuffle,
                    models_config=models_config,
                    experiments_config=experiments_config,
                    output_dir=args.output_dir,
                )
                generated_files.append(output_path)
            except Exception as e:
                print(f"ERROR generating {model_key}/{experiment}: {e}")
                print()

    # Summary
    print("=" * 60)
    print(f"Generated {len(generated_files)} job manifest(s)")
    print(f"Output directory: {args.output_dir}")
    print()
    print("To deploy:")
    for path in generated_files:
        print(f"  kubectl apply -f {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
