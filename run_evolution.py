#!/usr/bin/env python3
"""
EvoSimC Entrypoint - Run evolutionary APL optimization from YAML configuration.

This entrypoint allows for comprehensive configuration of all components through
a YAML config file, providing more flexibility than the factory function
create_evolution_controller.

Usage:
    python run_evolution.py config.yaml
    python run_evolution.py --config config.yaml
    python run_evolution.py --config config.yaml --dry-run
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from evosim.core import (
    EvolutionController, 
    EvolutionConfig,
    ProgramDatabase,
    create_database,
    create_prompt_sampler,
)
from evosim.llm import create_llm_client
from evosim.simc import SimCRunner


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_evolution_config(config_dict: Dict[str, Any]) -> EvolutionConfig:
    """Create EvolutionConfig from configuration dictionary."""
    evolution_config = config_dict.get('evolution', {})
    mlflow_config = config_dict.get('mlflow', {})
    
    return EvolutionConfig(
        max_generations=evolution_config.get('max_generations', 100),
        num_inspirations=evolution_config.get('num_inspirations', 2),
        simc_iterations=evolution_config.get('simc_iterations', 1000),
        simc_fight_length=evolution_config.get('simc_fight_length', 300),
        checkpoint_interval=evolution_config.get('checkpoint_interval', 10),
        checkpoint_path=evolution_config.get('checkpoint_path', 'checkpoints'),
        verbose=evolution_config.get('verbose', True),
        early_stopping_generations=evolution_config.get('early_stopping_generations', 10),
        experiment_name=mlflow_config.get('experiment_name', 'evosim_evolution'),
        log_artifacts=mlflow_config.get('log_artifacts', True),
        tracking_uri=mlflow_config.get('tracking_uri')
    )


def create_database_from_config(config_dict: Dict[str, Any]) -> ProgramDatabase:
    """Create ProgramDatabase from configuration dictionary."""
    db_config = config_dict.get('database', {})
    
    strategy_name = db_config.get('strategy', 'fitness')
    strategy_params = db_config.get('strategy_params', {})
    
    return create_database(
        strategy_name=strategy_name,
        max_population=db_config.get('max_population', 1000),
        **strategy_params
    )


def create_prompt_sampler_from_config(config_dict: Dict[str, Any]):
    """Create PromptSampler from configuration dictionary."""
    prompt_config = config_dict.get('prompt_sampler', {})
    
    sampler_type = prompt_config.get('type', 'basic')
    sampler_params = prompt_config.get('params', {})
    
    return create_prompt_sampler(sampler_type, **sampler_params)


def create_llm_client_from_config(config_dict: Dict[str, Any]):
    """Create LLM client from configuration dictionary."""
    llm_config = config_dict.get('llm', {})
    
    provider = llm_config.get('provider', 'gemini')
    model_name = llm_config.get('model_name')
    llm_params = llm_config.get('params', {})
    
    # Handle API key
    api_key = llm_config.get('api_key')
    if api_key:
        llm_params['api_key'] = api_key
    
    return create_llm_client(provider, model_name, **llm_params)


def create_simc_runner_from_config(config_dict: Dict[str, Any]) -> SimCRunner:
    """Create SimCRunner from configuration dictionary."""
    simc_config = config_dict.get('simc', {})
    
    return SimCRunner(
        image_name=simc_config.get('image_name', 'simulationcraftorg/simc'),
        **simc_config.get('params', {})
    )


def load_baseline_apl(config_dict: Dict[str, Any]) -> tuple[str, str]:
    """Load baseline APL from configuration."""
    baseline_config = config_dict.get('baseline', {})
    
    if 'apl_code' in baseline_config:
        # Inline APL code
        apl_code = baseline_config['apl_code']
    elif 'apl_file' in baseline_config:
        # APL code from file
        apl_file = Path(baseline_config['apl_file'])
        if not apl_file.exists():
            raise FileNotFoundError(f"Baseline APL file not found: {apl_file}")
        apl_code = apl_file.read_text().strip()
    else:
        raise ValueError("Baseline configuration must specify either 'apl_code' or 'apl_file'")
    
    description = baseline_config.get('description', 'Baseline APL')
    
    return apl_code, description


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary."""
    required_sections = ['baseline']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate baseline configuration
    baseline = config['baseline']
    if 'apl_code' not in baseline and 'apl_file' not in baseline:
        raise ValueError("Baseline configuration must specify either 'apl_code' or 'apl_file'")


def run_evolution(config_file: Path, dry_run: bool = False) -> None:
    """Run the evolutionary optimization from configuration file."""
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if dry_run:
        print("Dry run mode - configuration validated successfully!")
        return
    
    try:
        # Create components
        logger.info("Creating evolution components...")
        
        evolution_config = create_evolution_config(config)
        database = create_database_from_config(config)
        prompt_sampler = create_prompt_sampler_from_config(config)
        llm_client = create_llm_client_from_config(config)
        simc_runner = create_simc_runner_from_config(config)
        
        # Create controller
        controller = EvolutionController(
            database=database,
            prompt_sampler=prompt_sampler,
            llm_client=llm_client,
            simc_runner=simc_runner,
            config=evolution_config
        )
        
        # Load and initialize with baseline
        logger.info("Loading baseline APL...")
        baseline_apl, baseline_description = load_baseline_apl(config)
        controller.initialize_with_baseline(baseline_apl, baseline_description)
        
        # Run evolution
        logger.info("Starting evolutionary optimization...")
        results = controller.run_evolution()
        
        # Print final results
        print("\n" + "=" * 60)
        print("Evolution Complete!")
        print("=" * 60)
        
        evolution_stats = results['evolution_stats']
        database_stats = results['database_stats']
        
        print(f"Generations run: {evolution_stats['generations_run']}")
        print(f"Successful mutations: {evolution_stats['successful_mutations']}")
        print(f"Best DPS achieved: {evolution_stats['best_dps_seen']:,.1f}")
        print(f"Final program count: {database_stats['count']}")
        
        if results.get('best_programs'):
            print(f"\nTop 3 programs:")
            for i, program in enumerate(results['best_programs'][:3], 1):
                print(f"  {i}. ID {program['id']}: {program['dps']:,.1f} DPS (Gen {program['generation']})")
        
        # Get and display best APL
        try:
            best_program, best_apl = controller.get_best_apl()
            print(f"\nBest APL (DPS: {best_program.dps_score:,.1f}):")
            print("-" * 40)
            print(best_apl)
            print("-" * 40)
        except Exception as e:
            logger.warning(f"Could not retrieve best APL: {e}")
        
    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")
        print("\nEvolution interrupted!")
        controller.database.save_to_file('checkpoints/interrupted_database.json')
    except Exception as e:
        logger.error(f"Evolution failed: {e}", exc_info=True)
        print(f"Evolution failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if 'controller' in locals():
                controller.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")





def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run EvoSimC evolutionary APL optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evolution.py config.yaml
  python run_evolution.py --config my_config.yaml
  python run_evolution.py --config config.yaml --dry-run
  python run_evolution.py --example-config > example.yaml
        """
    )
    
    parser.add_argument(
        'config_file', 
        nargs='?',
        type=Path,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to YAML configuration file (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running evolution'
    )
    
    parser.add_argument(
        '--example-config',
        action='store_true',
        help='Print an example configuration file and exit'
    )
    
    args = parser.parse_args()
    
    if args.example_config:
        example_config_path = Path(__file__).parent / "config" / "example_config.yaml"
        try:
            with open(example_config_path, 'r') as f:
                print(f.read())
        except FileNotFoundError:
            print("Error: Example configuration file not found.")
            sys.exit(1)
        return
    
    # Determine config file
    config_file = args.config or args.config_file
    if not config_file:
        parser.error("Configuration file is required (provide as positional argument or with --config)")
    
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Run evolution
    run_evolution(config_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
