"""
Controller for EvoSimC - AlphaEvolve-inspired evolutionary optimization of SimC APLs.

This module implements the main evolutionary loop that coordinates:
- Parent and inspiration sampling from the database
- Prompt construction and LLM generation
- Diff application to create child programs  
- SimC evaluation for fitness assessment
- Database updates with new programs

Based on AlphaEvolve's evolutionary algorithm design.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import mlflow
import mlflow.metrics

from .database import ProgramDatabase
from .prompt_sampler import PromptSampler, DiffApplicationError
from ..entities import APLProgram
from ..llm import GeminiLLMClient, LLMGenerationError
from ..simc import SimCRunner


logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary process."""
    max_generations: int = 100
    num_inspirations: int = 2
    simc_iterations: int = 1000
    simc_fight_length: int = 300
    verbose: bool = True
    early_stopping_generations: int = 10  # Stop if no improvement for N generations
    
    # MLflow configuration
    experiment_name: str = "evosim_evolution"
    log_artifacts: bool = True
    tracking_uri: Optional[str] = None  # Use default local tracking if None
    

class EvolutionController:
    """
    Main controller for the evolutionary optimization process.
    
    Follows AlphaEvolve's core loop:
    1. parent_program, inspirations = database.sample()
    2. prompt = prompt_sampler.build(parent_program, inspirations)  
    3. diff = llm.generate(prompt)
    4. child_program = apply_diff(parent_program, diff)
    5. results = evaluator.execute(child_program)
    6. database.add(child_program, results)
    """
    
    def __init__(self, 
                 database: ProgramDatabase,
                 prompt_sampler: PromptSampler,
                 llm_client: GeminiLLMClient,
                 simc_runner: SimCRunner,
                 config: Optional[EvolutionConfig] = None):
        self.database = database
        self.prompt_sampler = prompt_sampler
        self.llm_client = llm_client
        self.simc_runner = simc_runner
        self.config = config or EvolutionConfig()
        
        # Track evolution statistics
        self.stats = {
            'generations_run': 0,
            'successful_mutations': 0,
            'failed_diff_applications': 0,
            'failed_evaluations': 0,
            'best_dps_seen': 0.0,
            'generations_without_improvement': 0
        }
        
        # Storage for baseline data to log when MLflow run is active
        self._baseline_data = None
        
        # Initialize MLflow tracking
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow experiment and run."""
        # Set tracking URI if specified
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.config.experiment_name)
            logger.info(f"Created new MLflow experiment: {self.config.experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")
        
        mlflow.set_experiment(experiment_id=experiment_id)
    
    def run_evolution(self) -> Dict[str, Any]:
        """
        Run the complete evolutionary optimization process.
        
        Returns:
            Dictionary containing evolution results and statistics
        """
        logger.info(f"Starting evolution for {self.config.max_generations} generations")
        
        # Start MLflow run
        with mlflow.start_run():
            return self._run_evolution_loop()
    
    def _run_evolution_loop(self) -> Dict[str, Any]:
        """Run evolution loop with MLflow tracking."""
        # Log baseline data if available (from initialize_with_baseline)
        if self._baseline_data:
            mlflow.log_params({
                "baseline_dps": self._baseline_data['baseline_dps'],
                "baseline_description": self._baseline_data['baseline_description']
            })
            mlflow.log_metric("baseline_dps", self._baseline_data['baseline_dps'])
            
            # Log baseline APL as artifact
            if self.config.log_artifacts:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.simc', delete=False) as f:
                    f.write(self._baseline_data['baseline_apl'])
                    mlflow.log_artifact(f.name, "baseline_apl.simc")
        
        # Log configuration parameters
        config_dict = asdict(self.config)
        for key, value in config_dict.items():
            if value is not None:
                mlflow.log_param(key, value)
        
        # Log initial database stats
        initial_stats = self.database.get_statistics()
        mlflow.log_metric("initial_program_count", initial_stats['count'])
        if initial_stats['count'] > 0:
            mlflow.log_metric("initial_best_dps", initial_stats['best_dps'])
            mlflow.log_metric("initial_avg_dps", initial_stats['avg_dps'])
        
        if self.config.verbose:
            self._print_initial_stats()
        
        for generation in range(self.config.max_generations):
            try:
                # Run one generation of evolution
                result = self._run_single_generation(generation)
                
                self.stats['generations_run'] = generation + 1
                
                if self.config.verbose:
                    self._print_generation_results(generation, result)
                
                # Check for improvement
                if result['child_dps'] > self.stats['best_dps_seen']:
                    self.stats['best_dps_seen'] = result['child_dps']
                    self.stats['generations_without_improvement'] = 0
                    
                    # Log new best DPS
                    mlflow.log_metric("best_dps", result['child_dps'], step=generation)
                else:
                    self.stats['generations_without_improvement'] += 1
                
                # Log per-generation metrics
                mlflow.log_metrics({
                    "generation_parent_dps": result['parent_dps'],
                    "generation_child_dps": result['child_dps'],
                    "generation_improvement": result['child_dps'] - result['parent_dps'],
                    "generations_without_improvement": self.stats['generations_without_improvement'],
                    "successful_mutations_total": self.stats['successful_mutations'],
                    "failed_diff_applications_total": self.stats['failed_diff_applications'],
                    "failed_evaluations_total": self.stats['failed_evaluations']
                }, step=generation)
                
                # Early stopping check
                if (self.stats['generations_without_improvement'] >= 
                    self.config.early_stopping_generations):
                    logger.info(f"Early stopping after {generation + 1} generations "
                              f"(no improvement for {self.config.early_stopping_generations})")
                    mlflow.log_param("early_stopped", True)
                    mlflow.log_param("early_stop_generation", generation + 1)
                    break
                    
            except Exception as e:
                logger.error(f"Error in generation {generation}: {e}")
                mlflow.log_metric("generation_errors", 1, step=generation)
                continue
        
        # Update database generation counter
        self.database.current_generation = self.stats['generations_run']
        
        # Log final results and artifacts
        final_results = self._get_final_results()
        self._log_final_results(final_results)
        
        return final_results
    
    def _run_single_generation(self, generation: int) -> Dict[str, Any]:
        """Run a single generation with MLflow tracking."""
        result = self.run_single_generation()
        
        return result
    
    def run_single_generation(self) -> Dict[str, Any]:
        """
        Run a single generation of the evolutionary algorithm.
        
        Returns:
            Dictionary containing generation results
        """
        # Step 1: Sample parent and inspirations from database
        parent_program, inspirations = self.database.sample(
            num_inspirations=self.config.num_inspirations
        )
        
        # Step 2: Build prompt with parent and inspirations
        prompt = self.prompt_sampler.build(parent_program, inspirations)
        
        # Step 3: Generate diff using LLM
        try:
            diff_response = self.llm_client.generate(prompt)
        except LLMGenerationError as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        
        # Step 4: Apply diff to create child program
        try:
            child_apl_code = self.prompt_sampler.apply_diff(
                parent_program.apl_code, diff_response
            )
        except DiffApplicationError as e:
            logger.warning(f"Diff application failed: {e}")
            self.stats['failed_diff_applications'] += 1
            # Create a failed child program with 0 DPS
            child_apl_code = parent_program.apl_code  # Fallback to parent
            evaluation_results = {'dps': 0.0, 'error': str(e)}
        else:
            # Step 5: Evaluate child program with SimC
            try:
                child_dps = self.simc_runner.evaluate_apl(
                    child_apl_code,
                    iterations=self.config.simc_iterations,
                    fight_length=self.config.simc_fight_length
                )
                evaluation_results = {
                    'dps': child_dps,
                    'iterations': self.config.simc_iterations,
                    'fight_length': self.config.simc_fight_length
                }
                
                if child_dps > 0:
                    self.stats['successful_mutations'] += 1
                else:
                    self.stats['failed_evaluations'] += 1
                    
            except Exception as e:
                logger.warning(f"SimC evaluation failed: {e}")
                self.stats['failed_evaluations'] += 1
                evaluation_results = {'dps': 0.0, 'error': str(e)}
        
        # Step 6: Add child program to database
        child_program = self.database.add(
            apl_code=child_apl_code,
            evaluation_results=evaluation_results,
            parent_id=parent_program.program_id,
            diff_applied=diff_response,
            reasoning=self._extract_reasoning(diff_response)
        )
        
        return {
            'parent_id': parent_program.program_id,
            'parent_dps': parent_program.dps_score,
            'child_id': child_program.program_id,
            'child_dps': child_program.dps_score,
            'inspirations': [p.program_id for p in inspirations],
            'diff_applied': len(diff_response) > 0,
            'evaluation_success': evaluation_results.get('dps', 0) > 0
        }
    
    def initialize_with_baseline(self, baseline_apl: str, 
                               baseline_description: str = "Baseline APL") -> APLProgram:
        """
        Initialize the database with a baseline APL program.
        
        Args:
            baseline_apl: The initial APL code
            baseline_description: Description for the baseline
            
        Returns:
            The created baseline APLProgram
        """
        logger.info("Evaluating baseline APL...")
        
        try:
            baseline_dps = self.simc_runner.evaluate_apl(
                baseline_apl,
                iterations=self.config.simc_iterations,
                fight_length=self.config.simc_fight_length
            )
            
            evaluation_results = {
                'dps': baseline_dps,
                'iterations': self.config.simc_iterations,
                'fight_length': self.config.simc_fight_length
            }
            
            baseline_program = self.database.add(
                apl_code=baseline_apl,
                evaluation_results=evaluation_results,
                reasoning=baseline_description
            )
            
            self.stats['best_dps_seen'] = baseline_dps
            
            # Store baseline data for later MLflow logging (when run is active)
            self._baseline_data = {
                'baseline_apl': baseline_apl,
                'baseline_dps': baseline_dps,
                'baseline_description': baseline_description
            }
            
            logger.info(f"Baseline APL added: DPS = {baseline_dps:,.1f}")
            return baseline_program
            
        except Exception as e:
            logger.error(f"Failed to evaluate baseline APL: {e}")
            raise
    
    def _extract_reasoning(self, llm_response: str) -> str:
        """Extract reasoning from LLM response (text before diff blocks)."""
        # Find the first diff block
        search_patterns = ['<<<<<<< SEARCH', '-<<<<<<< SEARCH']
        
        for pattern in search_patterns:
            if pattern in llm_response:
                reasoning = llm_response.split(pattern)[0].strip()
                # Clean up and limit length
                reasoning = reasoning.replace('\n\n\n', '\n\n')
                if len(reasoning) > 500:
                    reasoning = reasoning[:500] + "..."
                return reasoning
        
        # If no diff blocks found, use the whole response (truncated)
        reasoning = llm_response.strip()
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."
        return reasoning
    
    def _print_initial_stats(self):
        """Print initial database statistics."""
        stats = self.database.get_statistics()
        print(f"\n=== Initial Database Stats ===")
        print(f"Programs: {stats['count']}")
        if stats['count'] > 0:
            print(f"Best DPS: {stats['best_dps']:,.1f}")
            print(f"Average DPS: {stats['avg_dps']:,.1f}")
        print()
    
    def _print_generation_results(self, generation: int, result: Dict[str, Any]):
        """Print results for a single generation."""
        improvement = result['child_dps'] - result['parent_dps']
        improvement_pct = (improvement / result['parent_dps']) * 100 if result['parent_dps'] > 0 else 0
        
        status = "✓" if result['evaluation_success'] else "✗"
        
        print(f"Gen {generation + 1:3d}: {status} "
              f"Parent: {result['parent_dps']:8,.1f} → "
              f"Child: {result['child_dps']:8,.1f} "
              f"({improvement:+7.1f}, {improvement_pct:+5.1f}%)")
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Get final evolution results and statistics."""
        final_stats = self.database.get_statistics()
        llm_stats = self.llm_client.get_usage_stats()
        
        best_programs = self.database.get_best_programs(5)
        
        return {
            'evolution_stats': self.stats,
            'database_stats': final_stats,
            'llm_stats': llm_stats,
            'best_programs': [
                {
                    'id': p.program_id,
                    'dps': p.dps_score,
                    'generation': p.generation,
                    'reasoning': p.reasoning[:100] + "..." if p.reasoning and len(p.reasoning) > 100 else p.reasoning
                }
                for p in best_programs
            ]
        }
    
    def _log_final_results(self, results: Dict[str, Any]):
        """Log final evolution results to MLflow."""
        # Log final statistics as metrics
        evolution_stats = results['evolution_stats']
        database_stats = results['database_stats']
        llm_stats = results.get('llm_stats', {})
        
        # Final evolution metrics
        mlflow.log_metrics({
            "final_generations_run": evolution_stats['generations_run'],
            "final_successful_mutations": evolution_stats['successful_mutations'],
            "final_failed_diff_applications": evolution_stats['failed_diff_applications'],
            "final_failed_evaluations": evolution_stats['failed_evaluations'],
            "final_best_dps": evolution_stats['best_dps_seen'],
            "final_program_count": database_stats['count'],
            "final_avg_dps": database_stats.get('avg_dps', 0),
            "mutation_success_rate": evolution_stats['successful_mutations'] / max(evolution_stats['generations_run'], 1),
            "diff_application_success_rate": 1 - (evolution_stats['failed_diff_applications'] / max(evolution_stats['generations_run'], 1)),
            "evaluation_success_rate": 1 - (evolution_stats['failed_evaluations'] / max(evolution_stats['generations_run'], 1))
        })
        
        # Log LLM usage stats if available
        if llm_stats:
            for key, value in llm_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"llm_{key}", value)
        
        # Log best programs as artifacts
        if self.config.log_artifacts and results.get('best_programs'):
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(results['best_programs'], f, indent=2)
                mlflow.log_artifact(f.name, "best_programs.json")
            
            # Log the best APL code as an artifact
            try:
                best_program, best_apl_code = self.get_best_apl()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.simc', delete=False) as f:
                    f.write(best_apl_code)
                    mlflow.log_artifact(f.name, "best_apl.simc")
            except Exception as e:
                logger.warning(f"Failed to log best APL artifact: {e}")
    
    def get_best_apl(self) -> Tuple[APLProgram, str]:
        """
        Get the best performing APL and its code.
        
        Returns:
            Tuple of (best_program, apl_code)
        """
        best_programs = self.database.get_best_programs(1)
        if not best_programs:
            raise ValueError("No programs in database")
        
        best_program = best_programs[0]
        return best_program, best_program.apl_code
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.simc_runner.cleanup()
        except Exception as e:
            logger.warning(f"Error during SimC cleanup: {e}")
        
        # End MLflow run if active
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {e}")


# Factory function for easy controller creation
def create_evolution_controller(baseline_apl: str,
                              llm_api_key: Optional[str] = None,
                              simc_image: str = "simulationcraftorg/simc",
                              config: Optional[EvolutionConfig] = None,
                              experiment_name: str = "evosim_evolution",
                              mlflow_tracking_uri: Optional[str] = None) -> EvolutionController:
    """
    Factory function to create a complete evolution controller with all components.
    
    Args:
        baseline_apl: Initial APL code to start evolution from
        llm_api_key: API key for Gemini LLM (or use GOOGLE_API_KEY env var)
        simc_image: Docker image for SimulationCraft
        config: Evolution configuration (if None, will create with MLflow settings)
        experiment_name: Name for MLflow experiment
        mlflow_tracking_uri: MLflow tracking URI (if None, uses local tracking)
        
    Returns:
        Configured EvolutionController ready to run
    """
    from .database import create_database
    from .prompt_sampler import create_prompt_sampler
    from ..llm import create_llm_client
    
    # Create or update config with MLflow settings
    if config is None:
        config = EvolutionConfig(
            experiment_name=experiment_name,
            tracking_uri=mlflow_tracking_uri
        )
    else:
        # Update MLflow settings with provided parameters (override existing values)
        if experiment_name != "evosim_evolution":  # Only override if explicitly provided
            config.experiment_name = experiment_name
        if mlflow_tracking_uri is not None:
            config.tracking_uri = mlflow_tracking_uri
    
    # Create components
    database = create_database("basic")
    prompt_sampler = create_prompt_sampler("basic")
    llm_client = create_llm_client("gemini", api_key=llm_api_key)
    simc_runner = SimCRunner(image_name=simc_image)
    
    # Create controller
    controller = EvolutionController(
        database=database,
        prompt_sampler=prompt_sampler,
        llm_client=llm_client,
        simc_runner=simc_runner,
        config=config
    )
    
    # Initialize with baseline
    controller.initialize_with_baseline(baseline_apl)
    
    return controller