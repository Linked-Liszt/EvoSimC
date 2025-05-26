"""
Program Database for storing and managing APL evolution candidates.

Based on AlphaEvolve's evolutionary database concepts, this module provides
functionality for storing APL programs with their fitness scores and 
evolutionary metadata, plus sampling strategies for parent/inspiration selection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import json

from ..entities import APLProgram, SimCResult


class SamplingStrategy(ABC):
    """Abstract base class for different sampling strategies."""
    
    @abstractmethod
    def sample_parent(self, programs: List[APLProgram]) -> APLProgram:
        """Sample a parent program from the database."""
        pass
    
    @abstractmethod
    def sample_inspirations(self, programs: List[APLProgram], parent: APLProgram, 
                          num_inspirations: int = 2) -> List[APLProgram]:
        """Sample inspiration programs from the database."""
        pass


class ProgramDatabase:
    """
    Database for storing and managing APL programs in the evolutionary process.
    
    Inspired by AlphaEvolve's program database concepts, this class maintains
    a population of APL programs with their fitness scores and provides various
    sampling strategies for parent and inspiration selection.
    """
    
    def __init__(self, max_population: int = 1000, 
                 sampling_strategy: Optional[SamplingStrategy] = None):
        self.programs: List[APLProgram] = []
        self.max_population = max_population
        # Import here to avoid circular imports
        if sampling_strategy is None:
            from .sampling import FitnessProportionateSampling
            sampling_strategy = FitnessProportionateSampling()
        self.sampling_strategy = sampling_strategy
        self.current_generation = 0
        
    def add(self, apl_code: str, simc_result: SimCResult, 
            parent_id: Optional[str] = None, diff_applied: Optional[str] = None,
            reasoning: Optional[str] = None) -> APLProgram:
        """
        Add a new program to the database.
        
        Args:
            apl_code: The APL code string
            simc_result: Complete SimC evaluation result
            parent_id: ID of parent program (if this is an evolved program)
            diff_applied: The diff that was applied to create this program
            reasoning: LLM's reasoning for the diff/changes made
            
        Returns:
            The created APLProgram object
        """
        program = APLProgram(
            apl_code=apl_code,
            dps_score=simc_result.dps,
            generation=self.current_generation,
            parent_id=parent_id,
            diff_applied=diff_applied,
            reasoning=reasoning,
            simc_result=simc_result
        )
        
        print(program.apl_code)
        
        self.programs.append(program)
        self._maintain_population_size()
        
        return program
    
    def sample(self, num_inspirations: int = 2) -> Tuple[APLProgram, List[APLProgram]]:
        """
        Sample a parent program and inspiration programs for evolution.
        
        Args:
            num_inspirations: Number of inspiration programs to sample
            
        Returns:
            Tuple of (parent_program, list_of_inspiration_programs)
        """
        if not self.programs:
            raise ValueError("Cannot sample from empty database")
        
        parent = self.sampling_strategy.sample_parent(self.programs)
        inspirations = self.sampling_strategy.sample_inspirations(
            self.programs, parent, num_inspirations
        )
        
        return parent, inspirations
    
    def get_best_programs(self, n: int = 10) -> List[APLProgram]:
        """Get the top N programs by DPS score."""
        return sorted(self.programs, key=lambda x: x.dps_score, reverse=True)[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.programs:
            return {"count": 0, "generations": 0}
        
        dps_scores = [p.dps_score for p in self.programs]
        return {
            "count": len(self.programs),
            "generations": self.current_generation + 1,
            "best_dps": max(dps_scores),
            "avg_dps": sum(dps_scores) / len(dps_scores),
            "worst_dps": min(dps_scores),
            "unique_parents": len(set(p.parent_id for p in self.programs if p.parent_id))
        }
    
    def advance_generation(self):
        """Advance to the next generation."""
        self.current_generation += 1
    
    def set_sampling_strategy(self, strategy: SamplingStrategy):
        """Change the sampling strategy."""
        self.sampling_strategy = strategy
    
    def save_to_file(self, filepath: str):
        """Save database to JSON file."""
        data = {
            "max_population": self.max_population,
            "current_generation": self.current_generation,
            "programs": [
                {
                    "program_id": p.program_id,
                    "apl_code": p.apl_code,
                    "dps_score": p.dps_score,
                    "generation": p.generation,
                    "timestamp": p.timestamp,
                    "parent_id": p.parent_id,
                    "diff_applied": p.diff_applied,
                    "reasoning": p.reasoning,
                    "simc_result": {
                        "dps": p.simc_result.dps,
                        "raw_output": p.simc_result.raw_output,
                        "errors": p.simc_result.errors,
                        "is_valid": p.simc_result.is_valid
                    } if p.simc_result else None
                }
                for p in self.programs
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load database from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.max_population = data["max_population"]
        self.current_generation = data["current_generation"]
        
        self.programs = []
        for p_data in data["programs"]:
            # Reconstruct SimCResult if it exists
            simc_result = None
            if p_data.get("simc_result"):
                simc_data = p_data["simc_result"]
                simc_result = SimCResult(
                    dps=simc_data["dps"],
                    raw_output=simc_data["raw_output"],
                    errors=simc_data["errors"],
                    is_valid=simc_data["is_valid"]
                )
            
            program = APLProgram(
                apl_code=p_data["apl_code"],
                dps_score=p_data["dps_score"],
                generation=p_data["generation"],
                timestamp=p_data["timestamp"],
                parent_id=p_data["parent_id"],
                diff_applied=p_data["diff_applied"],
                reasoning=p_data["reasoning"],
                simc_result=simc_result,
                program_id=p_data["program_id"]
            )
            self.programs.append(program)
    
    def _maintain_population_size(self):
        """Maintain population size by culling low performers if needed."""
        if len(self.programs) <= self.max_population:
            return
        
        # Sort by DPS score and keep the top performers
        # Also ensure we keep some diversity by including recent programs
        sorted_by_fitness = sorted(self.programs, key=lambda x: x.dps_score, reverse=True)
        sorted_by_generation = sorted(self.programs, key=lambda x: x.generation, reverse=True)
        
        # Keep top 80% by fitness, 20% by recency
        fitness_keep = int(self.max_population * 0.8)
        generation_keep = self.max_population - fitness_keep
        
        to_keep = set()
        
        # Add top performers
        for i in range(min(fitness_keep, len(sorted_by_fitness))):
            to_keep.add(sorted_by_fitness[i].program_id)
        
        # Add recent programs (if not already included)
        for program in sorted_by_generation:
            if len(to_keep) >= self.max_population:
                break
            if program.program_id not in to_keep:
                to_keep.add(program.program_id)
        
        # Filter programs to keep only those in to_keep set
        self.programs = [p for p in self.programs if p.program_id in to_keep]


# Convenience function for quick database creation with different strategies
def create_database(strategy_name: str = "fitness", **kwargs) -> ProgramDatabase:
    """
    Create a ProgramDatabase with a specific sampling strategy.
    
    Args:
        strategy_name: One of "fitness", "tournament", "diversity"
        **kwargs: Additional arguments for the strategy or database
        
    Returns:
        Configured ProgramDatabase
    """
    from .sampling import FitnessProportionateSampling, TournamentSampling, DiversitySampling
    
    strategies = {
        "fitness": FitnessProportionateSampling,
        "tournament": TournamentSampling,
        "diversity": DiversitySampling
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    # Extract database-specific kwargs
    max_population = kwargs.pop("max_population", 1000)
    
    # Create strategy with remaining kwargs
    strategy = strategies[strategy_name](**kwargs)
    
    return ProgramDatabase(max_population=max_population, sampling_strategy=strategy)