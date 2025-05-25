"""
Sampling strategies for APL program selection in the evolutionary process.

This module contains concrete implementations of different sampling strategies
used to select parent and inspiration programs from the program database.
"""

import random
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import SamplingStrategy, APLProgram


class FitnessProportionateSampling:
    """Sample programs proportional to their fitness scores."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def sample_parent(self, programs: List["APLProgram"]) -> "APLProgram":
        if not programs:
            raise ValueError("Cannot sample from empty program list")
        
        # Apply temperature scaling to DPS scores
        scores = [max(0, p.dps_score) for p in programs]
        if self.temperature != 1.0:
            scores = [(s / self.temperature) for s in scores]
        
        # Handle case where all scores are 0
        if sum(scores) == 0:
            return random.choice(programs)
        
        # Weighted random selection
        total = sum(scores)
        r = random.uniform(0, total)
        cumsum = 0
        for i, score in enumerate(scores):
            cumsum += score
            if r <= cumsum:
                return programs[i]
        return programs[-1]  # fallback
    
    def sample_inspirations(self, programs: List["APLProgram"], parent: "APLProgram", 
                          num_inspirations: int = 2) -> List["APLProgram"]:
        # Filter out the parent and get top performers
        available = [p for p in programs if p.program_id != parent.program_id]
        if len(available) <= num_inspirations:
            return available
        
        # Sort by DPS and take from top performers with some randomness
        sorted_programs = sorted(available, key=lambda x: x.dps_score, reverse=True)
        top_third = sorted_programs[:max(1, len(sorted_programs) // 3)]
        
        return random.sample(top_third, min(num_inspirations, len(top_third)))


class TournamentSampling:
    """Tournament selection for balanced exploration/exploitation."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def sample_parent(self, programs: List["APLProgram"]) -> "APLProgram":
        if not programs:
            raise ValueError("Cannot sample from empty program list")
        
        tournament = random.sample(programs, min(self.tournament_size, len(programs)))
        return max(tournament, key=lambda x: x.dps_score)
    
    def sample_inspirations(self, programs: List["APLProgram"], parent: "APLProgram", 
                          num_inspirations: int = 2) -> List["APLProgram"]:
        available = [p for p in programs if p.program_id != parent.program_id]
        inspirations = []
        
        for _ in range(min(num_inspirations, len(available))):
            if not available:
                break
            tournament = random.sample(available, min(self.tournament_size, len(available)))
            selected = max(tournament, key=lambda x: x.dps_score)
            inspirations.append(selected)
            available.remove(selected)
        
        return inspirations


class DiversitySampling:
    """Sample based on both fitness and diversity (age/generation)."""
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def sample_parent(self, programs: List["APLProgram"]) -> "APLProgram":
        if not programs:
            raise ValueError("Cannot sample from empty program list")
        
        # Score based on fitness + diversity (prefer newer/different generations)
        max_gen = max(p.generation for p in programs) if programs else 0
        scores = []
        for p in programs:
            fitness_score = max(0, p.dps_score)
            diversity_score = (max_gen - p.generation + 1)  # newer = higher score
            combined_score = (1 - self.diversity_weight) * fitness_score + \
                           self.diversity_weight * diversity_score
            scores.append(combined_score)
        
        # Weighted selection
        total = sum(scores)
        if total == 0:
            return random.choice(programs)
        
        r = random.uniform(0, total)
        cumsum = 0
        for i, score in enumerate(scores):
            cumsum += score
            if r <= cumsum:
                return programs[i]
        return programs[-1]
    
    def sample_inspirations(self, programs: List["APLProgram"], parent: "APLProgram", 
                          num_inspirations: int = 2) -> List["APLProgram"]:
        available = [p for p in programs if p.program_id != parent.program_id]
        if len(available) <= num_inspirations:
            return available
        
        # Mix of high-fitness and diverse programs
        sorted_by_fitness = sorted(available, key=lambda x: x.dps_score, reverse=True)
        sorted_by_generation = sorted(available, key=lambda x: x.generation, reverse=True)
        
        # Take half from top performers, half from recent generations
        half = num_inspirations // 2
        inspirations = []
        
        # Add top performers
        for i in range(min(half, len(sorted_by_fitness))):
            if sorted_by_fitness[i] not in inspirations:
                inspirations.append(sorted_by_fitness[i])
        
        # Add recent/diverse programs
        for program in sorted_by_generation:
            if len(inspirations) >= num_inspirations:
                break
            if program not in inspirations:
                inspirations.append(program)
        
        return inspirations
