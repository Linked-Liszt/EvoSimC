"""
Lightweight tests for the database module.

These tests focus on core functionality and are designed to be resilient
to experimental changes in the implementation.
"""

import pytest
import tempfile
import os
from evosim.core.database import ProgramDatabase, create_database
from evosim.core.sampling import FitnessProportionateSampling, TournamentSampling
from evosim.entities import APLProgram, SimCResult

class TestProgramDatabase:
    """Test ProgramDatabase core functionality."""
    
    def test_empty_database(self):
        """Test database initialization and empty state."""
        db = ProgramDatabase()
        
        assert len(db.programs) == 0
        assert db.current_generation == 0
        assert db.max_population == 1000
        
        with pytest.raises(ValueError, match="Cannot sample from empty database"):
            db.sample()
    
    def test_add_program(self):
        """Test adding programs to database."""
        db = ProgramDatabase()
        
        simc_result = SimCResult(
            dps=50000.0,
            raw_output="simulation output",
            errors=[],
            is_valid=True
        )
        
        program = db.add(
            apl_code="actions=test",
            simc_result=simc_result
        )
        
        assert len(db.programs) == 1
        assert program.dps_score == 50000.0
        assert program.generation == 0
        assert program in db.programs
    
    def test_add_program_with_reasoning(self):
        """Test adding program with reasoning field."""
        db = ProgramDatabase()
        
        simc_result = SimCResult(
            dps=50000.0,
            raw_output="simulation output",
            errors=[],
            is_valid=True
        )
        
        program = db.add(
            apl_code="actions=test",
            simc_result=simc_result,
            parent_id="parent_123",
            diff_applied="+ actions.precombat=flask",
            reasoning="Testing flask optimization"
        )
        
        assert program.parent_id == "parent_123"
        assert program.diff_applied == "+ actions.precombat=flask"
        assert program.reasoning == "Testing flask optimization"
    
    def test_sampling(self):
        """Test basic sampling functionality."""
        db = ProgramDatabase()
        
        # Add some test programs
        for i in range(5):
            simc_result = SimCResult(
                dps=50000.0 + i * 1000,
                raw_output="simulation output",
                errors=[],
                is_valid=True
            )
            db.add(f"actions=test_{i}", simc_result)
        
        parent, inspirations = db.sample(num_inspirations=2)
        
        assert isinstance(parent, APLProgram)
        assert len(inspirations) <= 2
        assert parent not in inspirations
    
    def test_get_best_programs(self):
        """Test retrieving best programs by DPS."""
        db = ProgramDatabase()
        
        # Add programs with different DPS scores
        dps_scores = [45000, 52000, 48000, 55000, 47000]
        for i, dps in enumerate(dps_scores):
            simc_result = SimCResult(
                dps=dps,
                raw_output="simulation output",
                errors=[],
                is_valid=True
            )
            db.add(f"actions=test_{i}", simc_result)
        
        best_3 = db.get_best_programs(3)
        
        assert len(best_3) == 3
        assert best_3[0].dps_score == 55000  # Highest
        assert best_3[1].dps_score == 52000
        assert best_3[2].dps_score == 48000
    
    def test_statistics(self):
        """Test database statistics calculation."""
        db = ProgramDatabase()
        
        # Empty database
        stats = db.get_statistics()
        assert stats["count"] == 0
        assert stats["generations"] == 0
        
        # Add some programs
        dps_scores = [45000, 50000, 55000]
        for dps in dps_scores:
            simc_result = SimCResult(
                dps=dps,
                raw_output="simulation output",
                errors=[],
                is_valid=True
            )
            db.add("actions=test", simc_result)
        
        stats = db.get_statistics()
        assert stats["count"] == 3
        assert stats["best_dps"] == 55000
        assert stats["worst_dps"] == 45000
        assert stats["avg_dps"] == 50000
    
    def test_generation_advancement(self):
        """Test generation tracking."""
        db = ProgramDatabase()
        
        # Add program in generation 0
        simc_result1 = SimCResult(
            dps=50000,
            raw_output="simulation output",
            errors=[],
            is_valid=True
        )
        p1 = db.add("actions=test1", simc_result1)
        assert p1.generation == 0
        
        # Advance generation and add another
        db.advance_generation()
        simc_result2 = SimCResult(
            dps=51000,
            raw_output="simulation output",
            errors=[],
            is_valid=True
        )
        p2 = db.add("actions=test2", simc_result2)
        assert p2.generation == 1
    
    def test_population_limit(self):
        """Test population size management."""
        db = ProgramDatabase(max_population=3)
        
        # Add more programs than limit
        for i in range(5):
            simc_result = SimCResult(
                dps=50000 + i,
                raw_output="simulation output",
                errors=[],
                is_valid=True
            )
            db.add(f"actions=test_{i}", simc_result)
        
        # Should maintain population limit
        assert len(db.programs) == 3
        
        # Should keep best performers
        dps_scores = [p.dps_score for p in db.programs]
        assert max(dps_scores) == 50004  # Best one kept


class TestDatabasePersistence:
    """Test save/load functionality."""
    
    def test_save_and_load(self):
        """Test saving database to file and loading it back."""
        db = ProgramDatabase()
        
        # Add some test data
        simc_result = SimCResult(
            dps=50000,
            raw_output="simulation output",
            errors=[],
            is_valid=True
        )
        db.add(
            "actions=test",
            simc_result,
            reasoning="Test program"
        )
        db.advance_generation()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            db.save_to_file(filepath)
            
            # Load into new database
            new_db = ProgramDatabase()
            new_db.load_from_file(filepath)
            
            assert len(new_db.programs) == 1
            assert new_db.current_generation == 1
            assert new_db.programs[0].dps_score == 50000
            assert new_db.programs[0].reasoning == "Test program"
            
        finally:
            os.unlink(filepath)
    
    def test_new_format_loading(self):
        """Test loading files with new SimCResult format."""
        db = ProgramDatabase()
        
        # Simulate new file format with SimCResult
        new_format_data = {
            "max_population": 1000,
            "current_generation": 0,
            "programs": [{
                "program_id": "test_123",
                "apl_code": "actions=test",
                "dps_score": 50000.0,
                "generation": 0,
                "timestamp": 1234567890.0,
                "parent_id": None,
                "diff_applied": None,
                "reasoning": "Test program reasoning",
                "simc_result": {
                    "dps": 50000.0,
                    "raw_output": "simulation output",
                    "errors": [],
                    "is_valid": True
                }
            }]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(new_format_data, f)
            filepath = f.name
        
        try:
            db.load_from_file(filepath)
            assert len(db.programs) == 1
            assert db.programs[0].reasoning == "Test program reasoning"
            assert db.programs[0].simc_result.dps == 50000.0
            assert db.programs[0].simc_result.is_valid == True
            
        finally:
            os.unlink(filepath)


class TestCreateDatabaseFunction:
    """Test the convenience create_database function."""
    
    def test_create_with_default_strategy(self):
        """Test creating database with default fitness strategy."""
        db = create_database()
        
        assert isinstance(db, ProgramDatabase)
        assert isinstance(db.sampling_strategy, FitnessProportionateSampling)
    
    def test_create_with_tournament_strategy(self):
        """Test creating database with tournament strategy."""
        db = create_database("tournament", tournament_size=5)
        
        assert isinstance(db.sampling_strategy, TournamentSampling)
        assert db.sampling_strategy.tournament_size == 5
    
    def test_create_with_invalid_strategy(self):
        """Test creating database with invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_database("invalid_strategy")
    
    def test_create_with_custom_population(self):
        """Test creating database with custom population size."""
        db = create_database("fitness", max_population=500)
        
        assert db.max_population == 500


class TestSamplingIntegration:
    """Test sampling strategies work with database."""
    
    def test_sampling_strategy_change(self):
        """Test changing sampling strategy at runtime."""
        db = ProgramDatabase()
        
        # Add some test programs
        for i in range(3):
            simc_result = SimCResult(
                dps=50000 + i * 1000,
                raw_output="simulation output",
                errors=[],
                is_valid=True
            )
            db.add(f"actions=test_{i}", simc_result)
        
        # Test with fitness strategy
        original_strategy = db.sampling_strategy
        parent1, _ = db.sample()
        
        # Change to tournament strategy
        new_strategy = TournamentSampling(tournament_size=2)
        db.set_sampling_strategy(new_strategy)
        parent2, _ = db.sample()
        
        assert db.sampling_strategy is new_strategy
        assert db.sampling_strategy is not original_strategy
        assert isinstance(parent1, APLProgram)
        assert isinstance(parent2, APLProgram)
