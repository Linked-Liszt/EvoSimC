import pytest
import time
from evosim.entities import APLProgram, SimCResult


class TestSimCResult:
    """Test SimCResult dataclass functionality."""
    
    def test_simc_result_creation(self):
        """Test basic SimCResult creation."""
        result = SimCResult(
            dps=50000.0,
            raw_output="SimC output here",
            errors=[],
            is_valid=True
        )
        
        assert result.dps == 50000.0
        assert result.raw_output == "SimC output here"
        assert result.errors == []
        assert result.is_valid is True
    
    def test_simc_result_with_errors(self):
        """Test SimCResult with validation errors."""
        result = SimCResult(
            dps=0.0,
            raw_output="Error output",
            errors=["Invalid APL syntax", "Missing required action"],
            is_valid=False
        )
        
        assert result.dps == 0.0
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Invalid APL syntax" in result.errors
        assert "Missing required action" in result.errors


class TestAPLProgram:
    """Test APLProgram dataclass functionality."""
    
    def test_program_creation(self):
        """Test basic program creation with auto-generated ID."""
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1
        )
        
        assert program.apl_code == "actions=test"
        assert program.dps_score == 50000.0
        assert program.generation == 1
        assert program.program_id.startswith("apl_")
        assert program.parent_id is None
        assert program.diff_applied is None
        assert program.reasoning is None
        assert program.simc_result is None
    
    def test_program_with_metadata(self):
        """Test program creation with all metadata fields."""
        simc_result = SimCResult(
            dps=45000.0,
            raw_output="Simulation completed successfully",
            errors=[],
            is_valid=True
        )
        
        program = APLProgram(
            apl_code="actions=test",
            dps_score=45000.0,
            generation=2,
            parent_id="parent_123",
            diff_applied="+ actions.precombat=flask",
            reasoning="Added flask for better DPS",
            simc_result=simc_result
        )
        
        assert program.parent_id == "parent_123"
        assert program.diff_applied == "+ actions.precombat=flask"
        assert program.reasoning == "Added flask for better DPS"
        assert program.simc_result is not None
        assert program.simc_result.dps == 45000.0
        assert program.simc_result.is_valid is True
    
    def test_program_id_generation(self):
        """Test that program IDs are unique and properly formatted."""
        program1 = APLProgram(
            apl_code="actions=spell1",
            dps_score=50000.0,
            generation=1
        )
        
        time.sleep(0.001)  # Small delay to ensure different timestamp
        
        program2 = APLProgram(
            apl_code="actions=spell2", 
            dps_score=51000.0,
            generation=1
        )
        
        assert program1.program_id != program2.program_id
        assert program1.program_id.startswith("apl_")
        assert program2.program_id.startswith("apl_")
        assert len(program1.program_id.split("_")) == 3  # apl_timestamp_hash
        assert len(program2.program_id.split("_")) == 3
    
    def test_program_with_custom_id(self):
        """Test program creation with a custom ID."""
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1,
            program_id="custom_id_123"
        )
        
        assert program.program_id == "custom_id_123"
    
    def test_dps_consistency_with_simc_result(self):
        """Test that dps_score is consistent with simc_result.dps."""
        simc_result = SimCResult(
            dps=55000.0,
            raw_output="Test output",
            errors=[],
            is_valid=True
        )
        
        # Create program with different dps_score - should be overridden
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,  # This should be overridden
            generation=1,
            simc_result=simc_result
        )
        
        # dps_score should be updated to match simc_result.dps
        assert program.dps_score == 55000.0
        assert program.simc_result.dps == 55000.0
    
    def test_program_with_invalid_simc_result(self):
        """Test program with invalid simulation result."""
        simc_result = SimCResult(
            dps=0.0,
            raw_output="Simulation failed",
            errors=["Syntax error in APL", "Unknown spell name"],
            is_valid=False
        )
        
        program = APLProgram(
            apl_code="actions=invalid_spell",
            dps_score=0.0,
            generation=1,
            simc_result=simc_result
        )
        
        assert program.dps_score == 0.0
        assert program.simc_result.is_valid is False
        assert len(program.simc_result.errors) == 2
    
    def test_timestamp_generation(self):
        """Test that timestamp is automatically generated."""
        before_creation = time.time()
        
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1
        )
        
        after_creation = time.time()
        
        assert before_creation <= program.timestamp <= after_creation
    
    def test_program_with_custom_timestamp(self):
        """Test program creation with custom timestamp."""
        custom_timestamp = 1234567890.0
        
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1,
            timestamp=custom_timestamp
        )
        
        assert program.timestamp == custom_timestamp
    
    def test_program_comparison_by_dps(self):
        """Test comparing programs by DPS score."""
        program1 = APLProgram(
            apl_code="actions=spell1",
            dps_score=50000.0,
            generation=1
        )
        
        program2 = APLProgram(
            apl_code="actions=spell2",
            dps_score=55000.0,
            generation=1
        )
        
        assert program2.dps_score > program1.dps_score
    
    def test_program_serialization_fields(self):
        """Test that all expected fields are present for serialization."""
        simc_result = SimCResult(
            dps=50000.0,
            raw_output="Test output",
            errors=[],
            is_valid=True
        )
        
        program = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1,
            parent_id="parent_123",
            diff_applied="test diff",
            reasoning="test reasoning",
            simc_result=simc_result
        )
        
        # Check all fields are accessible
        assert hasattr(program, 'apl_code')
        assert hasattr(program, 'dps_score')
        assert hasattr(program, 'generation')
        assert hasattr(program, 'timestamp')
        assert hasattr(program, 'parent_id')
        assert hasattr(program, 'diff_applied')
        assert hasattr(program, 'reasoning')
        assert hasattr(program, 'simc_result')
        assert hasattr(program, 'program_id')
