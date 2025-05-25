import pytest
import tempfile
import os
from evosim.core.programs import APLProgram


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
    
    def test_program_with_metadata(self):
        """Test program creation with all metadata fields."""
        program = APLProgram(
            apl_code="actions=test",
            dps_score=45000.0,
            generation=2,
            parent_id="parent_123",
            diff_applied="+ actions.precombat=flask",
            reasoning="Added flask for better DPS",
            evaluation_metadata={"sim_time": 300}
        )
        
        assert program.parent_id == "parent_123"
        assert program.diff_applied == "+ actions.precombat=flask"
        assert program.reasoning == "Added flask for better DPS"
        assert program.evaluation_metadata["sim_time"] == 300


