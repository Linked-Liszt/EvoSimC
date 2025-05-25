"""
Test the prompt sampler functionality.
"""

import pytest
from evosim.core.programs import APLProgram
from evosim.core.prompt_sampler import (
    PromptSampler, 
    PromptConfig, 
    AdaptivePromptSampler,
    create_prompt_sampler,
    DiffApplicationError
)


class TestPromptSampler:
    """Test basic prompt sampler functionality."""
    
    def test_basic_prompt_creation(self):
        """Test creating a basic prompt with parent and inspirations."""
        sampler = PromptSampler()
        
        parent = APLProgram(
            apl_code="actions.precombat=flask\nactions=spell1\nactions+=/spell2,if=buff.up",
            dps_score=50000.0,
            generation=2,
            reasoning="Basic rotation"
        )
        
        inspirations = [
            APLProgram(
                apl_code="actions.precombat=flask,food\nactions=spell1\nactions+=/spell3,if=cooldown.ready",
                dps_score=52000.0,
                generation=1,
                reasoning="Added food buff"
            ),
            APLProgram(
                apl_code="actions.precombat=flask\nactions=spell2,if=buff.down\nactions+=/spell1",
                dps_score=51500.0,
                generation=1,
                reasoning="Reordered priorities"
            )
        ]
        
        prompt = sampler.build(parent, inspirations)
        
        # Check that key components are included
        assert "Act as an expert SimulationCraft APL" in prompt
        assert "50,000.0" in prompt  # Parent DPS
        assert "52,000.0" in prompt  # Inspiration 1 DPS  
        assert "51,500.0" in prompt  # Inspiration 2 DPS
        assert "actions.precombat=flask" in prompt  # Parent code
        assert "SEARCH" in prompt and "REPLACE" in prompt  # Diff instructions
        assert "Basic rotation" in prompt  # Parent reasoning
        assert "Added food buff" in prompt  # Inspiration reasoning
    
    def test_prompt_config_options(self):
        """Test different prompt configuration options."""
        config = PromptConfig(
            include_reasoning=False,
            include_evaluation_metadata=False,
            diff_format_instructions=False,
            max_inspirations=1
        )
        sampler = PromptSampler(config)
        
        parent = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1,
            reasoning="Should not appear",
            evaluation_metadata={"sim_time": 300}
        )
        
        inspirations = [
            APLProgram(
                apl_code="actions=inspiration1",
                dps_score=51000.0,
                generation=1,
                reasoning="Should not appear"
            ),
            APLProgram(
                apl_code="actions=inspiration2", 
                dps_score=52000.0,
                generation=1
            )
        ]
        
        prompt = sampler.build(parent, inspirations)
        
        # Should exclude reasoning and metadata
        assert "Should not appear" not in prompt
        assert "sim_time" not in prompt
        assert "SEARCH" not in prompt  # No diff instructions
        
        # Should only include 1 inspiration
        assert "inspiration1" in prompt
        assert "inspiration2" not in prompt
    
    def test_empty_inspirations(self):
        """Test handling of empty inspirations list."""
        sampler = PromptSampler()
        
        parent = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1
        )
        
        prompt = sampler.build(parent, [])
        
        assert "No inspiration programs available" in prompt
        assert "50,000.0" in prompt  # Parent DPS should still be there
    
    def test_adaptive_sampler(self):
        """Test adaptive prompt sampler."""
        sampler = AdaptivePromptSampler()
        
        # Simulate low improvement scenario
        sampler.update_generation_stats(1, 50000, 0.005)
        sampler.update_generation_stats(2, 50100, 0.008)
        sampler.update_generation_stats(3, 50150, 0.007)
        
        parent = APLProgram(
            apl_code="actions=test",
            dps_score=50200.0,
            generation=4
        )
        
        prompt = sampler.build(parent, [])
        
        assert "Adaptive Strategy" in prompt
        assert "Recent progress has been slow" in prompt
        assert "more significant changes" in prompt


class TestDiffApplication:
    """Test the apply_diff functionality."""
    
    def test_simple_diff_application(self):
        """Test applying a simple SEARCH/REPLACE diff."""
        sampler = PromptSampler()
        
        original_code = """actions.precombat=flask
actions=spell1
actions+=/spell2,if=buff.up"""
        
        llm_response = """Here's my improvement:

-<<<<<<< SEARCH
actions=spell1
actions+=/spell2,if=buff.up
=======
actions=spell1,if=cooldown.spell3.ready
actions+=/spell3,if=cooldown.spell3.ready
actions+=/spell2,if=buff.up
>>>>>>> REPLACE

This adds spell3 with proper cooldown management."""
        
        result = sampler.apply_diff(original_code, llm_response)
        
        expected = """actions.precombat=flask
actions=spell1,if=cooldown.spell3.ready
actions+=/spell3,if=cooldown.spell3.ready
actions+=/spell2,if=buff.up"""
        
        assert result == expected
    
    def test_multiple_diff_blocks(self):
        """Test applying multiple SEARCH/REPLACE blocks."""
        sampler = PromptSampler()
        
        original_code = """actions.precombat=flask
actions=spell1
actions+=/spell2"""
        
        llm_response = """Multiple improvements:

-<<<<<<< SEARCH
actions.precombat=flask
=======
actions.precombat=flask,food
>>>>>>> REPLACE

-<<<<<<< SEARCH
actions+=/spell2
=======
actions+=/spell2,if=target.health.pct>20
>>>>>>> REPLACE"""
        
        result = sampler.apply_diff(original_code, llm_response)
        
        expected = """actions.precombat=flask,food
actions=spell1
actions+=/spell2,if=target.health.pct>20"""
        
        assert result == expected
    
    def test_whitespace_normalization(self):
        """Test diff application with whitespace differences."""
        sampler = PromptSampler()
        
        original_code = """actions.precombat=flask
actions=spell1
actions+=/spell2,if=buff.up"""
        
        # LLM response with slightly different whitespace
        llm_response = """
-<<<<<<< SEARCH
actions=spell1
actions+=/spell2,if=buff.up
=======
actions=spell1,if=cooldown.ready
actions+=/spell2,if=buff.up
>>>>>>> REPLACE
"""
        
        result = sampler.apply_diff(original_code, llm_response)
        
        expected = """actions.precombat=flask
actions=spell1,if=cooldown.ready
actions+=/spell2,if=buff.up"""
        
        assert result == expected
    
    def test_diff_application_error_no_blocks(self):
        """Test error when no diff blocks found."""
        sampler = PromptSampler()
        
        original_code = "actions=spell1"
        llm_response = "Just some text without any diff blocks"
        
        with pytest.raises(DiffApplicationError, match="No valid SEARCH/REPLACE blocks found"):
            sampler.apply_diff(original_code, llm_response)
    
    def test_diff_application_error_search_not_found(self):
        """Test error when search text not found."""
        sampler = PromptSampler()
        
        original_code = "actions=spell1"
        llm_response = """
-<<<<<<< SEARCH
actions=nonexistent_spell
=======
actions=spell2
>>>>>>> REPLACE
"""
        
        with pytest.raises(DiffApplicationError, match="Search text not found"):
            sampler.apply_diff(original_code, llm_response)
    
    def test_diff_application_error_multiple_matches(self):
        """Test error when search text appears multiple times."""
        sampler = PromptSampler()
        
        original_code = """actions=spell1
actions+=/spell1
actions+=/spell2"""
        
        llm_response = """
-<<<<<<< SEARCH
spell1
=======
spell3
>>>>>>> REPLACE
"""
        
        with pytest.raises(DiffApplicationError, match="appears 2 times"):
            sampler.apply_diff(original_code, llm_response)
    
    def test_empty_search_text(self):
        """Test error when search text is empty."""
        sampler = PromptSampler()
        
        original_code = "actions=spell1"
        llm_response = """
-<<<<<<< SEARCH

=======
actions=spell2
>>>>>>> REPLACE
"""
        
        with pytest.raises(DiffApplicationError, match="Empty search text"):
            sampler.apply_diff(original_code, llm_response)


class TestPromptSamplerFactory:
    """Test the factory function for creating prompt samplers."""
    
    def test_create_basic_sampler(self):
        """Test creating basic sampler."""
        sampler = create_prompt_sampler("basic", include_reasoning=False)
        
        assert isinstance(sampler, PromptSampler)
        assert not sampler.config.include_reasoning
    
    def test_create_adaptive_sampler(self):
        """Test creating adaptive sampler."""
        sampler = create_prompt_sampler("adaptive", temperature_instructions=False)
        
        assert isinstance(sampler, AdaptivePromptSampler)
        assert not sampler.config.temperature_instructions
    
    def test_invalid_sampler_type(self):
        """Test creating sampler with invalid type."""
        with pytest.raises(ValueError, match="Unknown sampler type"):
            create_prompt_sampler("invalid_type")


class TestPromptContent:
    """Test the actual content and structure of generated prompts."""
    
    def test_prompt_structure(self):
        """Test that prompts have expected sections in order."""
        sampler = PromptSampler()
        
        parent = APLProgram(
            apl_code="actions=test",
            dps_score=50000.0,
            generation=1
        )
        
        inspirations = [
            APLProgram(
                apl_code="actions=inspiration",
                dps_score=52000.0,
                generation=1
            )
        ]
        
        prompt = sampler.build(parent, inspirations)
        sections = prompt.split("##")
        
        # Should have multiple sections
        assert len(sections) >= 4
        
        # Check section order (removing the first empty section)
        section_titles = [s.strip().split("\n")[0].strip() for s in sections[1:]]
        
        expected_sections = [
            "Context: APL Optimization",
            "High-Performing Programs (Inspirations)", 
            "Current Program to Improve",
            "Instructions: Generating Improvements",
            "Task"
        ]
        
        # Check that key sections are present
        for expected in expected_sections:
            assert any(expected in title for title in section_titles), f"Missing section: {expected}"
    
    def test_apl_code_formatting(self):
        """Test that APL code is properly formatted in prompts."""
        sampler = PromptSampler()
        
        apl_code = "actions.precombat=flask,food\nactions=spell1,if=buff.up\nactions+=/spell2"
        
        parent = APLProgram(
            apl_code=apl_code,
            dps_score=50000.0,
            generation=1
        )
        
        prompt = sampler.build(parent, [])
        
        # APL code should be in code blocks
        assert f"```\n{apl_code}\n```" in prompt
    
    def test_dps_formatting(self):
        """Test that DPS values are properly formatted."""
        sampler = PromptSampler()
        
        parent = APLProgram(
            apl_code="actions=test",
            dps_score=123456.789,
            generation=1
        )
        
        prompt = sampler.build(parent, [])
        
        # DPS should be formatted with commas and one decimal place
        assert "123,456.8" in prompt
