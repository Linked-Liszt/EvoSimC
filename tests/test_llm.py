"""
Test the simplified LLM wrapper functionality.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from evosim.llm import (
    GeminiLLMClient,
    LLMUsageStats,
    LLMGenerationError,
    create_llm_client
)


class TestLLMUsageStats:
    """Test basic usage statistics tracking."""
    
    def test_usage_stats_creation(self):
        """Test creating and using usage stats."""
        stats = LLMUsageStats()
        
        assert stats.total_requests == 0
        assert stats.failed_requests == 0
        
        # Add some requests
        stats.add_request(failed=False)
        stats.add_request(failed=False)
        stats.add_request(failed=True)
        
        summary = stats.get_summary()
        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["failed_requests"] == 1
        assert summary["success_rate"] == 2/3


class TestGeminiLLMClient:
    """Test Gemini LLM client functionality."""
    
    def test_client_creation_without_google_module(self):
        """Test that client raises error when google module not available."""
        # Mock the import to fail
        with patch.dict(sys.modules, {'google.generativeai': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'google'")):
                with pytest.raises(LLMGenerationError, match="google-generativeai package not found"):
                    GeminiLLMClient(api_key="test_key")
    
    def test_gemini_client_initialization(self):
        """Test GeminiLLMClient initialization."""
        # Create a mock google.generativeai module
        mock_genai = MagicMock()
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Patch the import in the LLM module
        with patch.dict(sys.modules, {'google.generativeai': mock_genai}):
            client = GeminiLLMClient(api_key="test_key")
            
            assert client.model_name == "gemini-1.5-pro"
            assert client.api_key == "test_key"
            mock_genai.configure.assert_called_once_with(api_key="test_key")
    
    def test_gemini_generation_success(self):
        """Test successful Gemini generation."""
        # Create a mock google.generativeai module
        mock_genai = MagicMock()
        
        # Mock the response structure
        mock_part = Mock()
        mock_part.text = "This is a test response with diff format:\n\n-<<<<<<< SEARCH\nold_code\n=======\nnew_code\n>>>>>>> REPLACE"
        
        mock_content = Mock()
        mock_content.parts = [mock_part]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model._safety_settings = []
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict(sys.modules, {'google.generativeai': mock_genai}):
            client = GeminiLLMClient()
            response = client.generate("test prompt")
            
            assert isinstance(response, str)
            assert "SEARCH" in response
            assert "REPLACE" in response
            
            # Check usage stats were updated
            stats = client.get_usage_stats()
            assert stats["total_requests"] == 1
            assert stats["successful_requests"] == 1

    def test_gemini_generation_failure(self):
        """Test failed Gemini generation."""
        # Create a mock google.generativeai module
        mock_genai = MagicMock()
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model._safety_settings = []
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict(sys.modules, {'google.generativeai': mock_genai}):
            client = GeminiLLMClient()
            
            with pytest.raises(LLMGenerationError, match="Gemini generation failed"):
                client.generate("test prompt")
            
            # Check usage stats were updated
            stats = client.get_usage_stats()
            assert stats["total_requests"] == 1
            assert stats["failed_requests"] == 1
            assert stats["success_rate"] == 0.0


class TestLLMClientFactory:
    """Test the LLM client factory function."""
    
    def test_create_gemini_client(self):
        """Test creating Gemini client."""
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = Mock()
        
        with patch.dict(sys.modules, {'google.generativeai': mock_genai}):
            client = create_llm_client("gemini", api_key="test")
            
            assert isinstance(client, GeminiLLMClient)
            assert client.model_name == "gemini-1.5-pro"
    
    def test_create_invalid_provider(self):
        """Test creating client with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("invalid_provider")


class TestIntegrationWithPromptSampler:
    """Test integration with the existing prompt sampler."""
    
    def test_integration_flow(self):
        """Test the basic integration flow as specified in the user request."""
        from evosim.core import APLProgram, create_prompt_sampler
        
        # Create a mock google.generativeai module
        mock_genai = MagicMock()
        
        # Mock the Gemini response
        mock_part = Mock()
        mock_part.text = """I'll improve this APL:

-<<<<<<< SEARCH
actions=spell1
actions+=/spell2
=======
actions=spell1,if=cooldown.ready
actions+=/spell3,if=buff.up
actions+=/spell2
>>>>>>> REPLACE

This adds better cooldown and buff management."""
        
        mock_content = Mock()
        mock_content.parts = [mock_part]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model._safety_settings = []
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict(sys.modules, {'google.generativeai': mock_genai}):
            # Create test data
            parent_program = APLProgram(
                apl_code="actions.precombat=flask\nactions=spell1\nactions+=/spell2",
                dps_score=50000.0,
                generation=1
            )
            
            inspirations = [
                APLProgram(
                    apl_code="actions.precombat=flask,food\nactions=spell3\nactions+=/spell1",
                    dps_score=52000.0,
                    generation=1
                )
            ]
            
            # Create components
            prompt_sampler = create_prompt_sampler("basic")
            llm_client = create_llm_client("gemini")
            
            # Test the flow: 
            # prompt = prompt_sampler.build(parent_program, inspirations)
            prompt = prompt_sampler.build(parent_program, inspirations)
            assert len(prompt) > 100  # Should be a substantial prompt
            
            # diff = llm.generate(prompt)
            diff = llm_client.generate(prompt)
            assert "SEARCH" in diff
            assert "REPLACE" in diff
            
            # child_program = apply_diff(parent_program, diff)
            child_apl_code = prompt_sampler.apply_diff(parent_program.apl_code, diff)
            assert child_apl_code != parent_program.apl_code  # Should be different
            
            # Verify usage tracking
            stats = llm_client.get_usage_stats()
            assert stats["total_requests"] == 1
            assert stats["successful_requests"] == 1
