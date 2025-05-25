"""
Lightweight LLM client for EvoSimC evolutionary system.

This module provides a simple interface for interacting with Gemini LLM
for generating APL optimization diffs.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass 
class LLMUsageStats:
    """Basic LLM usage tracking."""
    total_requests: int = 0
    failed_requests: int = 0
    
    def add_request(self, failed: bool = False):
        """Add a request to the usage statistics."""
        self.total_requests += 1
        if failed:
            self.failed_requests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of usage statistics."""
        success_rate = (self.total_requests - self.failed_requests) / max(1, self.total_requests)
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.total_requests - self.failed_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate
        }


class LLMGenerationError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class GeminiLLMClient:
    """
    Simple Google Gemini LLM client.
    
    Requires google-generativeai package to be installed.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self.usage_stats = LLMUsageStats()
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                # Will try to use GOOGLE_API_KEY environment variable
                genai.configure()
            
            # Configure generation parameters
            generation_config = {
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.8),
                "top_k": self.config.get("top_k", 40),
                "max_output_tokens": self.config.get("max_output_tokens", 8192),
            }
            
            # Configure safety settings to be permissive for code generation
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Initialized Gemini client with model: {self.model_name}")
            
        except ImportError:
            raise LLMGenerationError(
                "google-generativeai package not found. Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize Gemini client: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters that override defaults
            
        Returns:
            Generated content as string
            
        Raises:
            LLMGenerationError: If generation fails
        """
        if not self._client:
            raise LLMGenerationError("Gemini client not initialized")
        
        try:
            # Override any generation config parameters
            if kwargs:
                generation_config = {
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                    "top_p": kwargs.get("top_p", self.config.get("top_p", 0.8)),
                    "top_k": kwargs.get("top_k", self.config.get("top_k", 40)),
                    "max_output_tokens": kwargs.get("max_output_tokens", 
                                                   self.config.get("max_output_tokens", 8192)),
                }
                
                # Create a temporary model with updated config
                import google.generativeai as genai
                temp_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=self._client._safety_settings
                )
                response = temp_model.generate_content(prompt)
            else:
                response = self._client.generate_content(prompt)
            
            # Extract response content
            if response.candidates and response.candidates[0].content.parts:
                content = response.candidates[0].content.parts[0].text
            else:
                raise LLMGenerationError("No content in Gemini response")
            
            # Update usage statistics
            self.usage_stats.add_request(failed=False)
            
            logger.debug(f"Gemini generation successful")
            
            return content
            
        except Exception as e:
            self.usage_stats.add_request(failed=True)
            logger.error(f"Gemini generation failed: {e}")
            raise LLMGenerationError(f"Gemini generation failed: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.get_summary()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = LLMUsageStats()


# Factory function for creating the client
def create_llm_client(provider: str = "gemini", model_name: str = None, **kwargs) -> GeminiLLMClient:
    """
    Create an LLM client for use in the evolutionary system.
    
    Args:
        provider: LLM provider (currently only "gemini" supported)
        model_name: Model name (defaults to "gemini-1.5-pro")
        **kwargs: Additional configuration parameters
        
    Returns:
        GeminiLLMClient instance ready for use
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "gemini":
        default_model = "gemini-1.5-pro"
        return GeminiLLMClient(model_name or default_model, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Available: 'gemini'")