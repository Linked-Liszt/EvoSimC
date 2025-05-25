"""
LLM module for EvoSimC evolutionary system.

This module provides a lightweight wrapper for interacting with Gemini LLM
in the context of evolutionary APL optimization.
"""

from .llm import (
    GeminiLLMClient,
    LLMUsageStats,
    LLMGenerationError,
    create_llm_client
)

__all__ = [
    "GeminiLLMClient",
    "LLMUsageStats", 
    "LLMGenerationError",
    "create_llm_client"
]
