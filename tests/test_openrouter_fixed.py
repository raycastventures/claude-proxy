import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio

from providers.openrouter import OpenRouterProvider
from models import AnthropicRequest, AnthropicMessage


class TestOpenRouterFixed:
    """Test case to verify OpenRouter works with OpenAI SDK implementation"""
    
    def test_openrouter_inherits_from_openai(self):
        """Test that OpenRouter properly inherits from OpenAI provider"""
        # Create OpenRouter provider with model variants
        provider = OpenRouterProvider(
            api_key="test-key", 
            base_url="https://openrouter.ai/api",
            model_variants=[{"model": "anthropic/claude-3-sonnet:20240229"}]
        )
        
        # Verify provider attributes
        assert provider.provider_name == "openrouter"
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://openrouter.ai/api"
        assert len(provider.model_variants) == 1
        assert provider.model_variants[0]["model"] == "anthropic/claude-3-sonnet:20240229"
        
        # Verify it has the OpenAI client
        assert hasattr(provider, 'client')
        assert provider.client is not None
        
        # Verify it has required methods from base class
        assert hasattr(provider, 'forward_request')
        assert hasattr(provider, '_convert_anthropic_to_openai')
        assert hasattr(provider, '_handle_openai_response')
        
        print("✅ Test passed! OpenRouter correctly inherits from OpenAI provider")
        print(f"Provider name: {provider.provider_name}")
        print(f"Base URL: {provider.base_url}")
        print(f"Model variants: {provider.model_variants}")
    
    def test_openrouter_configuration(self):
        """Test various OpenRouter configurations"""
        # Test with no model variants
        provider1 = OpenRouterProvider(api_key="test-key")
        assert provider1.model_variants == []
        
        # Test with multiple model variants
        variants = [
            {"model": "anthropic/claude-3-sonnet:20240229"},
            {"model": "anthropic/claude-3-haiku:20240307"},
            {"model": "openai/gpt-4"}
        ]
        provider2 = OpenRouterProvider(
            api_key="test-key",
            base_url="https://openrouter.ai/api",
            model_variants=variants
        )
        assert len(provider2.model_variants) == 3
        
        # Test with custom base URL
        provider3 = OpenRouterProvider(
            api_key="test-key",
            base_url="https://custom.openrouter.ai"
        )
        assert provider3.base_url == "https://custom.openrouter.ai"
        
        print("✅ Configuration test passed!")


if __name__ == "__main__":
    # Run tests
    test = TestOpenRouterFixed()
    print("Running inheritance test...")
    test.test_openrouter_inherits_from_openai()
    print("\nRunning configuration test...")
    test.test_openrouter_configuration()