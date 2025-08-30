import logging
from typing import Dict, Any, Optional, List
from .openai import OpenAIProvider

# Configure logging
logger = logging.getLogger(__name__)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider implementation inheriting from OpenAIProvider"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api", model_variants: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model_variants=model_variants,
            provider_name="openrouter"
        )