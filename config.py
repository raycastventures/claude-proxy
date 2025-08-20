import os
import yaml
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from models import Config, ServerConfig, ProvidersConfig, RoutingConfig, BedrockProviderConfig, OpenRouterProviderConfig, CerebrasProviderConfig, GroqProviderConfig


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file and environment variables"""
    
    # Load .env file if it exists
    load_dotenv()
    
    # Default config path
    if config_path is None:
        config_path = "config.yaml"
    
    config_data = {}
    
    # Load YAML config if it exists
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Environment variable overrides
    env_overrides = {
        "server": {
            "port": os.getenv("PORT", "8000"),
            "read_timeout": os.getenv("READ_TIMEOUT", "60s"),
            "write_timeout": os.getenv("WRITE_TIMEOUT", "60s"),
            "idle_timeout": os.getenv("IDLE_TIMEOUT", "120s"),
        },
        "providers": {
            "bedrock": {
                "region": os.getenv("BEDROCK_REGION", "us-east-1"),
                "endpoint": os.getenv("BEDROCK_ENDPOINT"),
            },
            "openrouter": {
                "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai"),
                "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            },
            "cerebras": {
                "base_url": os.getenv("CEREBRAS_BASE_URL", "https://inference.cerebras.ai"),
                "api_key": os.getenv("CEREBRAS_API_KEY", ""),
            },
            "groq": {
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                "api_key": os.getenv("GROQ_API_KEY", ""),
            }
        },
        "routing": {
            "enable": os.getenv("ROUTING_ENABLE", "false").lower() == "true",
            "retry_timeout_millis": int(os.getenv("RETRY_TIMEOUT_MILLIS", "1000")),
            "rate_limit_seconds": int(os.getenv("RATE_LIMIT_SECONDS", "60")),
        }
    }
    
    # Merge YAML config with environment overrides
    if "server" in config_data:
        env_overrides["server"].update(config_data["server"])
    if "providers" in config_data:
        env_overrides["providers"].update(config_data["providers"])
    if "routing" in config_data:
        env_overrides["routing"].update(config_data["routing"])
    
    # Parse provider sequence from YAML if routing is enabled
    model_routing_configs = []
    if config_data.get("routing", {}).get("enable", False):
        for model_route_data in config_data["routing"].get("models", []):
            model_route = {
                "model": model_route_data["model"],
                "provider_sequence": []
            }
            
            # Parse provider sequence for this model
            for route_data in model_route_data.get("provider_sequence", []):
                route = {
                    "name": route_data["name"],
                    "variants": route_data.get("variants", [])
                }
                model_route["provider_sequence"].append(route)
            
            model_routing_configs.append(model_route)
    
    # Build final config - providers are now determined by routing config
    config = Config(
        server=ServerConfig(**env_overrides["server"]),
        providers=ProvidersConfig(
            bedrock=BedrockProviderConfig(**env_overrides["providers"]["bedrock"]),
            openrouter=OpenRouterProviderConfig(**env_overrides["providers"]["openrouter"]),
            cerebras=CerebrasProviderConfig(**env_overrides["providers"]["cerebras"]),
            groq=GroqProviderConfig(**env_overrides["providers"]["groq"])
        ),
        routing=RoutingConfig(
            enable=env_overrides["routing"]["enable"],
            models=model_routing_configs,
            retry_timeout_millis=env_overrides["routing"]["retry_timeout_millis"],
            rate_limit_seconds=env_overrides["routing"]["rate_limit_seconds"]
        )
    )
    
    return config
 