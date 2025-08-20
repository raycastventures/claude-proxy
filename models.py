from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AnthropicContentBlock(BaseModel):
    type: str
    text: str


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    service_tier: Optional[str] = None


class AnthropicResponse(BaseModel):
    content: List[AnthropicContentBlock]
    id: str
    model: str
    role: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    type: str
    usage: AnthropicUsage


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

    def get_content_blocks(self) -> List[AnthropicContentBlock]:
        if isinstance(self.content, str):
            return [AnthropicContentBlock(type="text", text=self.content)]
        elif isinstance(self.content, list):
            blocks = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        blocks.append(AnthropicContentBlock(type="text", text=text))
            return blocks
        return []


class AnthropicSystemMessage(BaseModel):
    text: str
    type: str
    cache_control: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    temperature: Optional[float] = None
    system: Optional[List[AnthropicSystemMessage]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None


class ProviderVariant(BaseModel):
    model: str
    region: Optional[str] = None
    endpoint: Optional[str] = None


class ProviderRoute(BaseModel):
    name: str
    variants: List[ProviderVariant]


class ModelRoutingConfig(BaseModel):
    model: str
    provider_sequence: List[ProviderRoute]


class RoutingConfig(BaseModel):
    enable: bool = False
    models: List[ModelRoutingConfig] = []
    retry_timeout_millis: int = 1000
    rate_limit_seconds: int = 60


class BedrockProviderConfig(BaseModel):
    region: str
    endpoint: Optional[str] = None


class OpenRouterProviderConfig(BaseModel):
    base_url: str
    api_key: str


class CerebrasProviderConfig(BaseModel):
    api_key: str
    base_url: Optional[str] = None


class GroqProviderConfig(BaseModel):
    api_key: str
    base_url: Optional[str] = None


class ProvidersConfig(BaseModel):
    bedrock: Optional[BedrockProviderConfig] = None
    openrouter: Optional[OpenRouterProviderConfig] = None
    cerebras: Optional[CerebrasProviderConfig] = None
    groq: Optional[GroqProviderConfig] = None


class ServerConfig(BaseModel):
    port: str = "8000"
    read_timeout: str = "60s"
    write_timeout: str = "60s"
    idle_timeout: str = "120s"


class Config(BaseModel):
    server: ServerConfig
    providers: ProvidersConfig
    routing: RoutingConfig


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {
            "id": "claude-3-sonnet-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic"
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic"
        },
        {
            "id": "claude-3-haiku-20240307",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic"
        }
    ])


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


class StreamingEvent(BaseModel):
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    content_block: Optional[Dict[str, Any]] = None 