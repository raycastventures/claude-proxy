import json
import time
from typing import Dict, Any, Optional, List
import httpx
from models import AnthropicRequest, AnthropicResponse, AnthropicContentBlock, AnthropicUsage


class OpenRouterProvider:
    """OpenRouter provider implementation"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=None)  # No timeout
    
    def name(self) -> str:
        return "openrouter"
    
    async def forward_request(self, request: AnthropicRequest, **kwargs) -> Dict[str, Any]:
        """Forward Anthropic request to OpenRouter (OpenAI-compatible API)"""
        
        # Get spinner callback if provided (OpenRouter doesn't use variants but accepts for consistency)
        spinner_callback = kwargs.get('spinner_callback')
        
        # Convert Anthropic request to OpenAI format
        openai_request = self._convert_anthropic_to_openai(request)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            if request.stream:
                # For streaming, we'll handle it differently
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/chat/completions",
                    headers=headers,
                    json=openai_request
                ) as response:
                    return await self._handle_openrouter_streaming(response, request)
            else:
                response = await self.client.post(
                    f"{self.base_url}/api/v1/chat/completions",
                    headers=headers,
                    json=openai_request
                )
                return await self._handle_openrouter_response(response, request)
                
        except httpx.HTTPStatusError as e:
            # Handle specific HTTP errors
            if e.response.status_code == 429:
                raise Exception("429: Rate limit exceeded")
            elif e.response.status_code == 401:
                raise Exception("401: Unauthorized - check API key")
            elif e.response.status_code == 400:
                raise Exception(f"400: Bad request - {e.response.text}")
            else:
                raise Exception(f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    def _convert_anthropic_to_openai(self, request: AnthropicRequest) -> Dict[str, Any]:
        """Convert Anthropic request format to OpenAI format"""
        openai_request = {
            "model": self._convert_model_name(request.model),
            "messages": [],
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        # Convert messages
        for msg in request.messages:
            if isinstance(msg.content, str):
                openai_request["messages"].append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                # Handle content blocks
                content = []
                for block in msg.content:
                    if block.get("type") == "text":
                        content.append(block)
                openai_request["messages"].append({
                    "role": msg.role,
                    "content": content
                })
        
        # Add temperature if present
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        
        # Add system message if present
        if request.system:
            system_content = "\n\n".join([s.text for s in request.system])
            openai_request["messages"].insert(0, {
                "role": "system",
                "content": system_content
            })
        
        return openai_request
    
    def _convert_model_name(self, anthropic_model: str) -> str:
        """Convert Anthropic model name to OpenRouter model name"""
        model_mapping = {
            "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet:20240229",
            "claude-3-opus-20240229": "anthropic/claude-3-opus:20240229",
            "claude-3-haiku-20240307": "anthropic/claude-3-haiku:20240307"
        }
        return model_mapping.get(anthropic_model, anthropic_model)
    
    async def _handle_openrouter_response(self, response: httpx.Response, request: AnthropicRequest) -> Dict[str, Any]:
        """Handle non-streaming OpenRouter response"""
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        choice = response_data['choices'][0]
        message = choice['message']
        
        # Convert OpenAI response to Anthropic format
        content_blocks = []
        if isinstance(message['content'], str):
            content_blocks.append(AnthropicContentBlock(
                type="text",
                text=message['content']
            ))
        elif isinstance(message['content'], list):
            for block in message['content']:
                if block.get('type') == 'text':
                    content_blocks.append(AnthropicContentBlock(
                        type="text",
                        text=block['text']
                    ))
        
        usage = AnthropicUsage(
            input_tokens=response_data.get('usage', {}).get('prompt_tokens', 0),
            output_tokens=response_data.get('usage', {}).get('completion_tokens', 0)
        )
        
        return {
            "content": content_blocks,
            "id": response_data.get('id', ''),
            "model": request.model,
            "role": "assistant",
            "stop_reason": choice.get('finish_reason', 'stop'),
            "type": "message",
            "usage": usage.dict()
        }
    
    async def _handle_openrouter_streaming(self, response: httpx.Response, request: AnthropicRequest) -> Dict[str, Any]:
        """Handle streaming OpenRouter response"""
        # This is a simplified streaming implementation
        # In a real implementation, you'd want to properly stream the response
        full_text = ""
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and chunk['choices']:
                        choice = chunk['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            if content:
                                full_text += content
                except json.JSONDecodeError:
                    continue
        
        content_blocks = [AnthropicContentBlock(type="text", text=full_text)]
        
        return {
            "content": content_blocks,
            "id": "streaming_response",
            "model": request.model,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 0, "output_tokens": 0}
        } 