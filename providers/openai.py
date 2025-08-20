import json
import time
import logging
from typing import Dict, Any, Optional, List
import openai
from models import AnthropicRequest, AnthropicResponse, AnthropicContentBlock, AnthropicUsage

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Generic OpenAI-compatible provider implementation with variant-based fallback"""
    
    def __init__(self, api_key: str, base_url: str, model_variants: Optional[List[Dict[str, Any]]] = None, provider_name: str = "openai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_variants = model_variants or []
        self.provider_name = provider_name
        
        # Initialize OpenAI client with no automatic retries
        self.client = openai.OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            max_retries=0  # Disable automatic retries so we can handle fallback ourselves
        )

        logging.getLogger("openai").setLevel(logging.WARN)
        logging.getLogger("httpx").setLevel(logging.WARN)
        logging.getLogger('botocore').setLevel(logging.WARN)

    
    def name(self) -> str:
        return self.provider_name
    
    async def forward_request(self, request: AnthropicRequest, **kwargs) -> Dict[str, Any]:
        """Forward Anthropic request to OpenAI-compatible API, trying variants in order"""
        
        # Get spinner callback if provided
        spinner_callback = kwargs.get('spinner_callback')
        
        if not self.model_variants:
            raise Exception(f"No {self.provider_name} model variants configured")
        
        last_error: Optional[str] = None
        
        # Try each variant in sequence
        for index, variant in enumerate(self.model_variants):
            target_model = variant.get("model") if isinstance(variant, dict) else variant
            if not target_model:
                logger.warning(f"⚠️ Skipping invalid {self.provider_name} variant at index {index}: {variant}")
                continue
            
            # Update spinner with current variant being tried
            if spinner_callback:
                variant_info = target_model
                spinner_callback(variant_info)
            
            openai_request = self._convert_anthropic_to_openai(request, target_model)
            
            try:
                if request.stream:
                    # For now, handle streaming requests as non-streaming to get content
                    # TODO: Implement proper streaming response handling
                    openai_request["stream"] = False  # Force non-streaming
                    response = self.client.chat.completions.create(**openai_request)
                    result = await self._handle_openai_response(response, request)
                    # Add metadata about which variant was used
                    result['actual_model'] = target_model
                    return result
                else:
                    response = self.client.chat.completions.create(**openai_request)
                    result = await self._handle_openai_response(response, request)
                    # Add metadata about which variant was used
                    result['actual_model'] = target_model
                    return result
            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)
                
                if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                    logger.warning(f"⚠️ {self.provider_name} rate limit (429) on model '{target_model}', trying next variant...")
                    continue
                elif "unauthorized" in error_str or "invalid api key" in error_str or "401" in error_str:
                    logger.error(f"❌ {self.provider_name} auth error; aborting fallback chain")
                    raise Exception("401: Unauthorized - check API key")
                elif "400" in error_str or "bad request" in error_str:
                    logger.warning(f"⚠️ {self.provider_name} bad request (400) on model '{target_model}': {e}")
                    # Try next variant for 400 errors (schema compatibility issues)
                    if index < len(self.model_variants) - 1:
                        continue
                    else:
                        break
                else:
                    logger.warning(f"⚠️ {self.provider_name} error on '{target_model}': {e}")
                    # Try next variant if available
                    if index < len(self.model_variants) - 1:
                        continue
                    else:
                        break
        
        raise Exception(f"All {self.provider_name} variants failed: {last_error}")
    
    def _convert_anthropic_to_openai(self, request: AnthropicRequest, model_name: str) -> Dict[str, Any]:
        """Convert Anthropic request format to OpenAI format, using provided model_name"""
        openai_request = {
            "model": model_name,
            "messages": [],
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        
        # Convert messages
        for msg in request.messages:
            if isinstance(msg.content, str):
                openai_request["messages"].append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                # Handle content blocks - extract text content
                text_parts = []
                for block in msg.content:
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block.get("text"))
                
                # Join all text parts into a single string
                if text_parts:
                    combined_text = "\n".join(text_parts)
                    openai_request["messages"].append({
                        "role": msg.role,
                        "content": combined_text
                    })
                else:
                    # If no text content found, skip this message or use empty string
                    openai_request["messages"].append({
                        "role": msg.role,
                        "content": ""
                    })
        
        # Add temperature if present (but check if model supports it like Go implementation)
        if request.temperature is not None:
            # Check if this is an o-series model (they don't support temperature)
            is_o_series_model = request.model.startswith("o1") or request.model.startswith("o3")
            if not is_o_series_model:
                openai_request["temperature"] = request.temperature
        
        # Add system message if present
        if request.system:
            system_content = "\n\n".join([s.text for s in request.system])
            openai_request["messages"].insert(0, {
                "role": "system",
                "content": system_content
            })
        
        # Convert Anthropic tools to OpenAI format (like Go implementation)
        if request.tools:
            tools = []
            for tool in request.tools:
                # Ensure tool has required fields
                if not tool.name:
                    logger.warning(f"⚠️ Skipping tool with empty name: {tool}")
                    continue
                
                # Build parameters with error checking
                parameters = {
                    "type": tool.input_schema.get("type", "object")  # Default to object type
                }
                
                # Handle properties
                if tool.input_schema.get("properties"):
                    # Fix array properties that are missing items field (like Go implementation)
                    fixed_properties = {}
                    for prop_name, prop_value in tool.input_schema["properties"].items():
                        if isinstance(prop_value, dict):
                            prop = prop_value.copy()
                            # Check if this is an array type missing items
                            if prop.get("type") == "array" and "items" not in prop:
                                # Add default items definition for arrays
                                prop["items"] = {"type": "string"}
                            fixed_properties[prop_name] = prop
                        else:
                            fixed_properties[prop_name] = prop_value
                    parameters["properties"] = fixed_properties
                else:
                    parameters["properties"] = {}
                
                # Handle required fields
                if tool.input_schema.get("required"):
                    parameters["required"] = tool.input_schema["required"]
                
                # Build function definition
                function_def = {
                    "name": tool.name,
                    "parameters": parameters
                }
                
                # Add description if present
                if tool.description:
                    function_def["description"] = tool.description
                
                openai_tool = {
                    "type": "function",
                    "function": function_def
                }
                tools.append(openai_tool)
            
            if tools:
                openai_request["tools"] = tools
                
                # Handle tool_choice if present (like Go implementation)
                if request.tool_choice:
                    if isinstance(request.tool_choice, dict):
                        choice_type = request.tool_choice.get("type")
                        if choice_type == "auto":
                            openai_request["tool_choice"] = "auto"
                        elif choice_type == "any":
                            openai_request["tool_choice"] = "required"
                        elif choice_type == "tool":
                            # Specific tool choice
                            name = request.tool_choice.get("name")
                            if name:
                                openai_request["tool_choice"] = {
                                    "type": "function",
                                    "function": {"name": name}
                                }
                        else:
                            # Default to auto if we can't determine
                            openai_request["tool_choice"] = "auto"
                    else:
                        # Default to auto
                        openai_request["tool_choice"] = "auto"
        
        # Debug logging to see what we're sending to the API
        return openai_request
    
    async def _handle_openai_response(self, response: Any, request: AnthropicRequest) -> Dict[str, Any]:
        """Handle non-streaming OpenAI-compatible response"""
        try:
            content_blocks: List[AnthropicContentBlock] = []
            
            # OpenAI client response has choices array
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                
                if hasattr(choice, 'message') and choice.message:
                    message = choice.message
                    
                    # Handle both content and tool calls like Go implementation
                    has_content = hasattr(message, 'content') and message.content
                    has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                    
                    if has_content:
                        content_text = message.content
                        content_blocks.append(AnthropicContentBlock(type="text", text=content_text))
                    
                    if has_tool_calls:
                        # Convert each tool call to Anthropic tool_use format (like Go implementation)
                        for tool_call in message.tool_calls:
                            if hasattr(tool_call, 'function') and tool_call.function:
                                function = tool_call.function
                                
                                # Create Anthropic tool_use block
                                tool_use_block = {
                                    "type": "tool_use",
                                    "id": getattr(tool_call, 'id', ''),
                                    "name": getattr(function, 'name', ''),
                                }
                                
                                # Handle arguments - parse JSON string if needed (like Go implementation)
                                if hasattr(function, 'arguments'):
                                    args = function.arguments
                                    if isinstance(args, str):
                                        try:
                                            # Parse JSON string arguments
                                            parsed_args = json.loads(args)
                                            tool_use_block["input"] = parsed_args
                                        except json.JSONDecodeError:
                                            # If parsing fails, wrap in raw field like Go implementation
                                            tool_use_block["input"] = {"raw": args}
                                    elif isinstance(args, dict):
                                        # Already a dict, use directly
                                        tool_use_block["input"] = args
                                    else:
                                        # Fallback for any other type
                                        tool_use_block["input"] = {"raw": str(args)}
                                
                                # Add to content blocks
                                content_blocks.append(tool_use_block)
                    
                    # If no content blocks were created, add a default empty text block (like Go implementation)
                    if not content_blocks:
                        content_blocks.append(AnthropicContentBlock(type="text", text=""))
                else:
                    logger.warning("⚠️ Choice has no message")
            else:
                logger.warning("⚠️ Response has no choices")
            
            # Handle usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'prompt_tokens'):
                    input_tokens = response.usage.prompt_tokens or 0
                if hasattr(response.usage, 'completion_tokens'):
                    output_tokens = response.usage.completion_tokens or 0
            
            usage = AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Handle finish_reason
            stop_reason = "end_turn"
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    stop_reason = choice.finish_reason
            
            return {
                "content": content_blocks,
                "id": getattr(response, 'id', ''),
                "model": request.model,
                "role": "assistant",
                "stop_reason": stop_reason,
                "type": "message",
                "usage": usage.dict()
            }
        except Exception as e:
            logger.error(f"Error processing {self.provider_name} response: {e}")
            return {
                "content": [AnthropicContentBlock(type="text", text="Response processing error")],
                "id": "error_response",
                "model": request.model,
                "role": "assistant",
                "stop_reason": "error",
                "type": "message",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
    
    async def _handle_openai_streaming(self, response: Any, request: AnthropicRequest) -> Dict[str, Any]:
        """Handle streaming OpenAI-compatible response (simplified)"""
        try:
            full_text = ""
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    if hasattr(choice.message, 'content') and choice.message.content:
                        full_text = choice.message.content
            
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
        except Exception as e:
            logger.error(f"Error processing {self.provider_name} streaming response: {e}")
            return {
                "content": [AnthropicContentBlock(type="text", text="Streaming response error")],
                "id": "streaming_error",
                "model": request.model,
                "role": "assistant",
                "stop_reason": "error",
                "type": "message",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            } 