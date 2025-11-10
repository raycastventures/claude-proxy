import json
import time
import logging
from typing import Dict, Any, Optional, List
import boto3
from models import AnthropicRequest, AnthropicResponse, AnthropicContentBlock, AnthropicUsage

# Configure logging
logger = logging.getLogger(__name__)


class BedrockProvider:
    """AWS Bedrock provider implementation with region-based client caching"""
    
    def __init__(self, profile_name = None, model_variants: Optional[List[Dict[str, Any]]] = None):
        self.model_variants = model_variants or []
        # Cache for region-based clients
        self._clients: Dict[str, Any] = {}
        self._session = boto3.Session(profile_name=profile_name)
        
    def name(self) -> str:
        return "bedrock"
    
    def _get_client(self, region: str):
        """Get or create a Bedrock client for the specified region"""
        if region not in self._clients:
            self._clients[region] = self._session.client(
                'bedrock-runtime',
                region_name=region,
                
                config=boto3.session.Config(
                    read_timeout=10000,  # No read timeout
                    connect_timeout=10000,  # No connect timeout
                    retries={'max_attempts': 0},  # No retries
                    parameter_validation=False,  # Disable parameter validation for speed
                    user_agent_extra='claude-code-proxy'  # Add user agent for tracking
                )
            )
        return self._clients[region]
    
    async def forward_request(self, request: AnthropicRequest, **kwargs) -> Dict[str, Any]:
        """Forward Anthropic request to Bedrock with fallback to next variants on 429"""
        
        # Get spinner callback if provided
        spinner_callback = kwargs.get('spinner_callback')
        
        # Convert Anthropic request to Bedrock format
        bedrock_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "messages": [msg.dict() for msg in request.messages]
        }
        
        # Add system message if present
        if request.system:
            system_text = "\n\n".join([s.text for s in request.system])
            bedrock_body["system"] = system_text
        
        # Add temperature if present
        if request.temperature is not None:
            bedrock_body["temperature"] = request.temperature
        
        # Add tools if present
        if request.tools:
            bedrock_body["tools"] = [tool.dict() for tool in request.tools]
        
        if request.tool_choice is not None:
            bedrock_body["tool_choice"] = request.tool_choice
        
        # Map any requested model to available variants with fallback
        if not self.model_variants:
            raise Exception("No model variants configured")
        
        # Try each variant in sequence until one succeeds
        last_error = None
        
        for i, variant in enumerate(self.model_variants):
            target_model = variant.get('model')
            target_region = variant.get('region')
            
            if not target_model or not target_region:
                logger.warning(f"⚠️ Skipping invalid variant {i}: model={target_model}, region={target_region}")
                continue
            
            # Update spinner with current variant being tried
            if spinner_callback:
                variant_info = f"{target_model}:{target_region}"
                spinner_callback(variant_info)
            
            # Get the appropriate client for this region
            client = self._get_client(target_region)
            
            try:
                if request.stream:
                    response = client.invoke_model_with_response_stream(
                        modelId=target_model,
                        body=json.dumps(bedrock_body)
                    )
                    result = await self._handle_bedrock_streaming(response, request)
                    # Add metadata about which variant was used
                    result['actual_model'] = target_model
                    result['actual_region'] = target_region
                    return result
                else:
                    response = client.invoke_model(
                        modelId=target_model,
                        body=json.dumps(bedrock_body)
                    )
                    result = await self._handle_bedrock_response(response, request)
                    # Add metadata about which variant was used
                    result['actual_model'] = target_model
                    result['actual_region'] = target_region
                    return result
                    
            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "throttling" in error_str or "too many requests" in error_str:
                    if i < len(self.model_variants) - 1:
                        continue
                    else:
                        logger.error(f"❌ All variants exhausted due to rate limiting")
                        raise Exception("429: All model variants are rate limited")
                else:
                    if i < len(self.model_variants) - 1:
                        continue
                    else:
                        logger.error(f"❌ All variants exhausted")
                        raise Exception(f"All model variants failed: {last_error}")
        
        # If we get here, all variants failed
        raise Exception(f"All model variants failed: {last_error}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs from config variants"""
        return [variant.get('model', '') for variant in self.model_variants if variant.get('model')]
    
    def get_available_regions(self) -> List[str]:
        """Get list of available regions from config variants"""
        regions = set()
        for variant in self.model_variants:
            if variant.get('region'):
                regions.add(variant['region'])
        return list(regions)
    
    async def _handle_bedrock_response(self, response: Dict[str, Any], request: AnthropicRequest) -> Dict[str, Any]:
        """Handle non-streaming Bedrock response"""
        response_body = json.loads(response['body'].read())
        
        # Convert Bedrock response to Anthropic format
        content_blocks = []
        if 'content' in response_body:
            for i, block in enumerate(response_body['content']):
                if block['type'] == 'text':
                    content_blocks.append(AnthropicContentBlock(
                        type="text",
                        text=block['text']
                    ))
                elif block['type'] == 'tool_use':
                    # Handle tool use blocks - convert to proper format
                    tool_use_block = {
                        "type": "tool_use",
                        "id": block.get('id', ''),
                        "name": block.get('name', ''),
                        "input": block.get('input', {})
                    }
                    content_blocks.append(tool_use_block)
        
        # If no content blocks were created, add a default empty text block
        if not content_blocks:
            content_blocks.append(AnthropicContentBlock(type="text", text=""))
        
        usage = AnthropicUsage(
            input_tokens=response_body.get('usage', {}).get('input_tokens', 0),
            output_tokens=response_body.get('usage', {}).get('output_tokens', 0)
        )
        
        return {
            "content": content_blocks,
            "id": response_body.get('id', ''),
            "model": request.model,  # Keep original requested model for compatibility
            "role": "assistant",
            "stop_reason": response_body.get('stop_reason', 'end_turn'),
            "type": "message",
            "usage": usage.dict()
        }
    
    async def _handle_bedrock_streaming(self, response: Any, request: AnthropicRequest) -> Dict[str, Any]:
        """Handle streaming Bedrock response"""
        # This is a simplified streaming implementation
        # In a real implementation, you'd want to properly stream the response
        full_text = ""
        usage_data = {"input_tokens": 0, "output_tokens": 0}
        message_id = "streaming_response"
        stop_reason = "end_turn"
        
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            # Handle different event types
            if chunk['type'] == 'message_start':
                # Extract message ID and initial usage if available
                if 'message' in chunk:
                    message_id = chunk['message'].get('id', message_id)
                    if 'usage' in chunk['message']:
                        usage_data['input_tokens'] = chunk['message']['usage'].get('input_tokens', 0)
            
            elif chunk['type'] == 'content_block_delta':
                if 'delta' in chunk and chunk['delta']['type'] == 'text_delta':
                    full_text += chunk['delta']['text']
            
            elif chunk['type'] == 'message_delta':
                # Capture updated usage and stop reason
                if 'delta' in chunk:
                    if 'stop_reason' in chunk['delta']:
                        stop_reason = chunk['delta']['stop_reason']
                    if 'usage' in chunk['delta']:
                        if 'output_tokens' in chunk['delta']['usage']:
                            usage_data['output_tokens'] = chunk['delta']['usage']['output_tokens']
            
            elif chunk['type'] == 'message_stop':
                # Final message event might contain usage data
                if 'amazon-bedrock-invocationMetrics' in chunk:
                    metrics = chunk['amazon-bedrock-invocationMetrics']
                    if 'inputTokenCount' in metrics:
                        usage_data['input_tokens'] = metrics['inputTokenCount']
                    if 'outputTokenCount' in metrics:
                        usage_data['output_tokens'] = metrics['outputTokenCount']
        
        content_blocks = [AnthropicContentBlock(type="text", text=full_text)]
        
        return {
            "content": content_blocks,
            "id": message_id,
            "model": request.model,  # Keep original requested model for compatibility
            "role": "assistant",
            "stop_reason": stop_reason,
            "type": "message",
            "usage": usage_data
        } 