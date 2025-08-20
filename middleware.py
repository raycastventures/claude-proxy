import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

# Configure logging
logger = logging.getLogger(__name__)


class SimpleLoggingMiddleware:
    """Simple, reliable middleware for logging requests"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Store original receive
            original_receive = receive
            
            # Track response
            response_status = [200]
            
            async def capture_response(message):
                if message["type"] == "http.response.start":
                    response_status[0] = message["status"]
                await send(message)
            
            # Process request
            try:
                await self.app(scope, receive, capture_response)
            except Exception as e:
                logger.error(f"❌ ERROR processing {scope['method']} {scope['path']}: {e}")
                raise
        else:
            await self.app(scope, receive, send)


async def get_request_body(request: Request) -> bytes:
    """Get request body bytes, similar to Go server's getBodyBytes"""
    # Always read body directly from request - simpler and more reliable
    body = await request.body()
    return body
