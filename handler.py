import json
import time
import uuid
import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import threading
import sqlite3
import os
import subprocess
import sys

from models import (
    AnthropicRequest, AnthropicResponse, ModelsResponse, 
    HealthResponse, ErrorResponse, ProviderRoute, ProviderVariant
)
from providers import BedrockProvider, OpenRouterProvider, CerebrasProvider, GroqProvider
from config import Config
from middleware import get_request_body

# Configure logging
logger = logging.getLogger()

# Create router
router = APIRouter()


class Spinner:
    """Simple spinner for indeterminate progress with continuous timing"""
    
    def __init__(self, desc="Processing"):
        self.desc = desc
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current_char = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.start_time = None
        self.timing_enabled = False
    
    def start(self):
        """Start the spinner in a separate thread"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the spinner"""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line and add newline for clean output
        print('\r', end='', flush=True)
        print(' ' * 80, end='\r', flush=True)  # Clear the line completely
    
    def update_description(self, desc):
        """Update the description"""
        with self.lock:
            self.desc = desc
    
    def enable_timing(self):
        """Enable continuous timing updates"""
        with self.lock:
            self.timing_enabled = True
            self.start_time = time.time()
    
    def _spin(self):
        """Spin the spinner with continuous timing updates"""
        while self.running:
            with self.lock:
                char = self.spinner_chars[self.current_char]
                if self.timing_enabled and self.start_time:
                    elapsed = time.time() - self.start_time
                    display_text = f"{char} {self.desc} ({elapsed:.1f}s)"
                else:
                    display_text = f"{char} {self.desc}"
                if self.running:
                    print(f'\r{display_text}', end='', flush=True)
                self.current_char = (self.current_char + 1) % len(self.spinner_chars)
            time.sleep(0.1)


class ProxyHandler:
    """Main handler for processing Anthropic requests and routing to providers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers: Dict[str, Provider] = {}
        self.rate_limit_map: Dict[str, float] = {}
        self.rate_limit_mutex = {}
        
        # Request caching system - store last 10 requests/responses
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_order: List[str] = []  # Track order for LRU eviction
        self.max_cache_size = 10
        
        # Initialize database
        self._init_database()
        
        # Initialize providers
        self._init_providers()
        
        # Register routes
        self._register_routes()
    
    @property
    def router(self) -> APIRouter:
        """Get the router with registered routes"""
        return router
    
    def _register_routes(self):
        """Register API routes with the router"""
        
        # Main Anthropic endpoint
        @router.post("/messages")
        async def messages_endpoint(request: Request):
            return await self.handle_messages(request)
    
    def _init_database(self):
        """Initialize SQLite database for request logging"""
        db_path = os.path.join(os.path.dirname(__file__), 'request_history.db')
        self.db_path = db_path
        
        # Create connection and table if not exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_id TEXT,
                success BOOLEAN,
                tokens_used INTEGER,
                original_model TEXT,
                provider TEXT,
                routed_model TEXT,
                duration_seconds REAL,
                error_message TEXT,
                is_streaming BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        
        # Check if is_streaming column exists, add it if not (migration for existing DBs)
        cursor.execute("PRAGMA table_info(request_history)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'is_streaming' not in columns:
            cursor.execute('ALTER TABLE request_history ADD COLUMN is_streaming BOOLEAN DEFAULT 0')
            conn.commit()
        
        conn.close()
    
    def _save_request_to_db(self, request_id: str, success: bool, tokens: int, 
                           original_model: str, provider: str, routed_model: str,
                           duration: float, error_msg: str = None, is_streaming: bool = False):
        """Save request details to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO request_history 
            (request_id, success, tokens_used, original_model, provider, routed_model, duration_seconds, error_message, is_streaming)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (request_id, success, tokens, original_model, provider, routed_model, duration, error_msg, is_streaming))
        
        conn.commit()
        conn.close()
    
    def _generate_request_hash(self, body_bytes: bytes) -> str:
        """Generate SHA256 hash of request body for caching"""
        return hashlib.sha256(body_bytes).hexdigest()
    
    def _add_to_cache(self, request_hash: str, response: Dict[str, Any], request_body: bytes):
        """Add request/response to cache with LRU eviction"""
        # Add to cache
        self.request_cache[request_hash] = {
            "response": response,
            "request_body": request_body,
            "timestamp": time.time(),
            "size": len(str(response))
        }
        
        # Update order (move to end if exists, add to end if new)
        if request_hash in self.cache_order:
            self.cache_order.remove(request_hash)
        self.cache_order.append(request_hash)
        
        # Evict oldest if cache is full
        if len(self.cache_order) > self.max_cache_size:
            oldest_hash = self.cache_order.pop(0)
            if oldest_hash in self.request_cache:
                del self.request_cache[oldest_hash]
        
        # Cache the response silently
    
    def _get_from_cache(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """Get response from cache if it exists"""
        if request_hash in self.request_cache:
            cached_data = self.request_cache[request_hash]
            logger.info(f"🎯 Cache hit: {request_hash[:8]}... (age: {time.time() - cached_data['timestamp']:.1f}s)")
            return cached_data["response"]
        return None
    
    def _init_providers(self):
        """Initialize provider instances based on routing configuration"""
        
        # Initialize providers based on routing configuration
        if hasattr(self.config, 'routing') and self.config.routing.enable and self.config.routing.models:
            for model_routing in self.config.routing.models:
                
                for route in model_routing.provider_sequence:
                    provider_name = route.name.lower()
                    provider_key = f"{provider_name}:{model_routing.model}"
                    
                    if provider_name == "bedrock":
                        try:
                            # Convert variants to dict format for the provider
                            model_variants = [variant.dict() for variant in route.variants]
                            self.providers[provider_key] = BedrockProvider(
                                model_variants=model_variants
                            )
                        except Exception as e:
                            logger.error(f"❌ Failed to initialize Bedrock provider '{provider_key}': {e}")
                    
                    elif provider_name == "openrouter":
                        try:
                            # OpenRouter needs base_url and api_key from main config
                            if hasattr(self.config.providers, 'openrouter') and self.config.providers.openrouter:
                                base_url = self.config.providers.openrouter.base_url or "https://openrouter.ai"
                                self.providers[provider_key] = OpenRouterProvider(
                                    base_url=base_url,
                                    api_key=self.config.providers.openrouter.api_key
                                )
                            else:
                                logger.warning("⚠️ OpenRouter provider not configured in providers section")
                        except Exception as e:
                            logger.error(f"❌ Failed to initialize OpenRouter provider '{provider_key}': {e}")
                    
                    elif provider_name == "cerebras":
                        try:
                            if hasattr(self.config.providers, 'cerebras') and self.config.providers.cerebras:
                                # Use default base_url if not specified in config
                                base_url = self.config.providers.cerebras.base_url or "https://api.cerebras.ai"
                                # Convert variants to dict format for the provider
                                model_variants = [variant.dict() for variant in route.variants]
                                self.providers[provider_key] = CerebrasProvider(
                                    base_url=base_url,
                                    api_key=self.config.providers.cerebras.api_key,
                                    model_variants=model_variants
                                )
                            else:
                                logger.warning("⚠️ Cerebras provider not configured in providers section")
                        except Exception as e:
                            logger.error(f"❌ Failed to initialize Cerebras provider '{provider_key}': {e}")
                    
                    elif provider_name == "groq":
                        try:
                            if hasattr(self.config.providers, 'groq') and self.config.providers.groq:
                                # Use default base_url if not specified in config
                                base_url = self.config.providers.groq.base_url or "https://api.groq.com/openai"
                                # Convert variants to dict format for the provider
                                model_variants = [variant.dict() for variant in route.variants]
                                self.providers[provider_key] = GroqProvider(
                                    base_url=base_url,
                                    api_key=self.config.providers.groq.api_key,
                                    model_variants=model_variants
                                )
                            else:
                                logger.warning("⚠️ Groq provider not configured in providers section")
                        except Exception as e:
                            logger.error(f"❌ Failed to initialize Groq provider '{provider_key}': {e}")
                    
                    else:
                        logger.warning(f"⚠️ Unknown provider type: {provider_name}")
    
    def get_provider(self, name: str) -> Optional[BedrockProvider | OpenRouterProvider | CerebrasProvider | GroqProvider]:
        """Get provider by name"""
        return self.providers.get(name)
    
    def is_rate_limited(self, model: str) -> bool:
        """Check if a model is currently rate limited"""
        if model not in self.rate_limit_map:
            return False
        
        last_failure = self.rate_limit_map[model]
        rate_limit_duration = self.config.routing.rate_limit_seconds
        time_since_failure = time.time() - last_failure
        
        return time_since_failure < rate_limit_duration
    
    def mark_rate_limited(self, model: str):
        """Mark a model as rate limited"""
        self.rate_limit_map[model] = time.time()
        logger.warning(f"🚫 Model {model} rate limited for {self.config.routing.rate_limit_seconds}s")
    
    async def handle_messages(self, request: Request) -> Response:
        """Handle /v1/messages endpoint (main Anthropic API endpoint)"""
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Parse request body - use middleware function like Go server
            body_bytes = await get_request_body(request)
            
            if not body_bytes:
                logger.error("❌ No request body found")
                raise HTTPException(status_code=400, detail="Error reading request body")
            
            # Generate request hash for caching
            request_hash = self._generate_request_hash(body_bytes)
            
            # Check cache first
            cached_response = self._get_from_cache(request_hash)
            if cached_response:
                return cached_response
            
            try:
                anthropic_req = AnthropicRequest.parse_raw(body_bytes)
                # Log essential request info
                model_emoji = "🤏" if "haiku" in anthropic_req.model else "🧠"

                logger.info(f"{model_emoji} ➡️ Request started | model={anthropic_req.model} | stream={anthropic_req.stream}")
            except Exception as e:
                logger.error(f"❌ Error parsing JSON: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON")
 
            # Process with fallback chain using custom spinner
            spinner = Spinner("Request started")
            spinner.start()
            try:
                response = await self._handle_with_fallback_chain(anthropic_req, request_id, spinner)
            except Exception as e:
                spinner.stop()
                raise
            finally:
                spinner.stop()
  
            response_time = round(time.time() - start_time, 2)
            model_emoji = "🤏" if "haiku" in anthropic_req.model else "🧠"
            tokens = response['usage']['input_tokens'] + response['usage']['output_tokens']
            logger.info(f"{model_emoji} ⬅️ Response completed in {response_time}s | provider={response['final_provider']} | model={response['actual_model']} | tokens={tokens}")
            
            # Save to database
            self._save_request_to_db(
                request_id=request_id,
                success=True,
                tokens=tokens,
                original_model=anthropic_req.model,
                provider=response.get('final_provider', 'unknown'),
                routed_model=response.get('actual_model', anthropic_req.model),
                duration=response_time,
                is_streaming=anthropic_req.stream
            )
            
            # Cache the response
            self._add_to_cache(request_hash, response, body_bytes)
            
            return response
            
        except HTTPException as e:
            # Save failed request to database
            response_time = round(time.time() - start_time, 2)
            self._save_request_to_db(
                request_id=request_id,
                success=False,
                tokens=0,
                original_model=anthropic_req.model if 'anthropic_req' in locals() else 'unknown',
                provider='none',
                routed_model='none',
                duration=response_time,
                error_msg=str(e.detail),
                is_streaming=anthropic_req.stream if 'anthropic_req' in locals() else False
            )
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Save failed request to database
            response_time = round(time.time() - start_time, 2)
            self._save_request_to_db(
                request_id=request_id,
                success=False,
                tokens=0,
                original_model=anthropic_req.model if 'anthropic_req' in locals() else 'unknown',
                provider='none',
                routed_model='none',
                duration=response_time,
                error_msg=str(e),
                is_streaming=anthropic_req.stream if 'anthropic_req' in locals() else False
            )
            logger.error(f"❌ Unexpected error processing request {request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def _handle_with_fallback_chain(self, request: AnthropicRequest, request_id: str, spinner: Spinner) -> AnthropicResponse:
        """Handle request with provider fallback chain"""
        
        # Determine provider order based on routing configuration
        provider_order = self._get_provider_order(request.model)
        
        last_error = None
        
        for provider_key in provider_order:
            if provider_key not in self.providers:
                continue
            
            provider = self.providers[provider_key]
            
            # Get provider name and model for display
            provider_name = provider_key.split(':')[0]
            routing_model = provider_key.split(':')[1]
            
            try:
                # Try the provider and get response with metadata
                spinner.update_description(f"Starting {provider_name}:{routing_model}")
                
                # Track start time for this provider attempt
                provider_start_time = time.time()
                
                # Enable continuous timing updates for this provider attempt
                spinner.enable_timing()
                
                # Create a callback for the provider to update spinner with variant info
                def update_spinner_variant(variant_info):
                    # Show variant info immediately
                    spinner.update_description(f"Trying {provider_name}:{variant_info}")
                
                response = await provider.forward_request(request, spinner_callback=update_spinner_variant)
                response['final_provider'] = provider_name
                response['final_model'] = routing_model
            
                # Calculate total time for this provider attempt
                provider_total_time = time.time() - provider_start_time
                spinner.update_description(f"Success: {provider_name}:{routing_model} in {provider_total_time:.1f}s")
                # Disable timing updates for final message
                spinner.timing_enabled = False
                return response
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                # Calculate total time for this provider attempt
                provider_total_time = time.time() - provider_start_time
                
                # Check if we should continue with fallback
                if "429" in error_msg:  # Rate limit
                    spinner.update_description(f"Rate limited on {provider_name}:{routing_model} after {provider_total_time:.1f}s, trying next...")
                    # Disable timing updates for error message
                    spinner.timing_enabled = False
                    # Mark this model as rate limited
                    self.mark_rate_limited(request.model)
                    continue
                elif "401" in error_msg:  # Auth error
                    logger.error(f"❌ Auth error on {provider_name}:{routing_model} after {provider_total_time:.1f}s, stopping fallback chain")
                    # Disable timing updates for error message
                    spinner.timing_enabled = False
                    break
                else:
                    spinner.update_description(f"Error on {provider_name}:{routing_model} after {provider_total_time:.1f}s, trying next...")
                    # Disable timing updates for error message
                    spinner.timing_enabled = False
                    continue
        
        # All providers failed
        spinner.update_description("All providers failed")
        logger.error(f"❌ All providers failed for request {request_id}. Last error: {last_error}")
        raise HTTPException(status_code=502, detail=f"All providers failed: {last_error}")
    
    def _get_provider_order(self, model: str) -> List[str]:
        """Get provider order based on routing configuration with model-specific routing"""
        routing = self.config.routing
        
        # Check for model-specific routing
        if routing.models:
            # Look for exact model match first
            for model_routing in routing.models:
                if model_routing.model == model:
                    provider_order = [f"{route.name.lower()}:{model_routing.model}" for route in model_routing.provider_sequence]
                    return provider_order
            
            # Look for wildcard matches (e.g., "haiku" in "claude-3-5-haiku-20241022")
            for model_routing in routing.models:
                if model_routing.model != "default" and model_routing.model.lower() in model.lower():
                    provider_order = [f"{route.name.lower()}:{model_routing.model}" for route in model_routing.provider_sequence]
                    return provider_order
            
            # Use default routing if no specific match found
            for model_routing in routing.models:
                if model_routing.model == "default":
                    provider_order = [f"{route.name.lower()}:{model_routing.model}" for route in model_routing.provider_sequence]
                    return provider_order
        
        # Fallback to hardcoded order
        return ['bedrock:fallback', 'openrouter:fallback']


def run_streamlit_app(port=8501):
    """Run the Streamlit app in a separate thread"""
    def start_streamlit():
        try:
            # Run streamlit with the app file
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "streamlit_app.py",
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--logger.level", "error"
            ], check=False)
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {e}")
    
    # Start in a daemon thread
    thread = threading.Thread(target=start_streamlit, daemon=True)
    thread.start()
    
    # Give Streamlit a moment to start
    time.sleep(2)
    
    # Print the URL
    print(f"\n{'='*60}")
    print(f"📊 Streamlit Dashboard Available at: http://localhost:{port}")
    print(f"{'='*60}\n")
    logger.info(f"📊 Streamlit dashboard running on http://localhost:{port}")