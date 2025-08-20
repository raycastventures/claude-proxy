import asyncio
import logging
import signal
import sys
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from config import load_config
from handler import ProxyHandler, run_streamlit_app
from middleware import SimpleLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for server reference
server = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
    if server:
        server.should_exit = True


def check_port_availability(port: int) -> bool:
    """Check if a port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError as e:
        # log the error
        logger.error(f"❌ Port {port} is not available. Please stop the service using this port or use a different port. Error: {e}")
        return False





async def main():
    """Main application entry point"""
    global server
    
    try:
        # Load configuration
        logger.info("⚙️ Loading configuration...")
        config = load_config()
        if not config:
            logger.error("❌ Failed to load configuration")
            sys.exit(1)
        
        # Check if port 3001 is available
        port = 3001
        if not check_port_availability(port):
            logger.error(f"❌ Port {port} is not available. Please stop the service using this port or use a different port.")
            sys.exit(1)
        
        # Create FastAPI app
        app = FastAPI(
            title="Claude Code Proxy",
            description="Python proxy for Claude Code requests to Bedrock/OpenRouter",
            version="1.0.0"
        )
        
        # Add CORS middleware - matching Go server's CORS configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Add explicit OPTIONS handler for CORS preflight
        @app.options("/{full_path:path}")
        async def options_handler(request: Request):
            """Handle CORS preflight requests"""
            return Response(
                content="",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        
        # Add logging middleware
        app.add_middleware(SimpleLoggingMiddleware)
        
        # Initialize handler
        handler = ProxyHandler(config)
        
        # Start Streamlit dashboard in a separate thread
        streamlit_port = 8501
        run_streamlit_app(port=streamlit_port)
        
        # Register routes
        app.include_router(handler.router, prefix="/v1")
        
        # Add test endpoint
        @app.get("/test")
        async def test_endpoint():
            """Test endpoint to verify server is running"""
            return {"status": "ok", "message": "Python proxy is running", "timestamp": time.time()}
        
        # Add health check
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "claude-code-proxy"}
        
        logger.info(f"🚀 Starting server on port {port}")
        logger.info(f"🔗 Proxy API: http://localhost:{port}")
        logger.info(f"📊 Dashboard: http://localhost:{streamlit_port}")
        
        # Configure uvicorn to use HTTP/2 for better performance
        config_uvicorn = uvicorn.Config(
            app=app,
            host="0.0.0.0",  # Bind to all interfaces
            port=port,
            log_level="info",
            access_log=False,
            http="httptools",  # Enable HTTP/2 support
            loop="asyncio",  # Use asyncio loop
        )
        
        server = uvicorn.Server(config_uvicorn)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully"""
            if server:
                server.should_exit = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        try:
            await server.serve()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
        finally:
            if server:
                server.should_exit = True
                # Force close any remaining connections
                if hasattr(server, 'server') and server.server:
                    server.server.close()
                    try:
                        await server.server.wait_closed()
                    except Exception as e:
                        logger.warning(f"⚠️ Error waiting for server to close: {e}")
    
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1) 