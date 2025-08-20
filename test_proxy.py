#!/usr/bin/env python3
"""
Simple test script for the Claude Code Proxy
"""

import asyncio
import json
import os
from pathlib import Path

# Add the current directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import AnthropicRequest, AnthropicMessage
from providers import BedrockProvider, OpenRouterProvider
from config import load_config


async def test_providers():
    """Test provider initialization and basic functionality"""
    print("🧪 Testing Provider Initialization")
    print("=" * 40)
    
    try:
        # Test Bedrock provider
        bedrock = BedrockProvider(region="us-east-1")
        print(f"✅ Bedrock provider created: {bedrock.name()}")
        
        # Test OpenRouter provider
        openrouter = OpenRouterProvider(
            base_url="https://openrouter.ai",
            api_key="test-key"
        )
        print(f"✅ OpenRouter provider created: {openrouter.name()}")
        
    except Exception as e:
        print(f"❌ Provider test failed: {e}")
        return False
    
    return True


async def test_models():
    """Test data model creation and validation"""
    print("\n🧪 Testing Data Models")
    print("=" * 40)
    
    try:
        # Test AnthropicMessage
        message = AnthropicMessage(
            role="user",
            content="Hello, Claude!"
        )
        print(f"✅ AnthropicMessage created: {message.role} - {message.content}")
        
        # Test AnthropicRequest
        request = AnthropicRequest(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[message]
        )
        print(f"✅ AnthropicRequest created: {request.model} with {len(request.messages)} messages")
        
        # Test content blocks
        blocks = message.get_content_blocks()
        print(f"✅ Content blocks extracted: {len(blocks)} blocks")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True


async def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration")
    print("=" * 40)
    
    try:
        # Test with example config
        config = load_config("config.yaml.example")
        print(f"✅ Configuration loaded: {config.server.port}")
        print(f"   Routing enabled: {config.routing.enable}")
        print(f"   Provider count: {len(config.routing.provider_sequence)}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True


async def test_request_processing():
    """Test request processing logic"""
    print("\n🧪 Testing Request Processing")
    print("=" * 40)
    
    try:
        # Create a test request
        message = AnthropicMessage(
            role="user",
            content="Write a Python hello world program"
        )
        
        request = AnthropicRequest(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[message],
            stream=False
        )
        
        print(f"✅ Test request created: {request.model}")
        print(f"   Max tokens: {request.max_tokens}")
        print(f"   Streaming: {request.stream}")
        print(f"   Messages: {len(request.messages)}")
        
        # Test request serialization
        request_dict = request.dict()
        print(f"✅ Request serialized to dict: {len(request_dict)} fields")
        
        # Test request validation
        validated_request = AnthropicRequest.parse_obj(request_dict)
        print(f"✅ Request validation successful: {validated_request.model}")
        
    except Exception as e:
        print(f"❌ Request processing test failed: {e}")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Claude Code Proxy - Test Suite")
    print("=" * 50)
    
    # Check if running with uv
    is_uv = "UV_RUNNING_DIR" in os.environ or "VIRTUAL_ENV" in os.environ and "uv" in os.environ.get("VIRTUAL_ENV", "")
    
    if is_uv:
        print("✅ Running with uv")
    else:
        print("💡 Tip: Consider using 'uv run test_proxy.py' for better performance")
    
    tests = [
        ("Provider Initialization", test_providers),
        ("Data Models", test_models),
        ("Configuration", test_config),
        ("Request Processing", test_request_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The proxy is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Copy config.yaml.example to config.yaml")
        print("   2. Configure your API keys and settings")
        print("   3. Run the proxy:")
        if is_uv:
            print("      uv run run.py")
        else:
            print("      uv run run.py          # Recommended")
            print("      # or")
            print("      python run.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1) 