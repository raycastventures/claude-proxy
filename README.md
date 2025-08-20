# Claude Code Proxy (Python)


## Setup

Add this to your ~/.zshrc or bashrc:

```
export ANTHROPIC_BASE_URL='http://localhost:3001'
export CLAUDE_CODE_OAUTH_TOKEN='test'
export API_TIMEOUT_MS=600000
```

Then

```
uv run main.py
```

## Intro

A simplified Python implementation of the Claude Code Proxy that focuses on routing Anthropic requests to AWS Bedrock and OpenRouter providers. This proxy maintains compatibility with the original Go implementation while providing a lightweight Python alternative.

## Features

- **Anthropic API Compatibility**: Implements the `/v1/messages` endpoint for Claude requests
- **Multi-Provider Routing**: Routes requests to AWS Bedrock and OpenRouter
- **Fallback Chain**: Automatic fallback between providers when one fails
- **Rate Limiting**: Built-in rate limiting with configurable timeouts
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Configuration Management**: YAML-based configuration with environment variable overrides

## Architecture

The Python proxy follows the same architecture as the Go version:

```
Client Request → Proxy Handler → Provider Selection → Provider API → Response
```

### Components

- **`models.py`**: Data structures for requests/responses
- **`config.py`**: Configuration loading and management
- **`providers.py`**: Provider implementations (Bedrock, OpenRouter)
- **`handler.py`**: Main request processing and routing logic
- **`main.py`**: FastAPI application setup

## Prerequisites

- **Python 3.8+**
- **uv** - Modern Python package manager (recommended) or pip

### Installing uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

## Installation

1. **Clone the repository**:
   ```bash
   cd python
   ```

2. **Install dependencies with uv** (recommended):
   ```bash
   uv sync
   ```

   **Or with pip**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up AWS credentials** (for Bedrock):
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Configure environment variables**:
   ```bash
   export OPENROUTER_API_KEY=your_openrouter_key
   export BEDROCK_REGION=us-east-1
   export ROUTING_ENABLE=true
   ```

## Configuration

The proxy uses a YAML configuration file with environment variable overrides.

### Example `config.yaml`:

```yaml
server:
  port: "8000"
  read_timeout: "60s"
  write_timeout: "60s"
  idle_timeout: "120s"

providers:
  bedrock:
    region: "us-east-1"
    endpoint: ""  # Optional custom endpoint
  
  openrouter:
    base_url: "https://openrouter.ai"
    api_key: "your-api-key-here"

routing:
  enable: true
  provider_sequence:
    - name: "bedrock"
      variants:
        - model: "anthropic.claude-3-sonnet-20240229-v1:0"
          region: "us-east-1"
        - model: "anthropic.claude-3-haiku-20240307-v1:0"
          region: "us-west-2"
    
    - name: "openrouter"
      variants:
        - model: "anthropic/claude-3-sonnet:20240229"
        - model: "anthropic/claude-3-haiku:20240307"
  
  retry_timeout_millis: 1000
  rate_limit_seconds: 60
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `BEDROCK_REGION` | AWS Bedrock region | `us-east-1` |
| `BEDROCK_ENDPOINT` | Custom Bedrock endpoint | None |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | `https://openrouter.ai` |
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `ROUTING_ENABLE` | Enable provider routing | `false` |
| `RETRY_TIMEOUT_MILLIS` | Retry delay in milliseconds | `1000` |
| `RATE_LIMIT_SECONDS` | Rate limit duration | `60` |

## Usage

### Starting the Proxy

#### With uv (recommended):
```bash
# Run directly
uv run main.py

# Or run in development mode with auto-reload
uv run --reload main.py
```

#### With pip:
```bash
python main.py
```

The proxy will start on port 8000 by default.

### API Endpoints

#### Main Endpoint
```bash
POST /v1/messages
Content-Type: application/json

{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1000,
  "messages": [
    {
      "role": "user",
      "content": "Hello, Claude!"
    }
  ]
}
```

#### Other Endpoints
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /debug` - Debug information
- `GET /` - Root endpoint with info

### Example Requests

#### Basic Chat Request
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ]
  }'
```

#### Streaming Request
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Write a short story about a robot."
      }
    ]
  }'
```

#### With System Message
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "system": [
      {
        "type": "text",
        "text": "You are a helpful coding assistant."
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate fibonacci numbers."
      }
    ]
  }'
```

## Provider Fallback Logic

The proxy implements intelligent fallback between providers:

1. **Primary Provider**: First provider in the sequence
2. **Fallback Providers**: Subsequent providers if primary fails
3. **Rate Limiting**: Automatically skips rate-limited models
4. **Retry Logic**: Waits between retry attempts
5. **Model Variants**: Tries different model variants within each provider

### Fallback Example

```yaml
routing:
  enable: true
  provider_sequence:
    - name: "bedrock"
      variants:
        - model: "anthropic.claude-3-sonnet-20240229-v1:0"
        - model: "anthropic.claude-3-haiku-20240307-v1:0"
    
    - name: "openrouter"
      variants:
        - model: "anthropic/claude-3-sonnet:20240229"
        - model: "anthropic/claude-3-haiku:20240307"
```

In this configuration:
1. Try Bedrock with Sonnet model
2. If that fails, try Bedrock with Haiku model
3. If Bedrock fails, try OpenRouter with Sonnet model
4. If that fails, try OpenRouter with Haiku model
5. If all fail, wait for retry timeout and repeat

## Error Handling

The proxy handles various error scenarios:

- **Provider Failures**: Automatic fallback to next provider
- **Rate Limiting**: Automatic model blacklisting with configurable duration
- **Request Timeouts**: Proper error responses for cancelled requests
- **Invalid Requests**: Validation and helpful error messages

## Monitoring and Debugging

### Debug Endpoint
```bash
curl http://localhost:8000/debug
```

Returns:
- Current rate limit status
- Routing configuration
- Provider status
- Timestamp information

### Health Check
```bash
curl http://localhost:8000/health
```

Returns:
- Service status
- Current timestamp

## Development

### Project Structure
```
python/
├── main.py              # FastAPI application
├── handler.py           # Request handling logic
├── providers.py         # Provider implementations
├── models.py            # Data models
├── config.py            # Configuration management
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
├── run.py               # Simple run script
├── test_proxy.py        # Test suite
└── README.md            # This file
```

### Development Commands

#### With uv (recommended):
```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Run tests
uv run test_proxy.py

# Run in development mode
uv run --reload main.py

# Format code
uv run black .

# Lint code
uv run flake8 .

# Security check
uv run bandit -r .
```

#### With pip:
```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements.txt[dev]

# Run tests
python test_proxy.py

# Run in development mode
uvicorn main:app --reload
```

### Adding New Providers

To add a new provider:

1. **Implement the Provider interface** in `providers.py`:
   ```python
   class NewProvider(Provider):
       def name(self) -> str:
           return "newprovider"
       
       async def forward_request(self, request: AnthropicRequest, **kwargs):
           # Implementation here
           pass
   ```

2. **Add configuration** in `config.py`:
   ```python
   class NewProviderConfig(BaseModel):
       api_key: str
       base_url: str
   ```

3. **Update the handler** in `handler.py`:
   ```python
   self.providers["newprovider"] = NewProvider(config)
   ```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/v1/models

# Test main endpoint
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}'
```

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS credentials are properly configured
2. **API Keys**: Verify OpenRouter API key is set
3. **Network Access**: Check firewall and network access to provider APIs
4. **Rate Limits**: Monitor debug endpoint for rate limiting information

### Logs

The proxy outputs detailed logs to stdout:
- 🚀 Request processing
- 🔍 Provider attempts
- ✅ Successful routing
- ❌ Errors and failures
- ⏸️ Rate limiting
- 🔄 Fallback attempts

## Differences from Go Version

This Python implementation is a **simplified version** that focuses on core functionality:

### What's Included
- ✅ Basic request routing
- ✅ Provider fallback logic
- ✅ Rate limiting
- ✅ Streaming support (basic)
- ✅ Configuration management
- ✅ Error handling

### What's Simplified
- ❌ Request/response logging to database
- ❌ Advanced conversation tracking
- ❌ Complex tool handling
- ❌ Subagent routing
- ❌ Advanced metrics and monitoring

## Performance

### Why uv?

- **Faster**: 10-100x faster than pip for dependency resolution
- **Reliable**: Better dependency resolution and lock files
- **Modern**: Built-in virtual environment management
- **Compatible**: Works with existing Python projects

### Benchmarks

- **Dependency installation**: 10-50x faster than pip
- **Virtual environment creation**: Instant
- **Dependency resolution**: Significantly faster
- **Lock file generation**: Automatic and reliable

## License

This project is part of the Claude Code Proxy and follows the same license terms. 