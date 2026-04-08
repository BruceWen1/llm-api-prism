# LLM API Prism

<p align="right">English | <a href="README_zh.md">中文</a></p>

An N×N multi-protocol LLM API proxy that enables seamless protocol conversion between any two LLM providers.

## Architecture

The project follows a clean three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Routing (proxy.py)                                │
│ - Route matrix generation for N×N protocol combinations     │
│ - Chain assembly: input protocol → IR → backend protocol    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Chain (BackendAdapter)                            │
│ - Pure HTTP client for each backend                         │
│ - Handles authentication, headers, and streaming            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Conversion (ProtocolConverter)                     │
│ - Bidirectional protocol ↔ IR transformation                 │
│ - IR uses OpenAI format as unified intermediate             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Client Request (Protocol A)
    ↓
proxy.py: Route to /{protocol_a}/{protocol_b}/...
    ↓
Protocol A Converter: Protocol A → IR (OpenAI format)
    ↓
Protocol B Converter: IR → Protocol B
    ↓
Protocol B Adapter: HTTP request to backend
    ↓
Backend Response (Protocol B)
    ↓
Protocol B Converter: Protocol B → IR
    ↓
Protocol A Converter: IR → Protocol A
    ↓
Client Response (Protocol A format)
```

## Supported Protocols

| Protocol | Provider | Status |
|----------|----------|--------|
| OpenAI | OpenAI | ✅ Supported |
| Anthropic | Claude | ✅ Supported |
| Gemini | Google | ✅ Supported |
| DashScope | Alibaba Tongyi Qianwen | ✅ Supported |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9876
```

The server will start on port `9876`.

## Usage

The proxy uses a URL pattern of `/{input_protocol}/{backend}/...` to specify the protocol transformation:

- **input_protocol**: The protocol your client uses
- **backend**: The actual LLM provider to call

Your client only needs to change the `base_url` to `http://localhost:9876/{input_protocol}/{backend}`, everything else remains unchanged.

### Example 1: OpenAI Client → DashScope Backend

```bash
curl http://localhost:9876/openai/dashscope/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DASHSCOPE_API_KEY" \
  -d '{
    "model": "qwen-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Example 2: Anthropic Client → OpenAI Backend

```bash
curl http://localhost:9876/anthropic/openai/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_OPENAI_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gpt-4",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Example 3: Gemini Client → Anthropic Backend

```bash
curl http://localhost:9876/gemini/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_ANTHROPIC_API_KEY" \
  -d '{
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "contents": [{"role": "user", "parts": [{"text": "Hello!"}]}]
  }'
```

### Example 4: DashScope Client → Gemini Backend

```bash
curl http://localhost:9876/dashscope/gemini/api/v1/services/aigc/text-generation/generation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{
    "model": "gemini-pro",
    "input": {
      "messages": [{"role": "user", "content": "Hello!"}]
    }
  }'
```

## Project Structure

```
app/
├── main.py                          # FastAPI entry point
├── models/
│   ├── common_models.py             # Common IR models (OpenAI format)
│   ├── openai_models.py             # OpenAI-specific models
│   ├── anthropic_models.py          # Anthropic-specific models
│   ├── gemini_models.py             # Gemini-specific models
│   └── dashscope_models.py          # DashScope-specific models
├── converters/
│   ├── base.py                      # ProtocolConverter abstract base class
│   ├── openai_converter.py          # OpenAI ↔ IR converter
│   ├── anthropic_converter.py       # Anthropic ↔ IR converter
│   ├── gemini_converter.py          # Gemini ↔ IR converter
│   └── dashscope_converter.py       # DashScope ↔ IR converter
├── adapters/
│   ├── base.py                      # BackendAdapter abstract base class
│   ├── openai_adapter.py            # OpenAI HTTP client
│   ├── anthropic_adapter.py         # Anthropic HTTP client
│   ├── gemini_adapter.py            # Gemini HTTP client
│   └── dashscope_adapter.py         # DashScope HTTP client
└── routers/
    └── proxy.py                     # N×N route matrix generation
```

## Extending with New Protocols

Adding support for a new LLM protocol requires only three steps:

### Step 1: Define Models

Create model definitions in `app/models/{protocol}_models.py`:

```python
from pydantic import BaseModel

class ProtocolRequest(BaseModel):
    # Define request structure
    pass

class ProtocolResponse(BaseModel):
    # Define response structure
    pass
```

### Step 2: Implement Converter

Create `app/converters/{protocol}_converter.py`:

```python
from app.converters.base import ProtocolConverter
from app.models.common_models import ChatCompletionRequest, ChatCompletionResponse
from app.models.protocol_models import ProtocolRequest, ProtocolResponse

class ProtocolConverter(ProtocolConverter):
    def request_to_ir(self, request: ProtocolRequest) -> ChatCompletionRequest:
        # Convert protocol request to IR (OpenAI format)
        pass
    
    def ir_to_request(self, ir: ChatCompletionRequest) -> ProtocolRequest:
        # Convert IR to protocol request
        pass
    
    def response_to_ir(self, response: ProtocolResponse) -> ChatCompletionResponse:
        # Convert protocol response to IR
        pass
    
    def ir_to_response(self, ir: ChatCompletionResponse) -> ProtocolResponse:
        # Convert IR to protocol response
        pass
```

### Step 3: Implement Adapter

Create `app/adapters/{protocol}_adapter.py`:

```python
from app.adapters.base import BackendAdapter

class ProtocolAdapter(BackendAdapter):
    async def chat_completion(self, request):
        # Make HTTP request to the backend
        pass
```

The route matrix in `proxy.py` will automatically include all protocol combinations once the converter and adapter are registered.

## License

MIT
