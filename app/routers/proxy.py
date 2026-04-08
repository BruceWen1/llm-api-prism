# -*- coding: utf-8 -*-
"""
proxy.py — N×N proxy router generation for LLM API Prism.

Three-layer architecture:
  Layer 1 (Routing):    This file — decides from/to, assembles the conversion chain
  Layer 2 (Chain):      input_converter.request_to_ir() → backend.call() → input_converter.ir_to_response()
  Layer 3 (Conversion): ProtocolConverter — pure protocol ↔ IR data transformation

Data flow for a request (e.g. anthropic-to-dashscope):
  1. AnthropicConverter decodes the request:  Anthropic format  →  IR
  2. DashScopeAdapter calls the backend:      IR  →  DashScope API  →  IR
  3. AnthropicConverter encodes the response:  IR  →  Anthropic format

URL format: /{input_protocol}/{backend}/...
  - input_protocol: determines which converter to use for decode/encode
  - backend:        determines which adapter to use for the upstream API call

All N×N combinations are registered automatically. To add a new protocol,
implement its Converter and Adapter, then add entries to CONVERTER_REGISTRY,
BACKEND_REGISTRY, and ROUTER_FACTORIES — the route matrix expands automatically.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Request
from sse_starlette.sse import EventSourceResponse

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.base import BackendAdapter
from app.adapters.dashscope_adapter import DashScopeAdapter
from app.adapters.gemini_adapter import GeminiAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.converters.anthropic_converter import AnthropicConverter
from app.converters.base import ProtocolConverter
from app.converters.dashscope_converter import DashScopeConverter
from app.converters.gemini_converter import GeminiConverter
from app.converters.openai_converter import OpenAIConverter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key extraction helpers
# ---------------------------------------------------------------------------

def extract_api_key_from_bearer(request: Request) -> str | None:
    """Extract API key from Authorization Bearer header (OpenAI / DashScope convention)."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        key = auth_header[len("Bearer "):].strip()
        return key if key else None
    return None


def extract_api_key_from_anthropic(request: Request) -> str | None:
    """Extract API key from Anthropic x-api-key header, with Bearer fallback."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    return extract_api_key_from_bearer(request)


def extract_api_key_from_gemini(request: Request) -> str | None:
    """Extract API key from Gemini x-goog-api-key header, ?key= query param, or Bearer fallback."""
    api_key = request.headers.get("x-goog-api-key")
    if api_key:
        return api_key
    api_key = request.query_params.get("key")
    if api_key:
        return api_key
    return extract_api_key_from_bearer(request)


def extract_api_key(input_protocol: str, request: Request) -> str | None:
    """Extract API key from the request based on the input protocol's convention."""
    if input_protocol == "anthropic":
        return extract_api_key_from_anthropic(request)
    if input_protocol == "gemini":
        return extract_api_key_from_gemini(request)
    return extract_api_key_from_bearer(request)


# ---------------------------------------------------------------------------
# Protocol registries
# ---------------------------------------------------------------------------

# Converter registry: maps protocol name → converter class (stateless, no API key needed)
CONVERTER_REGISTRY: dict[str, type[ProtocolConverter]] = {
    "openai": OpenAIConverter,
    "anthropic": AnthropicConverter,
    "gemini": GeminiConverter,
    "dashscope": DashScopeConverter,
}

# Backend registry: maps protocol name → adapter class (needs API key for HTTP calls)
BACKEND_REGISTRY: dict[str, type[BackendAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "dashscope": DashScopeAdapter,
}


def create_conversion_chain(
    input_protocol: str,
    backend: str,
    api_key: str | None,
) -> tuple[ProtocolConverter, BackendAdapter]:
    """Create the converter and backend adapter for a proxy request.

    Each adapter has its own DEFAULT_BASE_URL — edit the adapter class
    directly if you need to change the upstream endpoint.

    Args:
        input_protocol: The input/output protocol name (determines converter).
        backend: The backend protocol name (determines adapter).
        api_key: Optional API key for the backend adapter.

    Returns:
        A tuple of (input_converter, backend_adapter).
    """
    converter = CONVERTER_REGISTRY[input_protocol]()
    backend_class = BACKEND_REGISTRY[backend]
    backend_adapter = backend_class(api_key=api_key) if api_key else backend_class()
    return converter, backend_adapter


# ---------------------------------------------------------------------------
# Core proxy handler
# ---------------------------------------------------------------------------

async def handle_proxy_request(
    raw_request: Request,
    input_protocol: str,
    backend: str,
) -> Any:
    """Core handler for all proxy requests.

    Conversion chain:
      input_converter.request_to_ir()  →  backend.call()  →  input_converter.ir_to_response()

    Error mapping:
      - 400: request decode failure (malformed client input)
      - 502: backend call failure (upstream error)
      - 500: response encode failure (internal bug)
    """
    api_key = extract_api_key(input_protocol, raw_request)
    input_converter, backend_adapter = create_conversion_chain(input_protocol, backend, api_key)

    # Phase 1: decode request (client error → 400)
    try:
        raw_body = await raw_request.json()
        ir_request = input_converter.request_to_ir(raw_body)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Failed to decode %s request: %s", input_protocol, error)
        raise HTTPException(status_code=400, detail=f"Invalid {input_protocol} request: {error}")

    # Phase 2: call backend (upstream error → 502)
    try:
        if ir_request.stream:
            return EventSourceResponse(
                proxy_stream_generator(input_converter, backend_adapter, ir_request),
                media_type="text/event-stream",
            )
        ir_response = await backend_adapter.chat_completion(ir_request)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Backend %s request failed: %s", backend, error)
        raise HTTPException(status_code=502, detail=f"Backend request failed: {error}")

    # Phase 3: encode response (internal error → 500)
    try:
        return input_converter.ir_to_response(ir_response)
    except Exception as error:
        logger.exception("Failed to encode %s response: %s", input_protocol, error)
        raise HTTPException(status_code=500, detail=f"Response encoding failed: {error}")


def _emit_sse_event(encoded: dict[str, Any]) -> dict[str, str]:
    """Convert an encoded chunk dict into an SSE-compatible yield value.

    If the dict contains a ``_event`` key, the returned dict includes an
    ``event`` field so that ``sse-starlette`` emits a proper ``event:``
    line.  Otherwise only ``data`` is included.

    Args:
        encoded: Encoded chunk dict from a converter, possibly with ``_event``.

    Returns:
        A dict with ``data`` (and optionally ``event``) suitable for
        ``EventSourceResponse``.
    """
    event_type = encoded.pop("_event", None)
    data_str = json.dumps(encoded, ensure_ascii=False)
    if event_type:
        return {"event": event_type, "data": data_str}
    return {"data": data_str}


async def proxy_stream_generator(
    input_converter: ProtocolConverter,
    backend_adapter: BackendAdapter,
    ir_request: Any,
):
    """Stream generator that encodes each backend chunk using the input converter.

    If the encoded dict contains a ``_event`` key, the generator emits a
    proper ``event: <type>`` line before the ``data:`` line.  This is
    required by protocols like Anthropic that use named SSE events.

    The converter may return a single dict, a list of dicts (when one IR
    chunk maps to multiple protocol events), or None (skip).

    Yields:
        SSE event dicts in the input protocol format.
    """
    try:
        async for chunk in backend_adapter.chat_completion_stream(ir_request):
            encoded = input_converter.ir_to_stream_chunk(chunk)

            # Converter may return None to signal "skip this chunk"
            if encoded is None:
                continue

            # Handle list of events (e.g. Anthropic content_block_start + delta)
            if isinstance(encoded, list):
                for event in encoded:
                    yield _emit_sse_event(event)
            else:
                yield _emit_sse_event(encoded)

        done_signal = input_converter.stream_done_signal()
        if done_signal:
            yield {"data": done_signal}

    except Exception as error:
        logger.exception("Proxy stream failed: %s", error)
        error_data = {"error": {"message": str(error), "type": "backend_error"}}
        yield {"data": json.dumps(error_data, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# Router factories (one per input protocol)
# ---------------------------------------------------------------------------

def create_openai_input_router(backend: str) -> APIRouter:
    """Create a router for OpenAI-format input targeting a specific backend."""
    router = APIRouter(tags=[f"openai/{backend}"])

    @router.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        return await handle_proxy_request(raw_request, "openai", backend)

    return router


def create_anthropic_input_router(backend: str) -> APIRouter:
    """Create a router for Anthropic-format input targeting a specific backend."""
    router = APIRouter(tags=[f"anthropic/{backend}"])

    @router.post("/v1/messages")
    async def messages(raw_request: Request):
        return await handle_proxy_request(raw_request, "anthropic", backend)

    return router


def create_gemini_input_router(backend: str) -> APIRouter:
    """Create a router for Gemini-format input targeting a specific backend."""
    router = APIRouter(tags=[f"gemini/{backend}"])

    @router.post("/v1beta/models/{model_name}:generateContent")
    async def generate_content(raw_request: Request, model_name: str = Path(...)):
        return await handle_gemini_proxy(raw_request, model_name, backend, streaming=False)

    @router.post("/v1beta/models/{model_name}:streamGenerateContent")
    async def stream_generate_content(raw_request: Request, model_name: str = Path(...)):
        return await handle_gemini_proxy(raw_request, model_name, backend, streaming=True)

    return router


async def handle_gemini_proxy(
    raw_request: Request,
    model_name: str,
    backend: str,
    streaming: bool = False,
) -> Any:
    """Gemini-specific proxy handler that injects model_name from the URL path.

    Gemini determines streaming via URL path (streamGenerateContent vs
    generateContent), not via a request body field. The ``streaming`` flag
    is derived from the matched route and overrides ``ir_request.stream``.
    """
    api_key = extract_api_key("gemini", raw_request)
    input_converter, backend_adapter = create_conversion_chain("gemini", backend, api_key)

    try:
        raw_body = await raw_request.json()
        if "model" not in raw_body:
            raw_body["model"] = model_name
        ir_request = input_converter.request_to_ir(raw_body)
        ir_request.stream = streaming
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Failed to decode gemini request: %s", error)
        raise HTTPException(status_code=400, detail=f"Invalid gemini request: {error}")

    try:
        if ir_request.stream:
            return EventSourceResponse(
                proxy_stream_generator(input_converter, backend_adapter, ir_request),
                media_type="text/event-stream",
            )
        ir_response = await backend_adapter.chat_completion(ir_request)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Backend %s request failed: %s", backend, error)
        raise HTTPException(status_code=502, detail=f"Backend request failed: {error}")

    try:
        return input_converter.ir_to_response(ir_response)
    except Exception as error:
        logger.exception("Failed to encode gemini response: %s", error)
        raise HTTPException(status_code=500, detail=f"Response encoding failed: {error}")


def create_dashscope_input_router(backend: str) -> APIRouter:
    """Create a router for DashScope-format input targeting a specific backend."""
    router = APIRouter(tags=[f"dashscope/{backend}"])

    @router.post("/api/v1/services/aigc/text-generation/generation")
    async def text_generation(raw_request: Request):
        return await handle_proxy_request(raw_request, "dashscope", backend)

    return router


# ---------------------------------------------------------------------------
# Router factory registry & N×N route matrix
# ---------------------------------------------------------------------------

ROUTER_FACTORIES: dict[str, Any] = {
    "openai": create_openai_input_router,
    "anthropic": create_anthropic_input_router,
    "gemini": create_gemini_input_router,
    "dashscope": create_dashscope_input_router,
}

PROXY_ROUTERS: dict[str, APIRouter] = {}

for _input_protocol, _factory in ROUTER_FACTORIES.items():
    for _backend in BACKEND_REGISTRY:
        route_name = f"{_input_protocol}/{_backend}"
        PROXY_ROUTERS[route_name] = _factory(_backend)
