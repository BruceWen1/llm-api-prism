# -*- coding: utf-8 -*-
"""dashscope_adapter.py — DashScope backend (pure HTTP client).

Uses the DashScope-native protocol where messages live inside an ``input``
object and generation parameters live inside a ``parameters`` object.
Streaming is enabled via the ``X-DashScope-SSE`` request header.

All protocol conversion is delegated to :class:`DashScopeConverter`;
this class is responsible only for HTTP transport.
"""

from collections.abc import AsyncIterator

import httpx

from app.adapters.base import BackendAdapter
from app.converters.dashscope_converter import DashScopeConverter
from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class DashScopeAdapter(BackendAdapter):
    """Pure HTTP adapter for the Alibaba DashScope native API."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
    TEXT_GENERATION_PATH = "/services/aigc/text-generation/generation"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            api_key: DashScope API key.
            base_url: Base URL for the DashScope API endpoint.
        """
        self.api_key: str = api_key or ""
        self.base_url: str = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.generation_endpoint: str = f"{self.base_url}{self.TEXT_GENERATION_PATH}"
        self.converter: DashScopeConverter = DashScopeConverter()

    def _build_headers(self, streaming: bool = False) -> dict[str, str]:
        """Build HTTP headers for DashScope API requests.

        Args:
            streaming: Whether to include the SSE streaming header.

        Returns:
            Headers containing authorization and content type.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if streaming:
            headers["X-DashScope-SSE"] = "enable"
        return headers

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request to DashScope.

        Args:
            request: The chat completion request in common IR format.

        Returns:
            A ChatCompletionResponse in common IR format.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        body = self.converter.ir_to_request(request, streaming=False)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.generation_endpoint,
                headers=self._build_headers(streaming=False),
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        return self.converter.response_to_ir(data, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Send a streaming chat completion request to DashScope.

        Streaming is activated by the ``X-DashScope-SSE: enable`` header.

        Args:
            request: The chat completion request in common IR format.

        Yields:
            ChatCompletionStreamResponse chunks as they arrive from the API.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        request.stream = True
        body = self.converter.ir_to_request(request, streaming=True)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                self.generation_endpoint,
                headers=self._build_headers(streaming=True),
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self.converter.stream_line_to_ir(line, request.model)
                    if chunk is not None:
                        yield chunk
