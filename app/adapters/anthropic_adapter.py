# -*- coding: utf-8 -*-
"""anthropic_adapter.py — Anthropic backend (pure HTTP client)."""

import uuid
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import BackendAdapter
from app.converters.anthropic_converter import AnthropicConverter
from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class AnthropicAdapter(BackendAdapter):
    """Pure HTTP adapter for the Anthropic Claude Messages API.

    All protocol conversion is delegated to :class:`AnthropicConverter`;
    this class is responsible only for HTTP transport.
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            api_key: Anthropic API key.
            base_url: Base URL for the Anthropic API endpoint.
        """
        self.api_key: str = api_key or ""
        self.base_url: str = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.messages_endpoint: str = f"{self.base_url}/messages"
        self.converter: AnthropicConverter = AnthropicConverter()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Anthropic API requests.

        Returns:
            A dict containing the API key, version, and Content-Type headers.
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request to Anthropic.

        Args:
            request: The chat completion request in common IR format.

        Returns:
            A ChatCompletionResponse containing the generated message and
            usage statistics.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        request.stream = False
        body: dict = self.converter.ir_to_request(request, streaming=False)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.messages_endpoint,
                headers=self._build_headers(),
                json=body,
            )
            response.raise_for_status()
            data: dict = response.json()

        return self.converter.response_to_ir(data, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Send a streaming chat completion request to Anthropic.

        Args:
            request: The chat completion request in common IR format.

        Yields:
            ChatCompletionStreamResponse chunks as they arrive from the API.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        request.stream = True
        body: dict = self.converter.ir_to_request(request, streaming=True)
        completion_id: str = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                self.messages_endpoint,
                headers=self._build_headers(),
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self.converter.stream_line_to_ir(
                        line, request.model, completion_id=completion_id
                    )
                    if chunk is not None:
                        yield chunk