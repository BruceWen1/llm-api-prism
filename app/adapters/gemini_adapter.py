# -*- coding: utf-8 -*-
"""gemini_adapter.py — Gemini backend (pure HTTP client).

Uses Google's Gemini ``generateContent`` / ``streamGenerateContent`` endpoints.
Streaming uses the ``?alt=sse`` query parameter to receive Server-Sent Events.

All protocol conversion is delegated to :class:`GeminiConverter`;
this class is responsible only for HTTP transport.
"""

import uuid
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import BackendAdapter
from app.converters.gemini_converter import GeminiConverter
from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class GeminiAdapter(BackendAdapter):
    """Pure HTTP adapter for the Google Gemini API."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            api_key: Gemini API key.
            base_url: Base URL for the Gemini API endpoint.
        """
        self.api_key: str = api_key or ""
        self.base_url: str = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.converter: GeminiConverter = GeminiConverter()
        self._client = httpx.AsyncClient(timeout=120.0)

    def _get_endpoint(self, model: str, stream: bool = False) -> str:
        """Build the API endpoint URL for the given model.

        Args:
            model: Model name (e.g. ``gemini-pro``).
            stream: Whether to use the streaming endpoint.

        Returns:
            Complete endpoint URL including query parameters.
        """
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/models/{model}:{action}"
        if stream:
            url += "?alt=sse"
        return url

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Gemini API requests.

        Returns:
            Headers containing the API key and content type.
        """
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request to Gemini.

        Args:
            request: The chat completion request in common IR format.

        Returns:
            A ChatCompletionResponse in common IR format.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        body = self.converter.ir_to_request(request, streaming=False)
        endpoint = self._get_endpoint(request.model, stream=False)

        response = await self._client.post(
            endpoint,
            headers=self._build_headers(),
            json=body,
        )
        response.raise_for_status()
        data = response.json()

        return self.converter.response_to_ir(data, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Send a streaming chat completion request to Gemini.

        Uses the ``streamGenerateContent?alt=sse`` endpoint.

        Args:
            request: The chat completion request in common IR format.

        Yields:
            ChatCompletionStreamResponse chunks as they arrive from the API.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        body = self.converter.ir_to_request(request, streaming=True)
        endpoint = self._get_endpoint(request.model, stream=True)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        is_first_chunk = True

        async with self._client.stream(
            "POST",
            endpoint,
            headers=self._build_headers(),
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = self.converter.stream_line_to_ir(
                    line, request.model,
                    completion_id=completion_id,
                    is_first_chunk=is_first_chunk,
                )
                if chunk is not None:
                    is_first_chunk = False
                    yield chunk

    async def close(self) -> None:
        """Close the HTTP client.

        This method should be called when the adapter is no longer needed.
        """
        await self._client.aclose()
