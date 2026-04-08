# -*- coding: utf-8 -*-
"""openai_adapter.py — OpenAI backend (pure HTTP client)."""

from collections.abc import AsyncIterator

import httpx

from app.adapters.base import BackendAdapter
from app.converters.openai_converter import OpenAIConverter
from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class OpenAIAdapter(BackendAdapter):
    """Pure HTTP adapter for the OpenAI Chat Completion API.

    All protocol conversion is delegated to :class:`OpenAIConverter`;
    this class is responsible only for HTTP transport.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            api_key: OpenAI API key.
            base_url: Base URL for the OpenAI-compatible API endpoint.
        """
        self.api_key: str = api_key or ""
        self.base_url: str = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.chat_endpoint: str = f"{self.base_url}/chat/completions"
        self.converter: OpenAIConverter = OpenAIConverter()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for OpenAI API requests.

        Returns:
            A dict containing Authorization and Content-Type headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request to OpenAI.

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
                self.chat_endpoint,
                headers=self._build_headers(),
                json=body,
            )
            response.raise_for_status()
            data: dict = response.json()

        return self.converter.response_to_ir(data, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Send a streaming chat completion request to OpenAI.

        Args:
            request: The chat completion request in common IR format.

        Yields:
            ChatCompletionStreamResponse chunks as they arrive from the API.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
        """
        request.stream = True
        body: dict = self.converter.ir_to_request(request, streaming=True)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                self.chat_endpoint,
                headers=self._build_headers(),
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self.converter.stream_line_to_ir(line, request.model)
                    if chunk is not None:
                        yield chunk