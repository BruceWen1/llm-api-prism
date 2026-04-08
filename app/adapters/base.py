# -*- coding: utf-8 -*-
"""
base.py — Abstract base class for protocol backends (HTTP clients).

Three-layer architecture:
  Layer 1 (Routing):    proxy.py — decides from/to, assembles the conversion chain
  Layer 2 (Chain):      input_converter → backend.call() → input_converter
  Layer 3 (Conversion): ProtocolConverter — pure protocol ↔ IR data transformation

This file defines the Backend layer: pure HTTP clients that know how to call
a specific protocol's API. Backends use a ProtocolConverter internally to
translate between the common IR and the protocol's wire format.

All four backends are structurally identical:
  1. Accept an IR request
  2. Use converter.ir_to_request() to build the protocol request body
  3. Send the HTTP request
  4. Use converter.response_to_ir() to parse the protocol response back to IR
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class BackendAdapter(ABC):
    """Abstract base class for protocol backends.

    A backend is a pure HTTP client for a specific protocol's API.
    It uses a ProtocolConverter to handle data transformation, keeping
    the HTTP transport and data conversion concerns separated.

    Every subclass must implement two methods:
      - chat_completion:        non-streaming API call
      - chat_completion_stream: streaming API call
    """

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion call.

        Args:
            request: The chat completion request in common IR format.

        Returns:
            The chat completion response in common IR format.
        """

    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Streaming chat completion call.

        Args:
            request: The chat completion request in common IR format.

        Yields:
            ChatCompletionStreamResponse chunks in common IR format.
        """
