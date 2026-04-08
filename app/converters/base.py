# -*- coding: utf-8 -*-
"""
base.py — Abstract base class for protocol converters.

A Converter handles the pure data transformation between a specific protocol's
wire format and the common intermediate representation (IR). It has no knowledge
of HTTP, networking, or streaming transport — those concerns belong to the
Backend layer.

Every protocol implements the same set of methods, making all protocols
parallel at this layer. The conversion is always bidirectional:

  Protocol format  ⇄  IR (common_models)

Methods come in pairs for request/response, plus streaming variants:

  request_to_ir   /  ir_to_request      — full request conversion
  response_to_ir  /  ir_to_response     — full response conversion
  stream_line_to_ir / ir_to_stream_chunk — streaming chunk conversion
  stream_done_signal                     — protocol-specific termination
"""

from abc import ABC, abstractmethod
from typing import Any

from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class ProtocolConverter(ABC):
    """Abstract base class for protocol converters.

    Every protocol (OpenAI, Anthropic, Gemini, DashScope) implements all
    methods below. There are no default implementations — each protocol
    explicitly defines its own conversion logic.
    """

    # -- Request conversion --------------------------------------------------

    @abstractmethod
    def request_to_ir(self, raw_body: dict[str, Any]) -> ChatCompletionRequest:
        """Convert a protocol-specific request into the common IR.

        Args:
            raw_body: Raw request body in this protocol's format.

        Returns:
            ChatCompletionRequest in common IR format.
        """

    @abstractmethod
    def ir_to_request(
        self, request: ChatCompletionRequest, streaming: bool = False
    ) -> dict[str, Any]:
        """Convert a common IR request into this protocol's wire format.

        Args:
            request: ChatCompletionRequest in common IR format.
            streaming: Whether to include streaming-specific fields.

        Returns:
            Request body dict in this protocol's format.
        """

    # -- Response conversion -------------------------------------------------

    @abstractmethod
    def response_to_ir(
        self, raw_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Convert a protocol-specific response into the common IR.

        Args:
            raw_response: Raw response dict from this protocol's API.
            model: Model name fallback.

        Returns:
            ChatCompletionResponse in common IR format.
        """

    @abstractmethod
    def ir_to_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        """Convert a common IR response into this protocol's wire format.

        Args:
            response: ChatCompletionResponse in common IR format.

        Returns:
            Response dict in this protocol's format.
        """

    # -- Streaming conversion ------------------------------------------------

    @abstractmethod
    def stream_line_to_ir(
        self, line: str, model: str, **kwargs: Any
    ) -> ChatCompletionStreamResponse | None:
        """Convert a single SSE line from this protocol into an IR stream chunk.

        Args:
            line: Raw SSE line string.
            model: Model name fallback.
            **kwargs: Protocol-specific state (e.g. completion_id, is_first_chunk).

        Returns:
            ChatCompletionStreamResponse, or None if the line should be skipped.
        """

    @abstractmethod
    def ir_to_stream_chunk(
        self, chunk: ChatCompletionStreamResponse
    ) -> dict[str, Any]:
        """Convert a common IR stream chunk into this protocol's format.

        Args:
            chunk: ChatCompletionStreamResponse in common IR format.

        Returns:
            Chunk dict in this protocol's format.
        """

    @abstractmethod
    def stream_done_signal(self) -> str | None:
        """Return the stream termination sentinel for this protocol.

        Returns:
            The done signal string (e.g. "[DONE]"), or None if the protocol
            uses an in-band finish indicator instead.
        """
