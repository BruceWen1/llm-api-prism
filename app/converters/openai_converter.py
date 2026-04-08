# -*- coding: utf-8 -*-
"""
openai_converter.py — Protocol converter for the OpenAI Chat Completion API.

Extracts and encapsulates all protocol conversion logic from OpenAIAdapter.
Handles bidirectional transformation between OpenAI wire format and the
common intermediate representation (IR).
"""

import json
import logging
import uuid
from typing import Any

from app.converters.base import ProtocolConverter
from app.models.common_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    Tool,
    ToolCall,
    UsageInfo,
)
from app.models.openai_models import OpenAIRequest

logger = logging.getLogger(__name__)


class OpenAIConverter(ProtocolConverter):
    """Converter for the OpenAI Chat Completion API protocol.

    Implements pure data transformation between OpenAI wire format and
    the common IR. No HTTP, networking, or streaming transport concerns.
    """

    # ------------------------------------------------------------------
    # Request conversion: OpenAI <-> IR
    # ------------------------------------------------------------------

    def request_to_ir(self, raw_body: dict[str, Any]) -> ChatCompletionRequest:
        """Convert an OpenAI-format request into the common IR.

        Args:
            raw_body: Raw OpenAI request body.

        Returns:
            ChatCompletionRequest in common IR format.
        """
        openai_request = OpenAIRequest(**raw_body)
        return ChatCompletionRequest(**openai_request.model_dump(exclude_none=True))

    def ir_to_request(
        self, request: ChatCompletionRequest, streaming: bool = False
    ) -> dict[str, Any]:
        """Convert a common IR request into OpenAI wire format.

        Args:
            request: ChatCompletionRequest in common IR format.
            streaming: Whether to include streaming-specific fields.

        Returns:
            Request body dict in OpenAI format.
        """
        body: dict = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": streaming,
        }

        optional_fields = [
            "temperature", "top_p", "max_tokens", "stop",
            "presence_penalty", "frequency_penalty", "user",
            "tool_choice", "response_format", "seed", "n",
            "logprobs", "top_logprobs", "logit_bias", "parallel_tool_calls",
        ]
        for field in optional_fields:
            value = getattr(request, field, None)
            if value is not None:
                body[field] = value

        if request.tools is not None:
            body["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]

        return body

    # ------------------------------------------------------------------
    # Response conversion: OpenAI <-> IR
    # ------------------------------------------------------------------

    def response_to_ir(
        self, raw_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Convert an OpenAI-format response into the common IR.

        Handles extra fields like ``reasoning_content`` and detailed usage
        that may be present in OpenAI-compatible backends (e.g. DashScope).

        Args:
            raw_response: Raw OpenAI response dict.
            model: Model name fallback.

        Returns:
            ChatCompletionResponse in common IR format.
        """
        return ChatCompletionResponse(**raw_response)

    def ir_to_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        """Convert a common IR response into OpenAI wire format.

        Args:
            response: ChatCompletionResponse in common IR format.

        Returns:
            OpenAI-format response dict.
        """
        return response.model_dump(exclude_none=True)

    # ------------------------------------------------------------------
    # Streaming conversion: OpenAI <-> IR
    # ------------------------------------------------------------------

    def stream_line_to_ir(
        self, line: str, model: str, **kwargs: Any
    ) -> ChatCompletionStreamResponse | None:
        """Parse a single SSE line from OpenAI streaming response.

        Args:
            line: A single line from the SSE stream.
            model: The model name to use if not present in the response.
            **kwargs: Additional parameters (not used for OpenAI).

        Returns:
            A ChatCompletionStreamResponse if the line contains valid data,
            None if the line is empty, doesn't start with "data:", or is "[DONE]".
        """
        line = line.strip()

        if not line or not line.startswith("data:"):
            return None

        data_str = line[len("data:"):].strip()

        if data_str == "[DONE]":
            return None

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE data: %s", data_str)
            return None

        choices = []
        for raw_choice in data.get("choices", []):
            delta_data = raw_choice.get("delta", {})

            tool_calls_data = delta_data.get("tool_calls")
            tool_calls = None
            if tool_calls_data:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id"),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", ""),
                        ) if tc.get("function") else None,
                        index=tc.get("index"),
                    )
                    for tc in tool_calls_data
                ]

            choices.append(
                ChatCompletionStreamChoice(
                    index=raw_choice.get("index", 0),
                    delta=DeltaMessage(
                        role=delta_data.get("role"),
                        content=delta_data.get("content"),
                        reasoning_content=delta_data.get("reasoning_content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=raw_choice.get("finish_reason"),
                )
            )

        # Parse chunk-level usage if present
        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = UsageInfo(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
                completion_tokens_details=raw_usage.get("completion_tokens_details"),
                prompt_tokens_details=raw_usage.get("prompt_tokens_details"),
            )

        return ChatCompletionStreamResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
        )

    def ir_to_stream_chunk(
        self, chunk: ChatCompletionStreamResponse
    ) -> dict[str, Any]:
        """Convert a common IR stream chunk into OpenAI wire format.

        Args:
            chunk: ChatCompletionStreamResponse in common IR format.

        Returns:
            OpenAI-format streaming chunk dict.
        """
        return chunk.model_dump(exclude_none=True)

    def stream_done_signal(self) -> str | None:
        """Return the stream termination sentinel for OpenAI protocol.

        Returns:
            The string "[DONE]" used by OpenAI to signal stream end.
        """
        return "[DONE]"
