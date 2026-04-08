# -*- coding: utf-8 -*-
"""
dashscope_converter.py
Module: app.converters.dashscope_converter
=====================

Protocol converter for Alibaba DashScope native HTTP API (not the OpenAI-compatible mode).

Native DashScope protocol structure:
  Request:  { model, input: { messages }, parameters: { result_format, tools, ... } }
  Response: { output: { choices: [{ message, finish_reason }] }, usage: { input_tokens, output_tokens }, request_id }
  Streaming: Each SSE data line mirrors the response structure
"""

import json
import logging
import uuid
from typing import Any

from app.converters.base import ProtocolConverter
from app.models.common_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from app.models.dashscope_models import DashScopeRequest, DashScopeResponse

logger = logging.getLogger(__name__)


class DashScopeConverter(ProtocolConverter):
    """Converter for the Alibaba DashScope native HTTP API.

    Uses the DashScope-native protocol where messages live inside an `input`
    object and generation parameters live inside a `parameters` object.
    """

    def request_to_ir(self, raw_body: dict) -> ChatCompletionRequest:
        """Decode a DashScope-native request into the common intermediate format.

        Args:
            raw_body: Raw DashScope request body.

        Returns:
            ChatCompletionRequest in common intermediate format.
        """
        dashscope_request = DashScopeRequest(**raw_body)
        parameters = dashscope_request.parameters

        messages = [
            msg.model_dump(exclude_none=True)
            for msg in dashscope_request.input.messages
        ]

        is_streaming = (
            (parameters.incremental_output if parameters else False)
            or dashscope_request.stream
            or False
        )

        kwargs: dict = {
            "model": dashscope_request.model,
            "messages": messages,
            "stream": bool(is_streaming),
        }

        if parameters:
            field_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "max_tokens": "max_tokens",
                "stop": "stop",
                "presence_penalty": "presence_penalty",
                "seed": "seed",
                "n": "n",
                "response_format": "response_format",
                "tools": "tools",
                "tool_choice": "tool_choice",
            }
            for param_field, request_field in field_mapping.items():
                value = getattr(parameters, param_field, None)
                if value is not None:
                    if param_field == "tools":
                        kwargs[request_field] = [
                            t.model_dump(exclude_none=True) for t in value
                        ]
                    else:
                        kwargs[request_field] = value

        return ChatCompletionRequest(**kwargs)

    def ir_to_request(self, request: ChatCompletionRequest, streaming: bool = False) -> dict[str, Any]:
        """Convert an OpenAI ChatCompletionRequest to a DashScope native request body.

        DashScope native protocol structure:
          {
            "model": "...",
            "input": { "messages": [...] },
            "parameters": {
              "result_format": "message",
              "temperature": ...,
              "tools": [...],   # tool definitions go here, not top-level
              ...
            }
          }

        Args:
            request: The OpenAI-format chat completion request.
            streaming: Whether to add incremental_output for streaming mode.

        Returns:
            A dict ready to be serialised as the DashScope request body.
        """
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        parameters: dict = {
            # Always request message-format output so we get role/content/tool_calls
            "result_format": "message",
        }

        if streaming:
            parameters["incremental_output"] = True

        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        if request.top_p is not None:
            parameters["top_p"] = request.top_p
        if request.max_tokens is not None:
            parameters["max_tokens"] = request.max_tokens
        if request.stop is not None:
            parameters["stop"] = request.stop
        if request.presence_penalty is not None:
            parameters["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            parameters["seed"] = request.seed
        if request.n is not None:
            parameters["n"] = request.n
        if request.response_format is not None:
            parameters["response_format"] = request.response_format

        # In DashScope native protocol, tools and tool_choice live in parameters
        if request.tools is not None:
            parameters["tools"] = [
                tool.model_dump(exclude_none=True) for tool in request.tools
            ]
        if request.tool_choice is not None:
            parameters["tool_choice"] = request.tool_choice

        return {
            "model": request.model,
            "input": {"messages": messages},
            "parameters": parameters,
        }

    def response_to_ir(self, raw_response: dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse a DashScope native non-streaming response into common format.

        Args:
            raw_response: Raw JSON response from DashScope.
            model: Model name to use as fallback.

        Returns:
            A ChatCompletionResponse in common intermediate format.
        """
        dashscope_response = DashScopeResponse(**raw_response)

        choices = []
        for index, ds_choice in enumerate(dashscope_response.output.choices):
            message_data = ds_choice.message
            tool_calls = None
            if message_data.tool_calls:
                tool_calls = self._parse_tool_calls(
                    [tc.model_dump(exclude_none=True) for tc in message_data.tool_calls]
                )

            choices.append(
                ChatCompletionChoice(
                    index=index,
                    message=ChatMessage(
                        role=message_data.role,
                        content=message_data.content or "",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=ds_choice.finish_reason,
                )
            )

        usage = UsageInfo(
            prompt_tokens=dashscope_response.usage.input_tokens if dashscope_response.usage else 0,
            completion_tokens=dashscope_response.usage.output_tokens if dashscope_response.usage else 0,
            total_tokens=dashscope_response.usage.total_tokens if dashscope_response.usage else 0,
        )

        request_id = dashscope_response.request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return ChatCompletionResponse(
            id=request_id,
            model=model,
            choices=choices,
            usage=usage,
        )

    def ir_to_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        """Encode an OpenAI ChatCompletionResponse into DashScope-native format.

        DashScope response structure:
          {
            "output": {
              "choices": [
                { "message": { "role", "content", "tool_calls" }, "finish_reason" }
              ]
            },
            "usage": { "input_tokens", "output_tokens", "total_tokens" },
            "request_id": "..."
          }

        Args:
            response: OpenAI-format ChatCompletionResponse.

        Returns:
            DashScope-native response dict.
        """
        choices = []
        for choice in response.choices:
            message: dict = {
                "role": choice.message.role,
                "content": choice.message.content or "",
            }
            if choice.message.reasoning_content:
                message["reasoning_content"] = choice.message.reasoning_content
            if choice.message.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            choices.append({
                "message": message,
                "finish_reason": choice.finish_reason or "stop",
            })

        usage = response.usage
        return {
            "output": {"choices": choices},
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "request_id": response.id,
        }

    def stream_line_to_ir(self, line: str, model: str, **kwargs: Any) -> ChatCompletionStreamResponse | None:
        """Parse a single SSE line from DashScope streaming response.

        DashScope streaming SSE lines look like:
          data: {"output":{"choices":[{"message":{"role":"assistant","content":"..."},"finish_reason":"null"}]},...}

        With incremental_output=true, each chunk contains only the new delta.
        The finish_reason is "stop" (or other) on the final chunk, "null" otherwise.

        Args:
            line: A raw SSE line string.
            model: Model name to use as fallback.
            **kwargs: Protocol-specific state (not used for DashScope).

        Returns:
            An OpenAI-formatted ChatCompletionStreamResponse, or None if the
            line should be skipped (empty, comment, or unparseable).
        """
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None

        data_str = line[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            return None

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse DashScope SSE data: %s", data_str)
            return None

        output = data.get("output", {})
        raw_choices = output.get("choices", [])
        request_id = data.get("request_id", f"chatcmpl-{uuid.uuid4().hex[:12]}")

        choices = []
        for index, raw_choice in enumerate(raw_choices):
            # DashScope streaming uses "message" (not "delta") with incremental content
            delta_data = raw_choice.get("message", {})

            tool_calls_data = delta_data.get("tool_calls")
            tool_calls = self._parse_tool_calls(tool_calls_data) if tool_calls_data else None

            # DashScope uses "null" string (not JSON null) when generation is ongoing
            raw_finish_reason = raw_choice.get("finish_reason")
            finish_reason = None if raw_finish_reason in (None, "null") else raw_finish_reason

            choices.append(
                ChatCompletionStreamChoice(
                    index=index,
                    delta=DeltaMessage(
                        role=delta_data.get("role"),
                        content=delta_data.get("content"),
                        reasoning_content=delta_data.get("reasoning_content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            )

        # Parse chunk-level usage if present
        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = UsageInfo(
                prompt_tokens=raw_usage.get("input_tokens", 0),
                completion_tokens=raw_usage.get("output_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            )

        return ChatCompletionStreamResponse(
            id=request_id,
            model=model,
            choices=choices,
            usage=usage,
        )

    def ir_to_stream_chunk(self, chunk: ChatCompletionStreamResponse) -> dict[str, Any]:
        """Encode an OpenAI streaming chunk into DashScope-native SSE format.

        DashScope streaming chunk structure:
          {
            "output": {
              "choices": [
                { "message": { "role", "content" }, "finish_reason": "null" | "stop" }
              ]
            },
            "request_id": "..."
          }

        Args:
            chunk: OpenAI-format ChatCompletionStreamResponse.

        Returns:
            DashScope-native streaming chunk dict.
        """
        choices = []
        for choice in chunk.choices:
            delta = choice.delta
            message: dict = {}
            if delta.role is not None:
                message["role"] = delta.role
            if delta.content is not None:
                message["content"] = delta.content
            if delta.reasoning_content is not None:
                message["reasoning_content"] = delta.reasoning_content
            if delta.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name if tc.function else "",
                            "arguments": tc.function.arguments if tc.function else "",
                        },
                        "index": tc.index,
                    }
                    for tc in delta.tool_calls
                ]
            # DashScope uses the string "null" (not JSON null) for in-progress chunks
            finish_reason = choice.finish_reason if choice.finish_reason else "null"
            choices.append({"message": message, "finish_reason": finish_reason})

        return {
            "output": {"choices": choices},
            "request_id": chunk.id,
        }

    def stream_done_signal(self) -> str | None:
        """DashScope does not send a [DONE] sentinel; the last chunk carries finish_reason.

        Returns:
            None — no additional done signal is needed.
        """
        return None

    def _parse_tool_calls(self, tool_calls_data: list) -> list[ToolCall]:
        """Parse a DashScope tool_calls array into OpenAI ToolCall objects.

        Args:
            tool_calls_data: Raw tool_calls list from DashScope response.

        Returns:
            A list of OpenAI-format ToolCall objects.
        """
        result = []
        for index, tc in enumerate(tool_calls_data):
            function_data = tc.get("function", {})
            result.append(
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=function_data.get("name", ""),
                        arguments=function_data.get("arguments", ""),
                    ),
                    index=tc.get("index", index),
                )
            )
        return result
