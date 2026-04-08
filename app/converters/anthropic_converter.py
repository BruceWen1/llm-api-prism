# -*- coding: utf-8 -*-
"""
anthropic_converter.py — Protocol converter for the Anthropic Messages API.

Extracts and encapsulates all protocol conversion logic from AnthropicAdapter.
Handles bidirectional transformation between Anthropic wire format and the
common intermediate representation (IR).
"""

import json
import logging
import uuid
from typing import Any

from app.converters.base import ProtocolConverter
from app.models.anthropic_models import (
    AnthropicContentBlock,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
)
from app.models.common_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    FunctionDefinition,
    Tool,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)

# Map Anthropic finish reasons to OpenAI format
ANTHROPIC_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}

# Map OpenAI finish reasons to Anthropic format
OPENAI_TO_ANTHROPIC_FINISH_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

# Default max tokens when not specified in request
DEFAULT_MAX_TOKENS = 4096


class AnthropicConverter(ProtocolConverter):
    """Converter for the Anthropic Messages API protocol.

    Implements pure data transformation between Anthropic wire format and
    the common IR. No HTTP, networking, or streaming transport concerns.
    """

    def __init__(self) -> None:
        self._message_started = False
        self._text_block_started = False
        self._current_tool_block_index = 0

    # ------------------------------------------------------------------
    # Request conversion: Anthropic <-> IR
    # ------------------------------------------------------------------

    def request_to_ir(self, raw_body: dict[str, Any]) -> ChatCompletionRequest:
        """Convert an Anthropic-format request into the common IR.

        Args:
            raw_body: Raw Anthropic request body.

        Returns:
            ChatCompletionRequest in common IR format.
        """
        anthropic_request = AnthropicRequest(**raw_body)

        messages: list[ChatMessage] = []

        # Check if system message exists
        if anthropic_request.system:
            messages.append(ChatMessage(role="system", content=anthropic_request.system))

        # Process each message in the request
        for anthropic_msg in anthropic_request.messages:
            # Check if message content is a list (multi-modal content)
            if isinstance(anthropic_msg.content, list):
                text_parts = []
                msg_tool_calls = []
                tool_results = []

                # Process each content block
                for block in anthropic_msg.content:
                    # Check if block is text type
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    # Check if block is tool_use type
                    elif block.get("type") == "tool_use":
                        msg_tool_calls.append(
                            ToolCall(
                                id=block.get("id", ""),
                                type="function",
                                function=FunctionCall(
                                    name=block.get("name", ""),
                                    arguments=json.dumps(block.get("input", {})),
                                ),
                            )
                        )
                    # Check if block is tool_result type
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        result_text_parts = []
                        if isinstance(result_content, str):
                            result_text_parts.append(result_content)
                        elif isinstance(result_content, list):
                            for sub_block in result_content:
                                if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                                    result_text_parts.append(sub_block.get("text", ""))
                                else:
                                    result_text_parts.append(json.dumps(sub_block, ensure_ascii=False))
                        elif result_content is not None:
                            result_text_parts.append(json.dumps(result_content, ensure_ascii=False))
                        tool_results.append({
                            "tool_call_id": block.get("tool_use_id"),
                            "content": "".join(result_text_parts),
                        })

                # Build content string from text parts (for non-tool-result messages)
                content_str = "".join(text_parts) if text_parts else None

                # Check if message contains tool calls
                if msg_tool_calls:
                    messages.append(ChatMessage(
                        role=anthropic_msg.role,
                        content=content_str,
                        tool_calls=msg_tool_calls,
                    ))
                # Check if message contains tool results — emit one ChatMessage per result
                elif tool_results:
                    for tool_result in tool_results:
                        messages.append(ChatMessage(
                            role="tool",
                            content=tool_result["content"] or "",
                            tool_call_id=tool_result["tool_call_id"],
                        ))
                # Otherwise, add as regular message
                else:
                    messages.append(ChatMessage(role=anthropic_msg.role, content=content_str))
            # Handle simple text content
            else:
                messages.append(ChatMessage(role=anthropic_msg.role, content=anthropic_msg.content))

        # Convert tools from Anthropic to OpenAI format
        tools = None
        if anthropic_request.tools:
            tools = [
                Tool(
                    type="function",
                    function=FunctionDefinition(
                        name=t.name,
                        description=t.description,
                        parameters=t.input_schema.model_dump(exclude_none=True) if t.input_schema else None,
                    ),
                )
                for t in anthropic_request.tools
            ]

        # Convert tool_choice from Anthropic to OpenAI format
        tool_choice = None
        if anthropic_request.tool_choice:
            tc = anthropic_request.tool_choice
            tc_type = tc.get("type")
            # Map auto type
            if tc_type == "auto":
                tool_choice = "auto"
            # Map any type to required
            elif tc_type == "any":
                tool_choice = "required"
            # Map none type to none
            elif tc_type == "none":
                tool_choice = "none"
            # Map tool type with specific function
            elif tc_type == "tool":
                tool_choice = {"type": "function", "function": {"name": tc.get("name", "")}}

        return ChatCompletionRequest(
            model=anthropic_request.model,
            messages=messages,
            max_tokens=anthropic_request.max_tokens,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            stop=anthropic_request.stop_sequences,
            stream=anthropic_request.stream,
            tools=tools,
            tool_choice=tool_choice,
        )

    def ir_to_request(
        self, request: ChatCompletionRequest, streaming: bool = False
    ) -> dict[str, Any]:
        """Convert a common IR request into Anthropic wire format.

        Args:
            request: ChatCompletionRequest in common IR format.
            streaming: Whether to include streaming-specific fields.

        Returns:
            Request body dict in Anthropic format.
        """
        # Initialize system content parts and messages list
        system_parts: list[str] = []
        messages = []

        # Iterate through messages to separate system from others
        for msg in request.messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == "tool":
                # Convert OpenAI tool result to Anthropic tool_result content block
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content or "",
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Convert assistant message with tool_calls to Anthropic format
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments) if tc.function.arguments else {},
                    })
                messages.append({"role": "assistant", "content": content_blocks})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        # Build base request body with required fields
        body: dict = {
            "model": request.model,
            "messages": messages,
            # Use provided max_tokens or default value
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }

        # Add optional system content if present (merge multiple system messages)
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        # Add stream flag if requested
        if streaming:
            body["stream"] = True

        # Add temperature parameter if specified
        if request.temperature is not None:
            body["temperature"] = request.temperature

        # Add top_p parameter if specified
        if request.top_p is not None:
            body["top_p"] = request.top_p

        # Convert stop parameter to stop_sequences array
        if request.stop is not None:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            body["stop_sequences"] = stop_sequences

        # Convert tools from OpenAI format to Anthropic format
        if request.tools is not None:
            anthropic_tools = []
            for tool in request.tools:
                tool_def: dict[str, Any] = {
                    "name": tool.function.name,
                }
                if tool.function.description:
                    tool_def["description"] = tool.function.description
                if tool.function.parameters:
                    tool_def["input_schema"] = tool.function.parameters
                else:
                    tool_def["input_schema"] = {"type": "object", "properties": {}}
                anthropic_tools.append(tool_def)
            body["tools"] = anthropic_tools

        # Convert tool_choice from OpenAI format to Anthropic format
        if request.tool_choice is not None:
            if request.tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif request.tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                body["tool_choice"] = {"type": "none"}
            elif isinstance(request.tool_choice, dict):
                func_name = request.tool_choice.get("function", {}).get("name", "")
                body["tool_choice"] = {"type": "tool", "name": func_name}

        return body

    # ------------------------------------------------------------------
    # Response conversion: Anthropic <-> IR
    # ------------------------------------------------------------------

    def response_to_ir(
        self, raw_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Convert an Anthropic-format response into the common IR.

        Args:
            raw_response: Raw Anthropic response dict.
            model: Model name fallback.

        Returns:
            ChatCompletionResponse in common IR format.
        """
        # Extract text content from response blocks
        content_parts = []
        tool_calls = []
        for block in raw_response.get("content", []):
            # Check if block is text type
            if block.get("type") == "text":
                content_parts.append(block.get("text", ""))
            # Check if block is tool_use type
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function=FunctionCall(
                            name=block.get("name", ""),
                            arguments=json.dumps(block.get("input", {})),
                        ),
                    )
                )

        # Concatenate all text parts into full content
        full_content = "".join(content_parts)

        # Map Anthropic finish reason to OpenAI format
        raw_finish_reason = raw_response.get("stop_reason", "end_turn")
        finish_reason = ANTHROPIC_FINISH_REASON_MAP.get(raw_finish_reason, "stop")

        # Extract and build usage information
        usage_data = raw_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(
                usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0)
            ),
        )

        # Build and return OpenAI-formatted response
        return ChatCompletionResponse(
            id=raw_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            model=raw_response.get("model", model),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=full_content,
                        tool_calls=tool_calls if tool_calls else None
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    def ir_to_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        """Convert a common IR response into Anthropic wire format.

        Args:
            response: ChatCompletionResponse in common IR format.

        Returns:
            Anthropic-format response dict.
        """
        message = response.choices[0].message
        content_blocks = []

        # Check if message has text content
        if message.content:
            content_blocks.append(
                AnthropicContentBlock(type="text", text=message.content)
            )

        # Check if message has tool calls
        if message.tool_calls:
            # Process each tool call
            for tc in message.tool_calls:
                content_blocks.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        # Map OpenAI finish reason to Anthropic format
        stop_reason = OPENAI_TO_ANTHROPIC_FINISH_REASON.get(
            response.choices[0].finish_reason,
            response.choices[0].finish_reason,
        )

        # Build usage information if present
        usage = None
        if response.usage:
            usage = AnthropicUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        # Create Anthropic response object
        anthropic_response = AnthropicResponse(
            id=response.id,
            content=content_blocks,
            model=response.model,
            stop_reason=stop_reason,
            usage=usage,
        )

        return anthropic_response.model_dump(exclude_none=True)

    # ------------------------------------------------------------------
    # Streaming conversion: Anthropic <-> IR
    # ------------------------------------------------------------------

    def stream_line_to_ir(
        self, line: str, model: str, **kwargs: Any
    ) -> ChatCompletionStreamResponse | None:
        """Parse a single SSE line from Anthropic streaming response.

        Args:
            line: Raw SSE line from Anthropic API.
            model: Model name to use in response.
            **kwargs: Protocol-specific state (requires completion_id).

        Returns:
            ChatCompletionStreamResponse, or None if line should be skipped.
        """
        line = line.strip()
        # Skip empty lines
        if not line:
            return None

        # Skip event type lines (we only process data lines)
        if line.startswith("event:"):
            return None

        # Skip lines that don't start with data:
        if not line.startswith("data:"):
            return None

        # Extract data portion after "data:" prefix
        data_str = line[len("data:"):].strip()
        # Skip if data string is empty
        if not data_str:
            return None

        # Parse JSON data from line
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Anthropic SSE data: %s", data_str)
            return None

        # Get completion_id from kwargs
        completion_id = kwargs.get("completion_id", f"chatcmpl-{uuid.uuid4().hex[:12]}")

        # Get event type from parsed data
        event_type = data.get("type", "")

        # Handle message_start event - send initial response with role
        if event_type == "message_start":
            return ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )

        # Handle content_block_start event — tool_use blocks carry id + name
        if event_type == "content_block_start":
            content_block = data.get("content_block", {})
            if content_block.get("type") == "tool_use":
                block_index = data.get("index", 0)
                return ChatCompletionStreamResponse(
                    id=completion_id,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                tool_calls=[
                                    ToolCall(
                                        id=content_block.get("id", ""),
                                        type="function",
                                        function=FunctionCall(
                                            name=content_block.get("name", ""),
                                            arguments="",
                                        ),
                                        index=block_index,
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                )
            # text content_block_start — skip, content comes in deltas
            return None

        # Handle content_block_delta event
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")

            # Text content delta
            if delta_type == "text_delta":
                text = delta.get("text", "")
                return ChatCompletionStreamResponse(
                    id=completion_id,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=text),
                            finish_reason=None,
                        )
                    ],
                )

            # Tool use input JSON delta
            if delta_type == "input_json_delta":
                block_index = data.get("index", 0)
                partial_json = delta.get("partial_json", "")
                return ChatCompletionStreamResponse(
                    id=completion_id,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                tool_calls=[
                                    ToolCall(
                                        type="function",
                                        function=FunctionCall(
                                            name="",
                                            arguments=partial_json,
                                        ),
                                        index=block_index,
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                )

            return None

        # Handle content_block_stop event — skip, no IR equivalent needed
        if event_type == "content_block_stop":
            return None

        # Handle message_delta event - send finish reason
        if event_type == "message_delta":
            # Extract delta content
            delta = data.get("delta", {})
            raw_reason = delta.get("stop_reason", "end_turn")
            # Map Anthropic finish reason to OpenAI format
            finish_reason = ANTHROPIC_FINISH_REASON_MAP.get(raw_reason, "stop")
            return ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
            )

        # Skip unhandled event types
        return None

    def ir_to_stream_chunk(
        self, chunk: ChatCompletionStreamResponse
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Convert a common IR stream chunk into Anthropic wire format.

        The returned dict (or list of dicts) may contain a ``_event`` key
        specifying the SSE event type (e.g. ``message_start``,
        ``content_block_delta``).  The proxy layer uses this to emit a
        proper ``event:`` line before the ``data:`` line, which is required
        by the Anthropic SSE protocol.

        Returns a **list** when a single IR chunk maps to multiple Anthropic
        SSE events (e.g. content_block_start + first delta).

        Returns ``None`` to signal that the chunk should be skipped entirely
        (e.g. reasoning-only deltas that have no Anthropic representation).

        Args:
            chunk: ChatCompletionStreamResponse in common IR format.

        Returns:
            Anthropic format stream event dict/list (with optional ``_event``),
            or None if the chunk should be dropped.
        """
        delta = chunk.choices[0].delta

        # Emit message_start exactly once (the first chunk in the stream).
        if not self._message_started:
            self._message_started = True
            return {
                "_event": "message_start",
                "type": "message_start",
                "message": {
                    "id": chunk.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": chunk.model,
                },
            }

        # Text content delta — emit content_block_start on first text chunk
        if delta.content:
            events: list[dict[str, Any]] = []
            if not self._text_block_started:
                self._text_block_started = True
                events.append({
                    "_event": "content_block_start",
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
            events.append({
                "_event": "content_block_delta",
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta.content},
            })
            return events if len(events) > 1 else events[0]

        # Tool call deltas — emit all tool_call events, not just the first
        if delta.tool_calls:
            events = []
            # Close the text block before starting tool blocks
            if self._text_block_started:
                self._text_block_started = False
                events.append({
                    "_event": "content_block_stop",
                    "type": "content_block_stop",
                    "index": 0,
                })

            for tool_call in delta.tool_calls:
                if not (tool_call and tool_call.function):
                    continue
                block_index = tool_call.index if tool_call.index is not None else self._current_tool_block_index
                if tool_call.function.name:
                    # New tool_use block — track the index for subsequent argument deltas
                    self._current_tool_block_index = block_index
                    events.append({
                        "_event": "content_block_start",
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                        },
                    })
                if tool_call.function.arguments:
                    events.append({
                        "_event": "content_block_delta",
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": tool_call.function.arguments},
                    })

            if len(events) == 0:
                return None
            return events if len(events) > 1 else events[0]

        # Finish reason — close any open blocks first
        if chunk.choices[0].finish_reason:
            events = []
            if self._text_block_started:
                self._text_block_started = False
                events.append({
                    "_event": "content_block_stop",
                    "type": "content_block_stop",
                    "index": 0,
                })

            stop_reason = OPENAI_TO_ANTHROPIC_FINISH_REASON.get(
                chunk.choices[0].finish_reason,
                chunk.choices[0].finish_reason,
            )
            events.append({
                "_event": "message_delta",
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": None,
            })
            return events if len(events) > 1 else events[0]

        # Skip reasoning-only chunks — Anthropic has no equivalent representation
        return None

    def stream_done_signal(self) -> str | None:
        """Return the stream termination sentinel for Anthropic protocol.

        Returns:
            None as Anthropic uses message_stop event instead of [DONE].
        """
        # Reset stateful flags so the converter can be safely reused
        self._message_started = False
        self._text_block_started = False
        self._current_tool_block_index = 0
        return None
