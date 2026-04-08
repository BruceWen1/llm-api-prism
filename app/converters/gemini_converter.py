# -*- coding: utf-8 -*-
"""
gemini_converter.py
Module: app.converters.gemini_converter
=====================

Protocol converter for Google Gemini API.
Handles bidirectional conversion between Gemini format and common IR.

Key transformations:
- Uses contents[{role, parts[{text}]}] structure for messages
- System instructions passed via systemInstruction field
- Role values are "user" and "model" (not "assistant")
- Function calls use functionCall/functionResponse in parts
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
    FunctionDefinition,
    Tool,
    ToolCall,
    UsageInfo,
)
from app.models.gemini_models import (
    GeminiCandidate,
    GeminiContent,
    GeminiPart,
    GeminiRequest,
    GeminiResponse,
    GeminiUsageMetadata,
)

logger = logging.getLogger(__name__)

# Map OpenAI role names to Gemini role names
GEMINI_ROLE_MAP = {"assistant": "model", "user": "user"}

# Map Gemini role names back to OpenAI role names
GEMINI_ROLE_REVERSE_MAP = {"model": "assistant", "user": "user"}

# Map Gemini finish reasons to OpenAI finish reasons
GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "stop",
}

# Map OpenAI finish reasons to Gemini finish reasons
OPENAI_TO_GEMINI_FINISH_REASON = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
}


class GeminiConverter(ProtocolConverter):
    """Converter for Google Gemini API protocol.

    Handles full protocol conversion lifecycle:
    - request_to_ir: Gemini generateContent -> OpenAI format
    - ir_to_request: OpenAI format -> Gemini generateContent
    - response_to_ir: Gemini response -> OpenAI format
    - ir_to_response: OpenAI format -> Gemini response
    - stream_line_to_ir: Gemini SSE line -> OpenAI stream chunk
    - ir_to_stream_chunk: OpenAI stream chunk -> Gemini SSE chunk
    """

    def request_to_ir(self, raw_body: dict[str, Any]) -> ChatCompletionRequest:
        """Convert Gemini generateContent request to OpenAI format.

        Args:
            raw_body: Raw Gemini request body.

        Returns:
            ChatCompletionRequest in OpenAI format.
        """
        gemini_request = GeminiRequest(**raw_body)

        messages: list[ChatMessage] = []

        # Check if system instruction exists
        if gemini_request.systemInstruction and gemini_request.systemInstruction.parts:
            system_text = gemini_request.systemInstruction.parts[0].text
            if system_text:
                messages.append(ChatMessage(role="system", content=system_text))

        role_mapping = {"model": "assistant", "user": "user"}

        # Pre-scan: build a mapping from function name → tool_call_id so that
        # functionResponse messages can reference the correct id generated for
        # the preceding functionCall.
        func_name_to_call_id: dict[str, str] = {}
        for content in gemini_request.contents:
            for part in content.parts:
                if part.functionCall:
                    func_name = part.functionCall.get("name", "")
                    if func_name and func_name not in func_name_to_call_id:
                        func_name_to_call_id[func_name] = f"call_{uuid.uuid4().hex[:12]}"

        # Process each content in the request
        for content in gemini_request.contents:
            openai_role = role_mapping.get(content.role, content.role)

            # Check if any part contains a functionCall (model -> tool_calls)
            function_call_parts = [p for p in content.parts if p.functionCall]
            # Check if any part contains a functionResponse (user -> tool message)
            function_response_parts = [p for p in content.parts if p.functionResponse]

            if function_call_parts:
                # Convert Gemini functionCall parts to OpenAI tool_calls
                tool_calls = []
                for part in function_call_parts:
                    fc = part.functionCall
                    func_name = fc.get("name", "")
                    call_id = func_name_to_call_id.get(func_name, f"call_{uuid.uuid4().hex[:12]}")
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=json.dumps(fc.get("args", {})),
                            ),
                        )
                    )
                messages.append(ChatMessage(role="assistant", content=None, tool_calls=tool_calls))
            elif function_response_parts:
                # Convert Gemini functionResponse parts to OpenAI tool messages
                for part in function_response_parts:
                    fr = part.functionResponse
                    func_name = fr.get("name", "unknown")
                    response_content = fr.get("response", {})
                    # Look up the tool_call_id from the pre-scanned mapping
                    call_id = func_name_to_call_id.get(func_name, f"call_{func_name}")
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps(response_content) if isinstance(response_content, dict) else str(response_content),
                            tool_call_id=call_id,
                        )
                    )
            else:
                # Regular text content
                content_text = "".join(part.text for part in content.parts if part.text is not None)
                messages.append(ChatMessage(role=openai_role, content=content_text))

        # Extract model name and generation config
        model_name = gemini_request.model or "gemini-pro"
        generation_config = gemini_request.generationConfig
        temperature = generation_config.temperature if generation_config else None
        top_p = generation_config.topP if generation_config else None
        max_tokens = generation_config.maxOutputTokens if generation_config else None
        stop = generation_config.stopSequences if generation_config else None

        presence_penalty = generation_config.presencePenalty if generation_config else None
        frequency_penalty = generation_config.frequencyPenalty if generation_config else None
        response_format = None
        # Check if response format is JSON
        if generation_config and generation_config.responseMimeType:
            if generation_config.responseMimeType == "application/json":
                response_format = {"type": "json_object"}
        n = generation_config.candidateCount if generation_config else None

        # Convert tools from Gemini to OpenAI format
        tools = None
        if gemini_request.tools:
            openai_tools = []
            # Process each Gemini tool
            for gemini_tool in gemini_request.tools:
                if gemini_tool.functionDeclarations:
                    # Process each function declaration
                    for func_decl in gemini_tool.functionDeclarations:
                        openai_tools.append(
                            Tool(
                                type="function",
                                function=FunctionDefinition(
                                    name=func_decl.name,
                                    description=func_decl.description,
                                    parameters=func_decl.parameters,
                                ),
                            )
                        )
            tools = openai_tools if openai_tools else None

        # Convert tool_choice from Gemini toolConfig to OpenAI format
        tool_choice = None
        if gemini_request.toolConfig and gemini_request.toolConfig.functionCallingConfig:
            mode = gemini_request.toolConfig.functionCallingConfig.get("mode", "AUTO")
            if mode == "AUTO":
                tool_choice = "auto"
            elif mode == "ANY":
                tool_choice = "required"
            elif mode == "NONE":
                tool_choice = "none"

        return ChatCompletionRequest(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            stream=False,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
            n=n,
            tools=tools,
            tool_choice=tool_choice,
        )

    def ir_to_request(self, request: ChatCompletionRequest, streaming: bool = False) -> dict[str, Any]:
        """Convert OpenAI format request to Gemini API format.

        Key transformations:
        - Extract system messages to systemInstruction field
        - Map assistant role to model role
        - Convert content strings to parts[{text}] structure

        Args:
            request: OpenAI chat completion request object.
            streaming: Whether to include streaming-specific fields.

        Returns:
            Dictionary containing Gemini API request body.
        """
        system_content = None
        contents = []

        # Build a mapping from tool_call_id to function name for tool result messages
        tool_call_id_to_name: dict[str, str] = {}
        for msg in request.messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.id and tool_call.function:
                        tool_call_id_to_name[tool_call.id] = tool_call.function.name

        # Process each message in the request
        for msg in request.messages:
            # Extract system content separately
            if msg.role == "system":
                system_content = msg.content
                continue

            # Convert assistant messages with tool_calls to Gemini functionCall parts
            if msg.role == "assistant" and msg.tool_calls:
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tool_call in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tool_call.function.name,
                            "args": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                        }
                    })
                contents.append({"role": "model", "parts": parts})
                continue

            # Convert tool result messages to Gemini functionResponse parts
            if msg.role == "tool":
                tool_call_id = msg.tool_call_id or ""
                # Look up the function name from preceding assistant tool_calls
                func_name = tool_call_id_to_name.get(tool_call_id, tool_call_id)
                response_content = msg.content or ""
                try:
                    response_data = json.loads(response_content)
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": response_content}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": func_name,
                            "response": response_data,
                        }
                    }],
                })
                continue

            # Map role name to Gemini format
            gemini_role = GEMINI_ROLE_MAP.get(msg.role, msg.role)
            contents.append({
                "role": gemini_role,
                "parts": [{"text": msg.content or ""}],
            })

        # Build base request body with contents
        body: dict = {"contents": contents}

        # Add system instruction if present
        if system_content:
            body["systemInstruction"] = {
                "parts": [{"text": system_content}],
            }

        # Build generation config with optional parameters
        generation_config: dict = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.stop is not None:
            # Handle both string and list stop sequences
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            generation_config["stopSequences"] = stop_sequences
        if request.presence_penalty is not None:
            generation_config["presencePenalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            generation_config["frequencyPenalty"] = request.frequency_penalty
        if request.n is not None:
            generation_config["candidateCount"] = request.n
        if request.response_format is not None:
            fmt_type = request.response_format.get("type", "")
            # Check if response format is JSON object
            if fmt_type == "json_object":
                generation_config["responseMimeType"] = "application/json"
            # Check if response format is JSON schema
            elif fmt_type == "json_schema":
                generation_config["responseMimeType"] = "application/json"
                schema = request.response_format.get("json_schema", {}).get("schema")
                if schema:
                    generation_config["responseSchema"] = schema

        # Add generation config only if it has parameters
        if generation_config:
            body["generationConfig"] = generation_config

        # Convert tools from OpenAI to Gemini format
        if request.tools:
            func_declarations = []
            # Process each tool
            for tool in request.tools:
                func_declarations.append({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                })
            body["tools"] = [{"functionDeclarations": func_declarations}]

        # Convert tool_choice from OpenAI to Gemini format
        if request.tool_choice is not None:
            # Handle auto mode
            if request.tool_choice == "auto":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
            # Handle required mode
            elif request.tool_choice == "required":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            # Handle none mode
            elif request.tool_choice == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            # Handle specific function mode
            elif isinstance(request.tool_choice, dict):
                func_name = request.tool_choice.get("function", {}).get("name", "")
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [func_name],
                    }
                }

        return body

    def response_to_ir(self, raw_response: dict[str, Any], model: str) -> ChatCompletionResponse:
        """Convert Gemini non-streaming response to OpenAI format.

        Args:
            raw_response: Gemini API response dictionary.
            model: Model name for the response.

        Returns:
            OpenAI format chat completion response.
        """
        choices = []

        # Process each candidate in the response
        for index, candidate in enumerate(raw_response.get("candidates", [])):
            # Extract text content from parts
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Check parts for function calls
            tool_calls = []
            text_parts = []
            for part in parts:
                # Check if part has text content
                if part.get("text"):
                    text_parts.append(part["text"])
                # Check if part has function call
                elif part.get("functionCall"):
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            type="function",
                            function=FunctionCall(
                                name=fc.get("name", ""),
                                arguments=json.dumps(fc.get("args", {})),
                            ),
                        )
                    )
            text = "".join(text_parts)

            # Map finish reason
            raw_reason = candidate.get("finishReason", "STOP")
            finish_reason = GEMINI_FINISH_REASON_MAP.get(raw_reason, "stop")

            # Build choice object
            choices.append(
                ChatCompletionChoice(
                    index=index,
                    message=ChatMessage(role="assistant", content=text, tool_calls=tool_calls if tool_calls else None),
                    finish_reason=finish_reason,
                )
            )

        # Extract usage metadata
        usage_data = raw_response.get("usageMetadata", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        # Build and return response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model,
            choices=choices,
            usage=usage,
        )

    def ir_to_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        """Encode OpenAI response to Gemini generateContent format.

        Args:
            response: OpenAI ChatCompletionResponse.

        Returns:
            Gemini format response dictionary.
        """
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason or "stop"
        # Map OpenAI finish reason to Gemini format
        gemini_finish_reason = OPENAI_TO_GEMINI_FINISH_REASON.get(
            finish_reason, finish_reason.upper()
        )

        parts = []
        # Check if message has text content
        if message.content:
            parts.append(GeminiPart(text=message.content))
        # Check if message has tool calls
        if message.tool_calls:
            # Process each tool call
            for tc in message.tool_calls:
                parts.append(GeminiPart(
                    functionCall={
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                    }
                ))
        # Ensure at least one part exists
        if not parts:
            parts.append(GeminiPart(text=""))

        candidate = GeminiCandidate(
            content=GeminiContent(
                role="model",
                parts=parts,
            ),
            finishReason=gemini_finish_reason,
        )

        # Build usage metadata if present
        usage_metadata = None
        if response.usage:
            usage_metadata = GeminiUsageMetadata(
                promptTokenCount=response.usage.prompt_tokens,
                candidatesTokenCount=response.usage.completion_tokens,
                totalTokenCount=response.usage.total_tokens,
            )

        gemini_response = GeminiResponse(
            candidates=[candidate],
            usageMetadata=usage_metadata,
            modelVersion=response.model,
        )

        return gemini_response.model_dump(exclude_none=True)

    def stream_line_to_ir(self, line: str, model: str, **kwargs: Any) -> ChatCompletionStreamResponse | None:
        """Parse Gemini SSE line and convert to OpenAI streaming format.

        Gemini streaming responses (alt=sse mode) have format: data: {json},
        where each JSON object contains a candidates array.

        Args:
            line: Single line from SSE stream.
            model: Model name for the response.
            **kwargs: Protocol-specific state including completion_id and is_first_chunk.

        Returns:
            OpenAI format stream response, or None if line should be skipped.
        """
        completion_id = kwargs.get("completion_id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        is_first_chunk = kwargs.get("is_first_chunk", True)

        # Remove leading/trailing whitespace
        line = line.strip()

        # Skip empty lines or lines not starting with "data:"
        if not line or not line.startswith("data:"):
            return None

        # Extract JSON string after "data:" prefix
        data_str = line[len("data:"):].strip()
        if not data_str:
            return None

        # Parse JSON data
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini SSE data: %s", data_str)
            return None

        # Check for candidates in response
        candidates = data.get("candidates", [])
        if not candidates:
            return None

        # Extract content from first candidate
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Check parts for function calls
        tool_calls = None
        text_parts = []
        for part in parts:
            # Check if part has text content
            if part.get("text"):
                text_parts.append(part["text"])
            # Check if part has function call
            elif part.get("functionCall"):
                fc = part["functionCall"]
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        type="function",
                        function=FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    )
                )
        text = "".join(text_parts)

        # Map finish reason if present
        raw_reason = candidate.get("finishReason")
        finish_reason = None
        if raw_reason:
            finish_reason = GEMINI_FINISH_REASON_MAP.get(raw_reason, "stop")

        # Build delta message with content
        delta = DeltaMessage(content=text if text else None, tool_calls=tool_calls)

        # Add role to first chunk
        if is_first_chunk:
            delta.role = "assistant"

        # Build and return stream response
        return ChatCompletionStreamResponse(
            id=completion_id,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )

    def ir_to_stream_chunk(self, chunk: ChatCompletionStreamResponse) -> dict[str, Any] | None:
        """Encode OpenAI streaming chunk to Gemini format.

        Returns ``None`` for chunks that carry no meaningful content
        (e.g. reasoning-only deltas), so the proxy layer can skip them.

        Args:
            chunk: OpenAI ChatCompletionStreamResponse.

        Returns:
            Gemini format stream chunk dictionary, or None to skip.
        """
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        parts = []
        if delta.content:
            parts.append({"text": delta.content})
        if delta.tool_calls:
            for tc in delta.tool_calls:
                args = {}
                if tc.function and tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # In streaming, arguments may be incremental partial JSON
                        args = {"_raw": tc.function.arguments}
                parts.append({
                    "functionCall": {
                        "name": tc.function.name if tc.function else "",
                        "args": args,
                    }
                })

        # Skip chunks that have no content, no tool calls, and no finish reason
        if not parts and not finish_reason:
            return None

        # Ensure at least one part exists for the Gemini format
        if not parts:
            parts.append({"text": ""})

        candidate_content: dict[str, Any] = {
            "content": {
                "role": "model",
                "parts": parts,
            }
        }

        if finish_reason:
            gemini_finish_reason = OPENAI_TO_GEMINI_FINISH_REASON.get(
                finish_reason, finish_reason.upper()
            )
            candidate_content["finishReason"] = gemini_finish_reason

        return {"candidates": [candidate_content]}

    def stream_done_signal(self) -> str | None:
        """Get the stream done signal for Gemini API.

        Returns:
            None as Gemini does not use a [DONE] signal.
        """
        return None
