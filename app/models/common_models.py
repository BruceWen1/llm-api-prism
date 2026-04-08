# -*- coding: utf-8 -*-
"""
common_models.py — Intermediate representation (IR) models for LLM API Prism.

These Pydantic models define the hub format used internally by all adapters.
The structure follows the OpenAI Chat Completion API convention because it is
the de-facto industry standard, but these models are **not** tied to OpenAI —
they serve as the common language between any input protocol and any backend.
"""

import time
from typing import Any

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    """
    FunctionDefinition
    Function definition for tool calling.
    """
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """
    Tool
    Tool definition for function calling.
    """
    type: str = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """
    FunctionCall
    Function call in assistant message.
    """
    name: str
    arguments: str


class ToolCall(BaseModel):
    """
    ToolCall
    Tool call in assistant message or streaming delta.
    """
    id: str | None = None
    type: str = "function"
    function: FunctionCall | None = None
    index: int | None = None


class ChatMessage(BaseModel):
    """
    ChatMessage
    Represents a single message in a chat conversation.
    """
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """
    ChatCompletionRequest
    Request model for OpenAI Chat Completion API.
    """
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = None


class ChatCompletionChoice(BaseModel):
    """
    ChatCompletionChoice
    Represents a single choice in the chat completion response.
    """
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    """
    UsageInfo
    Token usage information for the API request.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: dict[str, Any] | None = None
    prompt_tokens_details: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """
    ChatCompletionResponse
    Response model for non-streaming OpenAI Chat Completion API.
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


class DeltaMessage(BaseModel):
    """
    DeltaMessage
    Represents a delta message for streaming responses.
    """
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionStreamChoice(BaseModel):
    """
    ChatCompletionStreamChoice
    Represents a single choice in the streaming chat completion response.
    """
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """
    ChatCompletionStreamResponse
    Response model for streaming OpenAI Chat Completion API (SSE chunk).
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageInfo | None = None