# -*- coding: utf-8 -*-
"""
openai_models.py — Pydantic models for the OpenAI Chat Completion API.

These models define the wire format of the OpenAI protocol. They are
structurally identical to the common IR (common_models.py) because the IR
was designed after the OpenAI format — but they are conceptually separate.

This file exists so that every protocol (OpenAI included) has its own
models file, keeping all four adapters symmetrical.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpenAIFunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAITool(BaseModel):
    """Tool definition in an OpenAI request."""

    type: str = "function"
    function: OpenAIFunctionDefinition


class OpenAIFunctionCall(BaseModel):
    """Function call in an assistant message."""

    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    """Tool call in an assistant message or streaming delta."""

    id: str | None = None
    type: str = "function"
    function: OpenAIFunctionCall | None = None
    index: int | None = None


class OpenAIMessage(BaseModel):
    """A single message in an OpenAI conversation."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIRequest(BaseModel):
    """Request model for OpenAI Chat Completion API."""

    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = None


class OpenAIDeltaMessage(BaseModel):
    """Delta message for streaming responses."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(BaseModel):
    """A single choice in a non-streaming response."""

    index: int
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIStreamChoice(BaseModel):
    """A single choice in a streaming response chunk."""

    index: int
    delta: OpenAIDeltaMessage
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIResponse(BaseModel):
    """Response model for non-streaming OpenAI Chat Completion API."""

    id: str
    object: str = "chat.completion"
    created: int | None = None
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None


class OpenAIStreamResponse(BaseModel):
    """Response model for streaming OpenAI Chat Completion API (SSE chunk)."""

    id: str
    object: str = "chat.completion.chunk"
    created: int | None = None
    model: str
    choices: list[OpenAIStreamChoice]
    usage: OpenAIUsage | None = None
