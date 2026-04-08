# -*- coding: utf-8 -*-
"""
dashscope_models.py — Pydantic models for the Alibaba DashScope native API.

DashScope native protocol uses a nested structure:
  Request:  { model, input: { messages }, parameters: { result_format, ... } }
  Response: { output: { choices }, usage: { input_tokens, ... }, request_id }
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DashScopeMessage(BaseModel):
    """A single message in a DashScope conversation."""

    role: str = Field(..., description="Message role: system / user / assistant / tool")
    content: str | None = Field(default=None, description="Text content")
    tool_calls: list[DashScopeToolCall] | None = Field(default=None, description="Tool calls made by the assistant")
    tool_call_id: str | None = Field(default=None, description="ID of the tool call this message responds to")
    name: str | None = Field(default=None, description="Name of the tool")


class DashScopeFunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str = Field(default="", description="Function name")
    arguments: str = Field(default="", description="JSON-encoded function arguments")


class DashScopeToolCall(BaseModel):
    """A single tool call in an assistant message."""

    id: str | None = Field(default=None, description="Unique tool call identifier")
    type: str = Field(default="function", description="Tool type")
    function: DashScopeFunctionCall | None = Field(default=None, description="Function call details")
    index: int | None = Field(default=None, description="Index for streaming tool calls")


class DashScopeToolFunction(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class DashScopeTool(BaseModel):
    """Tool definition in DashScope parameters."""

    type: str = "function"
    function: DashScopeToolFunction


class DashScopeInput(BaseModel):
    """The `input` object in a DashScope request."""

    messages: list[DashScopeMessage] = Field(..., description="Conversation messages")


class DashScopeParameters(BaseModel):
    """The `parameters` object in a DashScope request."""

    result_format: str | None = Field(default="message", description="Output format: message or text")
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    n: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[DashScopeTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    incremental_output: bool | None = Field(default=None, description="Enable incremental streaming output")


class DashScopeRequest(BaseModel):
    """Request model for DashScope text-generation API."""

    model: str = Field(..., description="Model identifier, e.g. qwen-plus")
    input: DashScopeInput = Field(..., description="Input containing messages")
    parameters: DashScopeParameters | None = Field(default=None, description="Generation parameters")
    stream: bool | None = Field(default=None, description="Alternative streaming flag")


class DashScopeChoiceMessage(BaseModel):
    """Message within a response choice."""

    role: str = Field(default="assistant", description="Message role")
    content: str | None = Field(default=None, description="Text content")
    tool_calls: list[DashScopeToolCall] | None = Field(default=None, description="Tool calls")


class DashScopeChoice(BaseModel):
    """A single choice in the DashScope response output."""

    message: DashScopeChoiceMessage = Field(..., description="The generated message")
    finish_reason: str | None = Field(default=None, description="Reason for finishing: stop, tool_calls, or 'null' (in-progress)")


class DashScopeOutput(BaseModel):
    """The `output` object in a DashScope response."""

    choices: list[DashScopeChoice] = Field(default_factory=list, description="List of generated choices")


class DashScopeUsage(BaseModel):
    """Token usage information in a DashScope response."""

    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")


class DashScopeResponse(BaseModel):
    """Response model for DashScope text-generation API."""

    output: DashScopeOutput = Field(..., description="Generation output")
    usage: DashScopeUsage | None = Field(default=None, description="Token usage")
    request_id: str | None = Field(default=None, description="Unique request identifier")


# Rebuild DashScopeMessage to resolve forward reference to DashScopeToolCall
DashScopeMessage.model_rebuild()
