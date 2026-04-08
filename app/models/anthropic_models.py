# -*- coding: utf-8 -*-
"""
anthropic_models.py
Anthropic API Models
=====================

Pydantic models for Anthropic Messages API requests and responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnthropicToolInputSchema(BaseModel):
    """
    AnthropicToolInputSchema
    JSON Schema for tool input parameters.
    """
    type: str = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class AnthropicTool(BaseModel):
    """
    AnthropicTool
    Tool definition for Anthropic API.
    """
    name: str
    description: str | None = None
    input_schema: AnthropicToolInputSchema | None = None


class AnthropicToolUse(BaseModel):
    """
    AnthropicToolUse
    Tool use content block in response.
    """
    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResult(BaseModel):
    """
    AnthropicToolResult
    Tool result content block in request.
    """
    type: str = "tool_result"
    tool_use_id: str
    content: str


class AnthropicMessage(BaseModel):
    """
    AnthropicMessage
    Represents a single message in the conversation for Anthropic API.
    """
    role: str = Field(..., description="The role of the message sender")
    content: str | list[dict[str, Any]] = Field(..., description="The text content or list of content blocks")


class AnthropicRequest(BaseModel):
    """
    AnthropicRequest
    Request model for Anthropic Messages API.
    """
    model: str = Field(..., description="The model identifier to use")
    messages: list[AnthropicMessage] = Field(..., description="List of conversation messages")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    system: str | None = Field(default=None, description="System prompt for behavior guidance")
    temperature: float | None = Field(default=None, description="Sampling temperature")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    stop_sequences: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream the response")
    tools: list[AnthropicTool] | None = Field(default=None, description="List of available tools")
    tool_choice: dict[str, Any] | None = Field(default=None, description="Tool choice strategy")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class AnthropicContentBlock(BaseModel):
    """
    AnthropicContentBlock
    Represents a content block in Anthropic API response.
    """
    type: str = Field(default="text", description="The type of content block")
    text: str | None = Field(default=None, description="The text content")
    id: str | None = Field(default=None, description="The ID of the tool use block")
    name: str | None = Field(default=None, description="The name of the tool")
    input: dict[str, Any] | None = Field(default=None, description="The input parameters for the tool")


class AnthropicUsage(BaseModel):
    """
    AnthropicUsage
    Token usage information for Anthropic API response.
    """
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")


class AnthropicResponse(BaseModel):
    """
    AnthropicResponse
    Response model for Anthropic Messages API.
    """
    id: str = Field(..., description="Unique response identifier")
    type: str = Field(default="message", description="The type of response")
    role: str = Field(default="assistant", description="The role of the message sender")
    content: list[AnthropicContentBlock] = Field(..., description="List of content blocks")
    model: str = Field(..., description="The model used for generation")
    stop_reason: str | None = Field(default=None, description="Reason for stopping")
    usage: AnthropicUsage | None = Field(default=None, description="Token usage information")


class AnthropicStreamEvent(BaseModel):
    """
    AnthropicStreamEvent
    Represents a streaming event from Anthropic API.
    """
    type: str = Field(..., description="The type of stream event")
    message: dict[str, Any] | None = Field(default=None, description="Message-related data")
    index: int | None = Field(default=None, description="Content block index")
    content_block: dict[str, Any] | None = Field(default=None, description="Content block data")
    delta: dict[str, Any] | None = Field(default=None, description="Delta data")