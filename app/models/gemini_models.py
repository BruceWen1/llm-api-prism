# -*- coding: utf-8 -*-
"""
gemini_models.py
Gemini API Models
=====================

Pydantic models for Google Gemini generateContent API requests and responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GeminiPart(BaseModel):
    """
    GeminiPart
    Represents a part of content in Gemini API.
    """
    text: str | None = Field(default=None, description="The text content")
    functionCall: dict[str, Any] | None = Field(default=None, description="Function call")
    functionResponse: dict[str, Any] | None = Field(default=None, description="Function response")


class GeminiContent(BaseModel):
    """
    GeminiContent
    Represents content with role and parts in Gemini API.
    Role is optional because systemInstruction uses GeminiContent without a role.
    """
    role: str | None = Field(default=None, description="The role of the content sender")
    parts: list[GeminiPart] = Field(..., description="List of content parts")


class GeminiFunctionDeclaration(BaseModel):
    """
    GeminiFunctionDeclaration
    Represents a function declaration in Gemini API.
    """
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GeminiTool(BaseModel):
    """
    GeminiTool
    Represents a tool in Gemini API.
    """
    functionDeclarations: list[GeminiFunctionDeclaration] | None = None


class GeminiToolConfig(BaseModel):
    """
    GeminiToolConfig
    Represents tool configuration in Gemini API.
    """
    functionCallingConfig: dict[str, Any] | None = None


class GeminiSafetySetting(BaseModel):
    """
    GeminiSafetySetting
    Represents a safety setting in Gemini API.
    """
    category: str
    threshold: str


class GeminiGenerationConfig(BaseModel):
    """
    GeminiGenerationConfig
    Configuration for generation in Gemini API.
    """
    temperature: float | None = None
    topP: float | None = None
    topK: int | None = None
    maxOutputTokens: int | None = None
    stopSequences: list[str] | None = None
    candidateCount: int | None = None
    presencePenalty: float | None = None
    frequencyPenalty: float | None = None
    responseMimeType: str | None = None
    responseSchema: dict[str, Any] | None = None


class GeminiRequest(BaseModel):
    """
    GeminiRequest
    Request model for Gemini generateContent API.
    """
    contents: list[GeminiContent]
    systemInstruction: GeminiContent | None = None
    generationConfig: GeminiGenerationConfig | None = None
    model: str | None = None
    tools: list[GeminiTool] | None = None
    toolConfig: GeminiToolConfig | None = None
    safetySettings: list[GeminiSafetySetting] | None = None


class GeminiCandidate(BaseModel):
    """
    GeminiCandidate
    Represents a candidate response in Gemini API.
    """
    content: GeminiContent
    finishReason: str | None = None
    safetyRatings: list[dict[str, Any]] | None = None


class GeminiUsageMetadata(BaseModel):
    """
    GeminiUsageMetadata
    Token usage metadata for Gemini API response.
    """
    promptTokenCount: int | None = Field(default=None, description="Number of prompt tokens")
    candidatesTokenCount: int | None = Field(default=None, description="Number of candidate tokens")
    totalTokenCount: int | None = Field(default=None, description="Total number of tokens")


class GeminiResponse(BaseModel):
    """
    GeminiResponse
    Response model for Gemini generateContent API.
    """
    candidates: list[GeminiCandidate] | None = Field(default=None, description="List of candidates")
    usageMetadata: GeminiUsageMetadata | None = Field(default=None, description="Token usage information")
    modelVersion: str | None = Field(default=None, description="Model version used")