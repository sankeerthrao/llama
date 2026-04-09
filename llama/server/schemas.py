# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Pydantic v2 schemas for the OpenAI-compatible LLaMA Inference API.

Defines request/response models for chat completions, text completions,
model listing, health checks, and error responses.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / primitive types
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., description="The role of the message author."
    )
    content: str = Field(
        ..., description="The content of the message."
    )


class UsageInfo(BaseModel):
    """Token usage statistics for a request."""

    prompt_tokens: int = Field(0, description="Number of tokens in the prompt.")
    completion_tokens: int = Field(0, description="Number of tokens in the completion.")
    total_tokens: int = Field(0, description="Total tokens used (prompt + completion).")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions (OpenAI-compatible)."""

    model: str = Field("llama", description="Model identifier.")
    messages: List[Message] = Field(
        ..., description="List of messages comprising the conversation."
    )
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="Nucleus sampling threshold.")
    max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum number of tokens to generate."
    )
    stream: bool = Field(False, description="Whether to stream partial results via SSE.")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Stop sequence(s)."
    )
    frequency_penalty: float = Field(
        0.0, ge=-2.0, le=2.0, description="Frequency penalty."
    )
    presence_penalty: float = Field(
        0.0, ge=-2.0, le=2.0, description="Presence penalty."
    )
    top_k: int = Field(0, ge=0, description="Top-k sampling. 0 = disabled.")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Min-p sampling. 0 = disabled.")
    repetition_penalty: float = Field(
        1.0, ge=0.0, description="Repetition penalty multiplier."
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice in a chat response."""

    index: int = Field(0, description="Choice index.")
    message: Message = Field(..., description="Generated message.")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Reason the generation stopped."
    )


class ChatCompletionResponse(BaseModel):
    """Response body for POST /v1/chat/completions."""

    id: str = Field(..., description="Unique completion ID.")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation.",
    )
    model: str = Field("llama", description="Model used.")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of completion choices."
    )
    usage: UsageInfo = Field(
        default_factory=UsageInfo, description="Token usage info."
    )


class ChatCompletionStreamChoice(BaseModel):
    """A single streamed delta choice."""

    index: int = 0
    delta: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """A single SSE chunk for streamed chat completions."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llama"
    choices: List[ChatCompletionStreamChoice] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Text completions
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions (OpenAI-compatible)."""

    model: str = Field("llama", description="Model identifier.")
    prompt: Union[str, List[str]] = Field(
        ..., description="Prompt(s) to generate completions for."
    )
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="Nucleus sampling threshold.")
    max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum number of tokens to generate."
    )
    stream: bool = Field(False, description="Whether to stream partial results via SSE.")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Stop sequence(s)."
    )
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    top_k: int = Field(0, ge=0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.0)


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = 0
    text: str = Field("", description="Generated text.")
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    """Response body for POST /v1/completions."""

    id: str = Field(..., description="Unique completion ID.")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llama"
    choices: List[CompletionChoice] = Field(
        ..., description="List of completion choices."
    )
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata for a single model."""

    id: str = Field(..., description="Model identifier.")
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = Field("meta", description="Model owner.")


class ModelListResponse(BaseModel):
    """Response body for GET /v1/models."""

    object: Literal["list"] = "list"
    data: List[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        "healthy", description="Overall server status."
    )
    model_loaded: bool = Field(False, description="Whether a model is loaded.")
    uptime_seconds: float = Field(0.0, description="Server uptime in seconds.")
    version: str = Field("2.0.0", description="API version string.")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """Inner detail of an error response."""

    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-style error response envelope."""

    error: ErrorDetail
