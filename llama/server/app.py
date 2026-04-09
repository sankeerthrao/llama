# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
OpenAI-compatible REST API server for LLaMA inference.

Provides ``/v1/chat/completions``, ``/v1/completions``, ``/v1/models``, and
``/health`` endpoints.  Streaming is supported via Server-Sent Events (SSE).

Start the server::

    python -m llama.server.app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Union

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from llama.server.middleware import (
    APIKeyAuth,
    RateLimiter,
    RateLimiterMiddleware,
    RequestLogger,
)
from llama.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    Message,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)

# Try to import the config system; fall back to defaults if unavailable.
# The top-level ``llama`` package's __init__.py imports generation.py which
# requires torch / fairscale / CUDA.  We use importlib to load config.py
# directly so the server can start on CPU-only machines without those deps.
try:
    import importlib.util as _ilu

    _config_spec = _ilu.spec_from_file_location(
        "llama.config",
        str(Path(__file__).resolve().parent.parent / "config.py"),
    )
    if _config_spec is not None and _config_spec.loader is not None:
        _config_mod = _ilu.module_from_spec(_config_spec)
        _config_spec.loader.exec_module(_config_mod)  # type: ignore[union-attr]
        InferenceConfig = _config_mod.InferenceConfig  # type: ignore[attr-defined]
    else:
        InferenceConfig = None  # type: ignore[assignment,misc]
except Exception:  # pragma: no cover – best-effort
    InferenceConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger("llama.server")

# ---------------------------------------------------------------------------
# Server start time (set once at module load; refined in lifespan)
# ---------------------------------------------------------------------------
_SERVER_START_TIME: float = time.time()

API_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Mock generator – used when no real model/GPU is available
# ---------------------------------------------------------------------------

class MockGenerator:
    """Deterministic mock generator for testing the API surface without a GPU.

    Returns plausible dummy text and approximate token counts so that
    every endpoint can be exercised end-to-end.
    """

    MODEL_NAME = "llama-mock"

    _LOREM = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
    )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Rough token count ≈ words × 1.3."""
        return max(1, int(len(text.split()) * 1.3))

    # -- public API ---------------------------------------------------------

    def generate_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> tuple[str, int, int]:
        """Return ``(generated_text, prompt_tokens, completion_tokens)``."""
        max_tokens = max_tokens or 64
        # Produce a deterministic but length-controlled response.
        words = self._LOREM.split()
        out_words = (words * ((max_tokens // len(words)) + 1))[:max_tokens]
        generated = " ".join(out_words)
        return generated, self._approx_tokens(prompt), self._approx_tokens(generated)

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> tuple[str, int, int]:
        """Return ``(assistant_reply, prompt_tokens, completion_tokens)``."""
        prompt_text = " ".join(m.content for m in messages)
        max_tokens = max_tokens or 64
        reply = (
            f"This is a mock response to: "
            f"{messages[-1].content[:120] if messages else '(empty)'}. "
            f"{self._LOREM}"
        )
        # Trim to approximate max_tokens worth of words.
        reply_words = reply.split()[:max_tokens]
        reply = " ".join(reply_words)
        return reply, self._approx_tokens(prompt_text), self._approx_tokens(reply)

    def generate_chat_stream(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> tuple[list[str], int, int]:
        """Return ``(word_chunks, prompt_tokens, total_completion_tokens)``.

        Each element of *word_chunks* is a small piece of text to be sent as
        one SSE event.
        """
        reply, prompt_tokens, completion_tokens = self.generate_chat(
            messages, max_tokens=max_tokens, **kwargs
        )
        # Split into small chunks (1-3 words each) for realistic streaming.
        words = reply.split()
        chunks: list[str] = []
        i = 0
        while i < len(words):
            chunk_size = min(2, len(words) - i)
            chunks.append(" ".join(words[i : i + chunk_size]))
            i += chunk_size
        return chunks, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _make_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def create_app(
    config: Optional[object] = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        config: An optional :class:`~llama.config.InferenceConfig` instance.
            When ``None`` sensible defaults are used and a
            :class:`MockGenerator` is wired in.
    """
    # Resolve configuration --------------------------------------------------
    if InferenceConfig is not None and config is not None:
        cfg = config
    elif InferenceConfig is not None:
        cfg = InferenceConfig()
        cfg.apply_env_overrides()
    else:
        cfg = None

    cors_origins: list[str] = ["*"]
    api_key: Optional[str] = None
    rate_limit_rpm: int = 60

    if cfg is not None:
        cors_origins = list(cfg.server.cors_origins)  # type: ignore[union-attr]
        api_key = cfg.server.api_key  # type: ignore[union-attr]
        rate_limit_rpm = cfg.server.rate_limit_rpm  # type: ignore[union-attr]

    # Build app ---------------------------------------------------------------
    app = FastAPI(
        title="LLaMA Inference API",
        version=API_VERSION,
        description="OpenAI-compatible REST API for LLaMA model inference.",
    )

    # -- state ----------------------------------------------------------------
    app.state.generator = MockGenerator()
    app.state.model_name = MockGenerator.MODEL_NAME
    app.state.start_time = _SERVER_START_TIME

    # -- middleware (order matters: outermost first) ---------------------------
    app.add_middleware(RequestLogger)
    app.add_middleware(
        RateLimiterMiddleware,
        rate_limiter=RateLimiter(max_rpm=rate_limit_rpm),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- auth dependency ------------------------------------------------------
    auth = APIKeyAuth(api_key=api_key)

    # -----------------------------------------------------------------------
    # Error handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(exc),
                    type="invalid_request_error",
                    param=str(first.get("loc", "")),
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error.",
                    type="server_error",
                    code="internal_error",
                )
            ).model_dump(),
        )

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Server health check."""
        return HealthResponse(
            status="healthy",
            model_loaded=app.state.generator is not None,
            uptime_seconds=round(time.time() - app.state.start_time, 2),
            version=API_VERSION,
        )

    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        tags=["models"],
        dependencies=[Depends(auth)],
    )
    async def list_models() -> ModelListResponse:
        """List available models."""
        model = ModelInfo(
            id=app.state.model_name,
            created=int(app.state.start_time),
            owned_by="meta",
        )
        return ModelListResponse(data=[model])

    # -- Text completions ----------------------------------------------------

    @app.post(
        "/v1/completions",
        response_model=CompletionResponse,
        tags=["completions"],
        dependencies=[Depends(auth)],
    )
    async def create_completion(body: CompletionRequest) -> CompletionResponse:
        """Create a text completion."""
        generator: MockGenerator = app.state.generator

        prompts: list[str] = (
            body.prompt if isinstance(body.prompt, list) else [body.prompt]
        )

        choices: list[CompletionChoice] = []
        total_prompt = 0
        total_completion = 0

        for idx, prompt in enumerate(prompts):
            text, p_tok, c_tok = generator.generate_completion(
                prompt=prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                min_p=body.min_p,
                frequency_penalty=body.frequency_penalty,
                presence_penalty=body.presence_penalty,
                repetition_penalty=body.repetition_penalty,
            )
            choices.append(
                CompletionChoice(index=idx, text=text, finish_reason="stop")
            )
            total_prompt += p_tok
            total_completion += c_tok

        return CompletionResponse(
            id=_make_completion_id(),
            model=app.state.model_name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_prompt + total_completion,
            ),
        )

    # -- Chat completions ----------------------------------------------------

    async def _stream_chat(
        body: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE ``data:`` lines for a streamed chat completion."""
        generator: MockGenerator = app.state.generator
        completion_id = _make_chat_completion_id()

        chunks, prompt_tokens, _ = generator.generate_chat_stream(
            messages=body.messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            min_p=body.min_p,
            frequency_penalty=body.frequency_penalty,
            presence_penalty=body.presence_penalty,
            repetition_penalty=body.repetition_penalty,
        )

        # First chunk: send role
        first_event = ChatCompletionStreamResponse(
            id=completion_id,
            model=app.state.model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                )
            ],
        )
        yield f"data: {first_event.model_dump_json()}\n\n"

        # Content chunks
        for piece in chunks:
            event = ChatCompletionStreamResponse(
                id=completion_id,
                model=app.state.model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"content": piece + " "},
                    )
                ],
            )
            yield f"data: {event.model_dump_json()}\n\n"
            # Small sleep to simulate real generation latency
            await asyncio.sleep(0.01)

        # Final chunk: finish_reason
        final_event = ChatCompletionStreamResponse(
            id=completion_id,
            model=app.state.model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_event.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    @app.post(
        "/v1/chat/completions",
        tags=["chat"],
        dependencies=[Depends(auth)],
        response_model=None,
    )
    async def create_chat_completion(
        body: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Create a chat completion, optionally streamed."""
        if body.stream:
            return StreamingResponse(
                _stream_chat(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        generator: MockGenerator = app.state.generator
        reply, prompt_tokens, completion_tokens = generator.generate_chat(
            messages=body.messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            min_p=body.min_p,
            frequency_penalty=body.frequency_penalty,
            presence_penalty=body.presence_penalty,
            repetition_penalty=body.repetition_penalty,
        )

        return ChatCompletionResponse(
            id=_make_chat_completion_id(),
            model=app.state.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and start the uvicorn server."""
    parser = argparse.ArgumentParser(
        description="LLaMA Inference API Server (OpenAI-compatible)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON or YAML config file.",
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config if provided
    config = None
    if args.config:
        if InferenceConfig is None:
            logger.warning(
                "InferenceConfig not available; ignoring --config %s", args.config
            )
        else:
            config_path = Path(args.config)
            if not config_path.exists():
                parser.error(f"Config file not found: {args.config}")
            config = InferenceConfig.from_file(args.config)
            config.apply_env_overrides()
            logger.info("Loaded config from %s", args.config)

    # If config overrides host/port and user did not pass explicit flags,
    # honour the config values.
    if config is not None:
        if args.host == "0.0.0.0" and config.server.host != "0.0.0.0":
            args.host = config.server.host
        if args.port == 8000 and config.server.port != 8000:
            args.port = config.server.port

    app = create_app(config=config)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required to run the server. Install with: pip install uvicorn"
        )

    logger.info("Starting LLaMA Inference API on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
