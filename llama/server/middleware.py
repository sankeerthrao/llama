# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
FastAPI middleware for the LLaMA Inference API.

Provides:
- Token-bucket rate limiting (in-memory, no external dependencies).
- Bearer-token API key authentication.
- Request logging with latency tracking.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("llama.server")


# ---------------------------------------------------------------------------
# Rate limiter – simple in-memory token bucket per client IP
# ---------------------------------------------------------------------------

class RateLimiter:
    """In-memory token-bucket rate limiter.

    Each client IP gets its own bucket. Tokens refill at a steady rate
    of ``max_rpm / 60`` tokens per second. A request consumes one token.

    Args:
        max_rpm: Maximum requests per minute. ``0`` disables limiting.
    """

    def __init__(self, max_rpm: int = 60) -> None:
        self.max_rpm = max_rpm
        self._tokens_per_second = max_rpm / 60.0
        self._buckets: Dict[str, _Bucket] = {}

    # -- public API ---------------------------------------------------------

    def allow(self, client_ip: str) -> bool:
        """Return ``True`` if the request is allowed, consuming one token."""
        if self.max_rpm <= 0:
            return True
        now = time.monotonic()
        bucket = self._buckets.get(client_ip)
        if bucket is None:
            bucket = _Bucket(
                tokens=float(self.max_rpm),
                last_refill=now,
            )
            self._buckets[client_ip] = bucket
        # refill
        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            float(self.max_rpm),
            bucket.tokens + elapsed * self._tokens_per_second,
        )
        bucket.last_refill = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


class _Bucket:
    """Internal mutable state for a single token bucket."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, last_refill: float) -> None:
        self.tokens = tokens
        self.last_refill = last_refill


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that applies :class:`RateLimiter` to every request."""

    def __init__(self, app: Callable, rate_limiter: RateLimiter) -> None:  # type: ignore[override]
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        if not self.rate_limiter.allow(client_ip):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please try again later.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


class APIKeyAuth:
    """FastAPI dependency that validates a Bearer token from the Authorization header.

    If no ``api_key`` is configured (``None`` or empty string), authentication
    is skipped and all requests are allowed through.

    Args:
        api_key: The expected API key string. ``None`` disables auth.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or None  # normalise empty string → None

    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> Optional[str]:
        """Validate the request and return the authenticated key (or ``None``)."""
        if self.api_key is None:
            return None

        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Missing Authorization header.",
                        "type": "authentication_error",
                        "code": "missing_api_key",
                    }
                },
            )

        if credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "message": "Invalid API key.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        return credentials.credentials


# ---------------------------------------------------------------------------
# Request logger middleware
# ---------------------------------------------------------------------------

class RequestLogger(BaseHTTPMiddleware):
    """Logs method, path, status code, and latency for every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response
