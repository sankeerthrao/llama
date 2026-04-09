# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for the OpenAI-compatible API server."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from llama.server.app import create_app
from llama.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
    Message,
    ModelInfo,
    ModelListResponse,
)
from llama.server.middleware import RateLimiter, APIKeyAuth


class TestSchemas:
    def test_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(
            messages=[Message(role="user", content="Hi")]
        )
        assert req.model == "llama"
        assert req.temperature == 0.6
        assert req.top_p == 0.9
        assert req.stream is False

    def test_chat_completion_request_custom(self):
        req = ChatCompletionRequest(
            model="llama-2-70b",
            messages=[Message(role="user", content="Hi")],
            temperature=0.3,
            top_p=0.8,
            max_tokens=100,
            stream=True,
        )
        assert req.model == "llama-2-70b"
        assert req.temperature == 0.3
        assert req.max_tokens == 100
        assert req.stream is True

    def test_health_response(self):
        hr = HealthResponse(
            status="healthy",
            model_loaded=True,
            uptime_seconds=123.4,
            version="2.0.0",
        )
        assert hr.status == "healthy"
        assert hr.model_loaded is True

    def test_model_info(self):
        mi = ModelInfo(id="llama-2-7b", object="model", owned_by="meta")
        assert mi.id == "llama-2-7b"


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_rpm=60)
        for _ in range(5):
            assert rl.allow("test-client") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_rpm=3)
        for _ in range(3):
            assert rl.allow("test-client") is True
        assert rl.allow("test-client") is False

    def test_different_clients_independent(self):
        rl = RateLimiter(max_rpm=2)
        assert rl.allow("client-a") is True
        assert rl.allow("client-a") is True
        assert rl.allow("client-a") is False  # client-a exhausted
        assert rl.allow("client-b") is True  # client-b still ok

    def test_zero_rpm_unlimited(self):
        rl = RateLimiter(max_rpm=0)
        for _ in range(100):
            assert rl.allow("test") is True


class TestAPIKeyAuth:
    def test_no_key_configured_creates_ok(self):
        auth = APIKeyAuth(api_key=None)
        assert auth is not None
        # No key configured -- auth is a no-op dependency
        assert auth.api_key is None

    def test_with_key_configured(self):
        auth = APIKeyAuth(api_key="secret-key")
        assert auth.api_key == "secret-key"

    def test_is_callable(self):
        """APIKeyAuth is a FastAPI dependency (callable)."""
        auth = APIKeyAuth(api_key="secret-key")
        assert callable(auth)


class TestServerEndpoints:
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "no_model")
        assert "version" in data
        assert "uptime_seconds" in data

    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    def test_text_completion(self, client):
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": "Hello world",
                "max_tokens": 10,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

    def test_chat_completion(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]

    def test_chat_completion_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Should contain SSE data lines
        body = resp.text
        assert "data:" in body
        assert "[DONE]" in body

    def test_invalid_request(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"invalid": "payload"},
        )
        assert resp.status_code == 422  # Validation error


class TestServerWithAPIKey:
    @pytest.fixture
    def client(self):
        from llama.config import InferenceConfig
        cfg = InferenceConfig()
        cfg.server.api_key = "test-secret"
        app = create_app(config=cfg)
        return TestClient(app)

    def test_health_no_auth_needed(self, client):
        """Health endpoint should work without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_completion_without_key_rejected(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 5},
        )
        # Should be 401 or 403
        assert resp.status_code in (401, 403)

    def test_completion_with_correct_key(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 5},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert resp.status_code == 200
