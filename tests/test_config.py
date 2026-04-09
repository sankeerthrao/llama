# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for the configuration system."""

import json
import os
import tempfile

import pytest

from llama.config import (
    GenerationConfig,
    InferenceConfig,
    LoggingConfig,
    ModelConfig,
    QuantizationConfig,
    SecurityConfig,
    ServerConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.max_seq_len == 2048
        assert cfg.max_batch_size == 32
        assert cfg.dtype == "float16"
        assert cfg.device == "auto"

    def test_custom_values(self):
        cfg = ModelConfig(max_seq_len=4096, dtype="bfloat16", device="cpu")
        assert cfg.max_seq_len == 4096
        assert cfg.dtype == "bfloat16"
        assert cfg.device == "cpu"


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.6
        assert cfg.top_p == 0.9
        assert cfg.top_k == 0
        assert cfg.min_p == 0.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.frequency_penalty == 0.0
        assert cfg.presence_penalty == 0.0
        assert cfg.stop_sequences == []

    def test_custom(self):
        cfg = GenerationConfig(temperature=0.0, top_k=50, min_p=0.05)
        assert cfg.temperature == 0.0
        assert cfg.top_k == 50
        assert cfg.min_p == 0.05


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.workers == 1
        assert cfg.cors_origins == ["*"]
        assert cfg.api_key is None
        assert cfg.rate_limit_rpm == 60

    def test_custom(self):
        cfg = ServerConfig(port=9000, api_key="test-key", rate_limit_rpm=120)
        assert cfg.port == 9000
        assert cfg.api_key == "test-key"
        assert cfg.rate_limit_rpm == 120


class TestInferenceConfig:
    def test_default_construction(self):
        cfg = InferenceConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.security, SecurityConfig)
        assert isinstance(cfg.quantization, QuantizationConfig)

    def test_to_dict(self):
        cfg = InferenceConfig()
        d = cfg.to_dict()
        assert "model" in d
        assert "generation" in d
        assert "server" in d
        assert d["model"]["max_seq_len"] == 2048

    def test_to_json(self):
        cfg = InferenceConfig()
        j = cfg.to_json()
        parsed = json.loads(j)
        assert parsed["model"]["max_seq_len"] == 2048

    def test_from_dict(self):
        data = {
            "model": {"max_seq_len": 4096, "dtype": "bfloat16"},
            "generation": {"temperature": 0.3, "top_k": 40},
            "server": {"port": 9000},
        }
        cfg = InferenceConfig.from_dict(data)
        assert cfg.model.max_seq_len == 4096
        assert cfg.model.dtype == "bfloat16"
        assert cfg.generation.temperature == 0.3
        assert cfg.generation.top_k == 40
        assert cfg.server.port == 9000
        # Defaults preserved for unset fields
        assert cfg.model.max_batch_size == 32
        assert cfg.generation.top_p == 0.9

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "model": {"max_seq_len": 1024, "unknown_field": "ignored"},
        }
        cfg = InferenceConfig.from_dict(data)
        assert cfg.model.max_seq_len == 1024
        assert not hasattr(cfg.model, "unknown_field")

    def test_from_json_file(self):
        data = {
            "model": {"max_seq_len": 512},
            "server": {"port": 3000},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            cfg = InferenceConfig.from_json_file(path)
            assert cfg.model.max_seq_len == 512
            assert cfg.server.port == 3000
        finally:
            os.unlink(path)

    def test_from_file_json(self):
        data = {"model": {"max_seq_len": 256}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            cfg = InferenceConfig.from_file(path)
            assert cfg.model.max_seq_len == 256
        finally:
            os.unlink(path)

    def test_from_file_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                InferenceConfig.from_file(path)
        finally:
            os.unlink(path)

    def test_env_overrides(self):
        cfg = InferenceConfig()
        os.environ["LLAMA_MODEL_MAX_SEQ_LEN"] = "1024"  # Note: underscore in field name
        # The split("_", 1) gives ["model", "max_seq_len"] -- but our impl splits LLAMA_ prefix,
        # then the remainder by "_" with maxsplit=1. "MODEL_MAX_SEQ_LEN" -> section="model", field="max_seq_len"
        # Wait, it splits on underscore with maxsplit=1, so: "model" and "max_seq_len" -- correct!
        os.environ["LLAMA_SERVER_PORT"] = "9090"
        try:
            cfg.apply_env_overrides()
            # The env var is LLAMA_MODEL_MAX_SEQ_LEN, split("_", 1) gives ["model", "max_seq_len"]
            # But wait: after removing "LLAMA_" prefix we get "MODEL_MAX_SEQ_LEN"
            # .lower() -> "model_max_seq_len"
            # .split("_", 1) -> ["model", "max_seq_len"] -- correct!
            assert cfg.model.max_seq_len == 1024  # Should work because field is "max_seq_len"
            # Actually wait -- the field name has underscores. split("_", 1) on "model_max_seq_len"
            # gives section="model", field="max_seq_len" -- and ModelConfig has max_seq_len. This works!
        finally:
            del os.environ["LLAMA_MODEL_MAX_SEQ_LEN"]
            del os.environ["LLAMA_SERVER_PORT"]

    def test_env_override_bool(self):
        cfg = InferenceConfig()
        os.environ["LLAMA_SECURITY_ENABLE_INPUT_SANITIZATION"] = "false"
        try:
            cfg.apply_env_overrides()
            # "security_enable_input_sanitization" split("_", 1) -> ["security", "enable_input_sanitization"]
            assert cfg.security.enable_input_sanitization is False
        finally:
            del os.environ["LLAMA_SECURITY_ENABLE_INPUT_SANITIZATION"]

    def test_validate_valid_config(self):
        cfg = InferenceConfig()
        errors = cfg.validate()
        assert errors == []

    def test_validate_invalid_config(self):
        cfg = InferenceConfig()
        cfg.model.max_seq_len = 0
        cfg.generation.temperature = -1
        cfg.generation.top_p = 0
        cfg.server.port = 0
        errors = cfg.validate()
        assert len(errors) >= 4
        assert any("max_seq_len" in e for e in errors)
        assert any("temperature" in e for e in errors)
        assert any("top_p" in e for e in errors)
        assert any("port" in e for e in errors)

    def test_validate_repetition_penalty(self):
        cfg = InferenceConfig()
        cfg.generation.repetition_penalty = 0.5
        errors = cfg.validate()
        assert any("repetition_penalty" in e for e in errors)

    def test_validate_security(self):
        cfg = InferenceConfig()
        cfg.security.max_input_length = 0
        errors = cfg.validate()
        assert any("max_input_length" in e for e in errors)

    def test_roundtrip_dict(self):
        """Test that to_dict -> from_dict produces equivalent config."""
        original = InferenceConfig()
        original.model.max_seq_len = 1024
        original.generation.temperature = 0.3
        d = original.to_dict()
        restored = InferenceConfig.from_dict(d)
        assert restored.model.max_seq_len == 1024
        assert restored.generation.temperature == 0.3

    def test_roundtrip_json(self):
        """Test that to_json -> from_dict(json.loads) produces equivalent config."""
        original = InferenceConfig()
        original.server.port = 5555
        j = original.to_json()
        restored = InferenceConfig.from_dict(json.loads(j))
        assert restored.server.port == 5555
