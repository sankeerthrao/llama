# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Configuration system for LLaMA inference engine.
Supports YAML/JSON config files with validation and environment variable overrides.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ModelConfig:
    """Configuration for model loading and architecture."""
    ckpt_dir: str = ""
    tokenizer_path: str = ""
    max_seq_len: int = 2048
    max_batch_size: int = 32
    model_parallel_size: Optional[int] = None
    seed: int = 1
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    enabled: bool = False
    method: Literal["none", "int8", "int4"] = "none"
    group_size: int = 128


@dataclass
class GenerationConfig:
    """Configuration for text generation defaults."""
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 0  # 0 = disabled
    min_p: float = 0.0  # 0 = disabled
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_gen_len: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    rate_limit_rpm: int = 60  # requests per minute, 0 = unlimited
    max_concurrent_requests: int = 64
    request_timeout: float = 300.0


@dataclass
class LoggingConfig:
    """Configuration for structured logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["text", "json"] = "text"
    file: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 9090


@dataclass
class SecurityConfig:
    """Configuration for security features."""
    enable_input_sanitization: bool = True
    max_input_length: int = 4096
    blocked_patterns: List[str] = field(default_factory=list)
    enable_content_filter: bool = False


@dataclass
class InferenceConfig:
    """Root configuration for the entire inference engine."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        config = cls()
        if "model" in data:
            config.model = ModelConfig(**{k: v for k, v in data["model"].items() if hasattr(ModelConfig, k)})
        if "quantization" in data:
            config.quantization = QuantizationConfig(
                **{k: v for k, v in data["quantization"].items() if hasattr(QuantizationConfig, k)}
            )
        if "generation" in data:
            config.generation = GenerationConfig(
                **{k: v for k, v in data["generation"].items() if hasattr(GenerationConfig, k)}
            )
        if "server" in data:
            config.server = ServerConfig(**{k: v for k, v in data["server"].items() if hasattr(ServerConfig, k)})
        if "logging" in data:
            config.logging = LoggingConfig(**{k: v for k, v in data["logging"].items() if hasattr(LoggingConfig, k)})
        if "security" in data:
            config.security = SecurityConfig(
                **{k: v for k, v in data["security"].items() if hasattr(SecurityConfig, k)}
            )
        return config

    @classmethod
    def from_json_file(cls, path: str) -> "InferenceConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "InferenceConfig":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "InferenceConfig":
        """Load config from a file, auto-detecting format from extension."""
        path_obj = Path(path)
        if path_obj.suffix in (".yaml", ".yml"):
            return cls.from_yaml_file(path)
        elif path_obj.suffix == ".json":
            return cls.from_json_file(path)
        else:
            raise ValueError(f"Unsupported config file format: {path_obj.suffix}")

    def apply_env_overrides(self) -> "InferenceConfig":
        """Apply environment variable overrides. Format: LLAMA_SECTION_KEY=value"""
        prefix = "LLAMA_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, field_name = parts
            section_obj = getattr(self, section, None)
            if section_obj is None:
                continue
            if not hasattr(section_obj, field_name):
                continue
            current = getattr(section_obj, field_name)
            if isinstance(current, bool):
                setattr(section_obj, field_name, value.lower() in ("true", "1", "yes"))
            elif isinstance(current, int):
                setattr(section_obj, field_name, int(value))
            elif isinstance(current, float):
                setattr(section_obj, field_name, float(value))
            elif isinstance(current, str):
                setattr(section_obj, field_name, value)
        return self

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.model.max_seq_len < 1:
            errors.append("model.max_seq_len must be >= 1")
        if self.model.max_batch_size < 1:
            errors.append("model.max_batch_size must be >= 1")
        if self.generation.temperature < 0:
            errors.append("generation.temperature must be >= 0")
        if not (0 < self.generation.top_p <= 1.0):
            errors.append("generation.top_p must be in (0, 1]")
        if self.generation.top_k < 0:
            errors.append("generation.top_k must be >= 0")
        if self.generation.repetition_penalty < 1.0:
            errors.append("generation.repetition_penalty must be >= 1.0")
        if self.server.port < 1 or self.server.port > 65535:
            errors.append("server.port must be between 1 and 65535")
        if self.security.max_input_length < 1:
            errors.append("security.max_input_length must be >= 1")
        return errors
