"""Configuration shared by the AutoGluon SageMaker serving processes."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw_value = environment.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _log_level(environment: Mapping[str, str]) -> int:
    raw_value = environment.get("SAGEMAKER_CONTAINER_LOG_LEVEL", "INFO").strip()
    if not raw_value:
        raise ValueError("SAGEMAKER_CONTAINER_LOG_LEVEL must not be empty")

    try:
        numeric_level = int(raw_value)
    except ValueError:
        numeric_level = logging.getLevelNamesMapping().get(raw_value.upper())
        if numeric_level is None:
            raise ValueError(
                f"SAGEMAKER_CONTAINER_LOG_LEVEL is not a valid log level: {raw_value!r}"
            )
    return numeric_level


@dataclass(frozen=True)
class Settings:
    """Validated serving configuration.

    Only established SageMaker inference variables are consumed here. Framework-
    or model-server-specific aliases are intentionally not supported.
    """

    base_dir: Path
    program: str
    bind_port: int
    workers: int
    timeout: int
    default_accept: str
    log_level: int
    codeartifact_repository_arn: str | None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        environment = os.environ if environment is None else environment
        base_dir = Path(environment.get("SAGEMAKER_BASE_DIR", "/opt/ml"))
        program = environment.get("SAGEMAKER_PROGRAM", "inference.py").strip()
        default_accept = environment.get(
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT", "application/json"
        ).strip()

        if not program:
            raise ValueError("SAGEMAKER_PROGRAM must not be empty")
        if not default_accept:
            raise ValueError("SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT must not be empty")

        bind_port = _positive_int(environment, "SAGEMAKER_BIND_TO_PORT", 8080)
        if bind_port > 65535:
            raise ValueError(f"SAGEMAKER_BIND_TO_PORT must not exceed 65535, got {bind_port}")

        repository_arn = environment.get("CA_REPOSITORY_ARN")
        return cls(
            base_dir=base_dir,
            program=program,
            bind_port=bind_port,
            workers=_positive_int(environment, "SAGEMAKER_MODEL_SERVER_WORKERS", 1),
            timeout=_positive_int(environment, "SAGEMAKER_MODEL_SERVER_TIMEOUT", 60),
            default_accept=default_accept,
            log_level=_log_level(environment),
            codeartifact_repository_arn=repository_arn.strip() if repository_arn else None,
        )

    @property
    def model_dir(self) -> Path:
        return self.base_dir / "model"

    @property
    def code_dir(self) -> Path:
        return self.model_dir / "code"

    @property
    def requirements_path(self) -> Path:
        return self.code_dir / "requirements.txt"

    @property
    def handler_path(self) -> Path:
        program = Path(self.program)
        return program if program.is_absolute() else self.code_dir / program

    @property
    def gunicorn_log_level(self) -> str:
        if self.log_level <= logging.DEBUG:
            return "debug"
        if self.log_level <= logging.INFO:
            return "info"
        if self.log_level <= logging.WARNING:
            return "warning"
        if self.log_level <= logging.ERROR:
            return "error"
        return "critical"
