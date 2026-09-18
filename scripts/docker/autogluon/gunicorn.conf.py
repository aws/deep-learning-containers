"""Gunicorn configuration for the AutoGluon SageMaker server."""

import os

bind = f"0.0.0.0:{os.getenv('SAGEMAKER_BIND_TO_PORT', '8080')}"
workers = int(os.getenv("SAGEMAKER_MODEL_SERVER_WORKERS", "1"))
timeout = int(os.getenv("SAGEMAKER_MODEL_SERVER_TIMEOUT", "60"))
graceful_timeout = timeout

worker_class = "sync"

accesslog = "-"
errorlog = "-"
capture_output = True
raw_log_level = os.getenv("SAGEMAKER_CONTAINER_LOG_LEVEL", "info").strip().lower() or "info"
loglevel = {
    "10": "debug",
    "20": "info",
    "30": "warning",
    "40": "error",
    "50": "critical",
}.get(raw_log_level, raw_log_level)
