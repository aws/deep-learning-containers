"""Gunicorn configuration for the AutoGluon SageMaker server."""

from pathlib import Path
from runpy import run_path

Settings = run_path(str(Path(__file__).with_name("settings.py")))["Settings"]
settings = Settings.from_environment()

bind = f"0.0.0.0:{settings.bind_port}"
workers = settings.workers
timeout = settings.timeout
graceful_timeout = settings.timeout
worker_class = "sync"
threads = 1
preload_app = False

accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = settings.gunicorn_log_level
