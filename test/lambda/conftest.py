"""Lambda DLC test configuration.

Tests run directly inside the container — the workflow handles all Docker
orchestration (build image, start container, volume-mount /test, attach GPUs).
No Docker-related fixtures are needed.

Directory layout:
  unit/        — CPU-only tests (imports, env vars, binaries, entrypoint)
  single_gpu/  — GPU tests (CUDA ops, inference, FFmpeg GPU transcode)
"""
