"""Evict unused cached models to free disk space.

Evicts smallest models first (cheapest to re-download). Uses flock to
check if a model is in use — if an exclusive lock succeeds, no job
holds a shared lock on it, so it's safe to delete.
"""

import argparse
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def get_free_gb(path: str) -> int:
    """Return free disk space in GB for the filesystem containing path."""
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) // (1024**3)


def get_cached_models(cache_dir: str) -> list[tuple[int, str]]:
    """Return list of (size_bytes, model_name) sorted smallest first."""
    models = []
    for entry in os.scandir(cache_dir):
        if not entry.is_dir():
            continue
        size = sum(f.stat().st_size for f in os.scandir(entry.path) if f.is_file())
        models.append((size, entry.name))
    models.sort()
    return models


def is_model_in_use(cache_dir: str, model_name: str) -> bool:
    """Try an exclusive nonblocking flock. Returns True if model is locked."""
    lock_file = os.path.join(cache_dir, f".lock-{model_name}")
    result = subprocess.run(
        ["flock", "--exclusive", "--nonblock", lock_file, "true"],
        capture_output=True,
    )
    return result.returncode != 0


def evict_models(cache_dir: str, min_free_gb: int) -> None:
    """Evict unused models smallest-first until min_free_gb is available."""
    os.makedirs(cache_dir, exist_ok=True)

    free_gb = get_free_gb(cache_dir)
    logger.info("Disk free: %dG (threshold: %dG)", free_gb, min_free_gb)

    if free_gb >= min_free_gb:
        logger.info("Sufficient disk space. No eviction needed.")
        return

    logger.info("Disk space low. Evicting unused models (smallest first)...")
    for size_bytes, model_name in get_cached_models(cache_dir):
        size_gb = size_bytes // (1024**3)

        if is_model_in_use(cache_dir, model_name):
            logger.info("Skipping: %s (~%dG, in use)", model_name, size_gb)
            continue

        logger.info("Evicting: %s (~%dG)", model_name, size_gb)
        shutil.rmtree(os.path.join(cache_dir, model_name), ignore_errors=True)
        etag_file = os.path.join(cache_dir, f".etag-{model_name}")
        if os.path.exists(etag_file):
            os.remove(etag_file)

        free_gb = get_free_gb(cache_dir)
        logger.info("Disk free: %dG", free_gb)
        if free_gb >= min_free_gb:
            logger.info("Sufficient disk space restored.")
            return

    logger.warning("Could not free enough space. Current: %dG", get_free_gb(cache_dir))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evict unused cached models")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--min-free-gb", type=int, required=True)
    args = parser.parse_args()
    evict_models(args.cache_dir, args.min_free_gb)


if __name__ == "__main__":
    main()
