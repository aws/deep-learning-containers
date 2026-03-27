"""Download a model from S3 with ETag-based caching.

Checks if the model is already cached and the S3 ETag matches. If so,
skips the download. Otherwise downloads the tarball, extracts it, and
saves the ETag for future cache checks. The ETag is written last so an
interrupted download won't leave a stale cache marker.
"""

import argparse
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def get_s3_etag(s3_path: str) -> str:
    """Get the ETag of an S3 object."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ETag",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def is_cached(model_dir: str, etag_file: str, s3_etag: str) -> bool:
    """Check if model is cached and ETag matches."""
    if not os.path.isdir(model_dir) or not os.path.isfile(etag_file):
        return False
    cached_etag = open(etag_file).read().strip()
    logger.info("Cached ETag: %s", cached_etag)
    return cached_etag == s3_etag


def download_and_extract(s3_path: str, model_name: str, cache_dir: str, model_dir: str) -> None:
    """Download tarball from S3 and extract to model_dir."""
    tarball = os.path.join(cache_dir, f"{model_name}.tar.gz")

    logger.info("Downloading %s from %s...", model_name, s3_path)
    subprocess.run(["aws", "s3", "cp", s3_path, tarball], check=True)

    logger.info("Extracting %s (this may take several minutes)...", model_name)
    subprocess.run(
        ["tar", "xzf", tarball, "-C", model_dir, "--checkpoint=2000000", "--checkpoint-action=dot"],
        check=True,
    )
    logger.info("Extraction complete.")
    os.remove(tarball)

    # Flatten if tarball contains a single subdirectory
    entries = list(os.scandir(model_dir))
    if len(entries) == 1 and entries[0].is_dir():
        subdir = entries[0].path
        for item in os.scandir(subdir):
            shutil.move(item.path, model_dir)
        os.rmdir(subdir)


def download_model(s3_path: str, model_name: str, cache_dir: str) -> str:
    """Download model if not cached or stale. Returns model directory path."""
    model_dir = os.path.join(cache_dir, model_name)
    etag_file = os.path.join(cache_dir, f".etag-{model_name}")
    os.makedirs(cache_dir, exist_ok=True)

    s3_etag = get_s3_etag(s3_path)
    logger.info("S3 ETag: %s", s3_etag)

    if is_cached(model_dir, etag_file, s3_etag):
        logger.info("Model '%s' is cached and up to date.", model_name)
        return model_dir

    if os.path.isdir(model_dir):
        logger.info("Model '%s' is stale. Re-downloading.", model_name)
        shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    download_and_extract(s3_path, model_name, cache_dir, model_dir)

    # ETag written last — incomplete download won't have an ETag,
    # so next run will re-download.
    with open(etag_file, "w") as f:
        f.write(s3_etag)
    logger.info("Model '%s' downloaded and cached.", model_name)
    return model_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download model from S3 with caching")
    parser.add_argument("--s3-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    download_model(args.s3_path, args.model_name, args.cache_dir)


if __name__ == "__main__":
    main()
