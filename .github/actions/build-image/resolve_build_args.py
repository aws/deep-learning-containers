#!/usr/bin/env python3
"""Resolve build-args from image config file.

Reads the build: block from a YAML config file, skips reserved keys, and
writes each remaining key as UPPER_CASE=value to $GITHUB_ENV. Also writes
EXTRA_BUILD_ARGS (space-separated list of key names) for build_image.sh.

Reserved keys (not forwarded as --build-arg):
  dockerfile, target

Keys in PRESERVE_CASE_KEYS are forwarded verbatim instead of upper-cased, for
Dockerfiles that declare a lower-case ARG. Docker build-args are case-sensitive,
so an upper-cased name silently fails to bind to a lower-case ARG.

Usage in GitHub Actions:
  - name: Resolve build args
    run: python3 scripts/ci/resolve_build_args.py --config-file ${{ env.CONFIG_FILE }}

Local usage:
  python3 scripts/ci/resolve_build_args.py --config-file .github/refactor/config/image/vllm/ec2-amzn2023.yml

Requires: pyyaml (pip install pyyaml) or yq on PATH as fallback.
"""

import argparse
import json
import os
import subprocess
import sys

RESERVED_KEYS = {"dockerfile", "target"}

# Keys whose Dockerfile ARG is lower-case, so the name must not be upper-cased.
# vLLM's upstream setup.py reads TORCH_CUDA_ARCH_LIST from the environment, which
# the Dockerfile sets from a lower-case `ARG torch_cuda_arch_list`.
PRESERVE_CASE_KEYS = {"torch_cuda_arch_list"}


def load_yaml(path):
    """Load YAML file, trying pyyaml first, falling back to yq."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        result = subprocess.run(
            ["yq", "-o=json", ".", path], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve build-args from image config file.")
    parser.add_argument("--config-file", required=True, help="Path to the image config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.config_file):
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml(args.config_file)
    build = config.get("build", {})

    github_env = os.environ.get("GITHUB_ENV")
    keys = []

    for key, value in build.items():
        if key in RESERVED_KEYS:
            continue
        env_key = key if key in PRESERVE_CASE_KEYS else key.upper()
        keys.append(env_key)

        if github_env:
            with open(github_env, "a") as f:
                f.write(f"{env_key}={value}\n")

        print(f"  {env_key}={value}")

    extra = " ".join(keys)

    if github_env:
        with open(github_env, "a") as f:
            f.write(f"EXTRA_BUILD_ARGS={extra}\n")

    print(f"\nEXTRA_BUILD_ARGS={extra}")
    print(f"Total: {len(keys)} build-args resolved")


if __name__ == "__main__":
    main()
