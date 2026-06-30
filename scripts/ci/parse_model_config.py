#!/usr/bin/env python3
"""Parse a model-tests YAML config into JSON matrices for GitHub Actions.

Usage:
    python3 parse_model_config.py --config <path> [--section smoke-test] [--runner-type all]

Output (written to $GITHUB_OUTPUT, or stdout if not in CI):
    codebuild-fleet=<json-array>     (when runner-type is "all")
    runner-scale-sets=<json-array>   (when runner-type is "all")
    matrix=<json-array>              (when runner-type is a specific type)

Handles all framework variations:
    - s3_model → s3_path (prepends s3_prefix)
    - hf_model → model_source="hf"
    - test_fixtures → test_fixtures_paths (prepends test_fixtures_prefix, space-joined)
    - test_cases → test_cases_json (stringified JSON)
"""

import argparse
import json
import os
import subprocess
import sys

try:
    import yaml
except ImportError:
    yaml = None


def transform_model(model: dict, s3_prefix: str, fixtures_prefix: str) -> dict:
    m = dict(model)

    if "s3_model" in m:
        prefix = m.pop("s3_prefix", s3_prefix)
        m["s3_path"] = f"{prefix}/{m.pop('s3_model')}"
        m["model_source"] = "s3"
    elif "hf_model" in m:
        m["model_source"] = "hf"

    if "test_fixtures" in m:
        m["test_fixtures_paths"] = " ".join(
            f"{fixtures_prefix}/{f}" for f in m.pop("test_fixtures")
        )
    else:
        m.setdefault("test_fixtures_paths", "")

    if "test_cases" in m:
        m["test_cases_json"] = json.dumps(m.pop("test_cases"))
    else:
        m.setdefault("test_cases_json", "")

    return m


def load_yaml(path: str) -> dict:
    if yaml:
        with open(path) as f:
            return yaml.safe_load(f)
    result = subprocess.run(
        ["yq", "-o=json", ".", path], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def parse_config(config_path: str, section: str, runner_type: str) -> dict[str, str]:
    cfg = load_yaml(config_path)

    s3_prefix = cfg.get("s3_prefix", "")
    fixtures_prefix = cfg.get("test_fixtures_prefix", "")

    results = {}
    types = ["codebuild-fleet", "runner-scale-sets"] if runner_type == "all" else [runner_type]

    for rt in types:
        models = cfg.get(section, {}).get(rt, []) or []
        transformed = [transform_model(m, s3_prefix, fixtures_prefix) for m in models]
        key = rt if runner_type == "all" else "matrix"
        results[key] = json.dumps(transformed, separators=(",", ":"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse model-tests config into GHA matrix JSON.")
    parser.add_argument("--config", required=True, help="Path to model-tests YAML config")
    parser.add_argument(
        "--section", default="smoke-test", help="Config section (default: smoke-test)"
    )
    parser.add_argument(
        "--runner-type",
        default="all",
        help="Runner type: all, codebuild-fleet, or runner-scale-sets",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    results = parse_config(args.config, args.section, args.runner_type)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            for key, value in results.items():
                f.write(f"{key}={value}\n")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
