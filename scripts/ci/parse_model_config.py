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
        m["s3_path"] = f"{s3_prefix}/{m.pop('s3_model')}"
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


def matches_customer_type(model: dict, customer_type: str) -> bool:
    """A model runs on a config unless it pins a different customer_type.

    Models without a ``customer_type`` field run everywhere (backward
    compatible). A model that pins e.g. ``customer_type: sagemaker`` only runs
    when the config's customer type matches — used to gate tests for features
    that exist on only one container variant (e.g. the SageMaker routing
    middleware, which adds the JSON->multipart video path absent on EC2).
    """
    pinned = model.get("customer_type")
    if not pinned or not customer_type:
        return True
    return pinned == customer_type


def parse_config(
    config_path: str, section: str, runner_type: str, customer_type: str = ""
) -> dict[str, str]:
    cfg = load_yaml(config_path)

    s3_prefix = cfg.get("s3_prefix", "")
    fixtures_prefix = cfg.get("test_fixtures_prefix", "")

    results = {}
    types = ["codebuild-fleet", "runner-scale-sets"] if runner_type == "all" else [runner_type]

    for rt in types:
        models = cfg.get(section, {}).get(rt, []) or []
        models = [m for m in models if matches_customer_type(m, customer_type)]
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
    parser.add_argument(
        "--customer-type",
        default="",
        help="Config customer type (e.g. ec2, sagemaker). When set, drops models "
        "that pin a different customer_type. Empty = include all models.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    results = parse_config(args.config, args.section, args.runner_type, args.customer_type)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            for key, value in results.items():
                f.write(f"{key}={value}\n")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
