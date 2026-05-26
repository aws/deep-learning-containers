"""Generate markdown benchmark report from SGLang result artifacts."""

import glob
import json
import os
import sys

import yaml

CONFIG = ".github/config/model-tests/sglang-model-tests.yml"


def _parse_artifact_name(filename):
    """Parse model name and runner from filename like throughput_qwen3-32b_gpu-efa-runners.json."""
    base = os.path.basename(filename).replace("throughput_", "", 1).replace(".json", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return base, "unknown"


def get_tp(extra_args):
    parts = extra_args.split()
    for i, p in enumerate(parts):
        if p in ("--tp", "--tensor-parallel-size") and i + 1 < len(parts):
            return parts[i + 1]
    return "1"


def load_model_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bm = cfg.get("benchmark", {})
    models = {}
    for m in bm.get("codebuild-fleet", []):
        models[m["name"]] = {**m, "runner": m.get("fleet", "")}
    for m in bm.get("runner-scale-sets", []):
        models[m["name"]] = {**m, "runner": m.get("runner_label", "gpu-efa-runners")}
    return models


def main(results_dir):
    models = load_model_config(CONFIG)

    print("# SGLang Benchmark Report\n")

    print("## Throughput\n")
    print(
        "| Model | Runner | TP | Input Len | Output Len | Prompts "
        "| Output Tok/s | Request/s | Total Time (s) |"
    )
    print(
        "|-------|--------|----|-----------|------------|---------|"
        "-----------------|-----------|----------------|"
    )
    for f in sorted(glob.glob(f"{results_dir}/**/throughput_*.json", recursive=True)):
        name, runner = _parse_artifact_name(f)
        c = models.get(name, {})
        tp = get_tp(c.get("extra_args", ""))
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        output_tps = r.get("output_throughput", 0)
        rps = r.get("request_throughput", 0)
        total_time = r.get("total_time", 0)
        print(
            f"| {name} | {runner} | {tp} "
            f"| {c.get('input_len', '')} | {c.get('output_len', '')} "
            f"| {c.get('num_prompts', '')} | {output_tps:.2f} "
            f"| {rps:.2f} | {total_time:.2f} |"
        )

    print("\n## Latency (Online Serving)\n")
    print(
        "| Model | Runner | TP "
        "| Mean TTFT (ms) | p99 TTFT (ms) "
        "| Mean TPOT (ms) | p99 TPOT (ms) "
        "| Mean ITL (ms) | p99 ITL (ms) |"
    )
    print(
        "|-------|--------|----|"
        "----------------|---------------|"
        "----------------|---------------|"
        "---------------|--------------|"
    )
    for f in sorted(glob.glob(f"{results_dir}/**/throughput_*.json", recursive=True)):
        name, runner = _parse_artifact_name(f)
        c = models.get(name, {})
        tp = get_tp(c.get("extra_args", ""))
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        print(
            f"| {name} | {runner} | {tp} "
            f"| {r.get('mean_ttft_ms', 0):.2f} | {r.get('p99_ttft_ms', 0):.2f} "
            f"| {r.get('mean_tpot_ms', 0):.2f} | {r.get('p99_tpot_ms', 0):.2f} "
            f"| {r.get('mean_itl_ms', 0):.2f} | {r.get('p99_itl_ms', 0):.2f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
