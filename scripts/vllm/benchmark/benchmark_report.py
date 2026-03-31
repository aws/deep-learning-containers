"""Generate markdown benchmark report from result artifacts."""

import glob
import json
import os
import sys

import yaml

CONFIG = ".github/config/vllm-model-tests.yml"


def get_tp(extra_args):
    parts = extra_args.split()
    for i, p in enumerate(parts):
        if p == "--tensor-parallel-size" and i + 1 < len(parts):
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
        models[m["name"]] = {**m, "runner": "runner-scale-sets"}
    return models


def main(results_dir):
    models = load_model_config(CONFIG)

    print("# vLLM Benchmark Report\n")

    print("## Throughput\n")
    print(
        "| Model | Runner | TP | Input Len | Output Len | Prompts | Output Tokens/s | Total Tokens/s | Requests/s | Elapsed (s) |"
    )
    print(
        "|-------|--------|----|-----------|------------|---------|-----------------|----------------|------------|-------------|"
    )
    for f in sorted(glob.glob(f"{results_dir}/**/throughput_*.json", recursive=True)):
        name = os.path.basename(f).replace("throughput_", "").replace(".json", "")
        c = models.get(name, {})
        tp = get_tp(c.get("extra_args", ""))
        with open(f) as fh:
            r = json.load(fh)
        output_tps = r.get("output_tokens_per_second", 0)
        print(
            f"| {name} | {c.get('runner', 'unknown')} | {tp} "
            f"| {c.get('input_len', '')} | {c.get('output_len', '')} "
            f"| {c.get('num_prompts', '')} | {output_tps:.2f} "
            f"| {r['tokens_per_second']:.2f} "
            f"| {r['requests_per_second']:.2f} | {r['elapsed_time']:.2f} |"
        )

    print("\n## Latency\n")
    print("| Model | Runner | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |")
    print("|-------|--------|----|------------|---------|---------|---------|---------|")
    for f in sorted(glob.glob(f"{results_dir}/**/latency_*.json", recursive=True)):
        name = os.path.basename(f).replace("latency_", "").replace(".json", "")
        c = models.get(name, {})
        tp = get_tp(c.get("extra_args", ""))
        with open(f) as fh:
            r = json.load(fh)
        p = r.get("percentiles", {})
        print(
            f"| {name} | {c.get('runner', 'unknown')} | {tp} "
            f"| {c.get('batch_size', '')} | {r['avg_latency']:.4f} "
            f"| {p.get('50', 0):.4f} | {p.get('90', 0):.4f} | {p.get('99', 0):.4f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
