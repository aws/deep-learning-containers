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


def _parse_env_vars(env_vars_str):
    """Parse 'KEY=VAL KEY2=VAL2' into dict."""
    result = {}
    for item in (env_vars_str or "").split():
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
    return result


def load_model_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bm = cfg.get("benchmark", {})
    models = {}
    for m in bm.get("codebuild-fleet", []):
        env = _parse_env_vars(m.get("env_vars", ""))
        models[m["name"]] = {**m, "runner": m.get("fleet", ""), **env}
    for m in bm.get("runner-scale-sets", []):
        env = _parse_env_vars(m.get("env_vars", ""))
        models[m["name"]] = {**m, "runner": m.get("runner_label", "gpu-efa-runners"), **env}
    return models


def main(results_dir):
    models = load_model_config(CONFIG)

    print("# SGLang Benchmark Report\n")

    print("## Throughput\n")
    print(
        "| Model | Runner | TP | Input Len | Output Len | Prompts "
        "| Output Tok/s | Total Tok/s | Requests/s | Elapsed (s) |"
    )
    print(
        "|-------|--------|----|-----------|------------|---------|"
        "--------------|-------------|------------|-------------|"
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
        total_tps = r.get("total_throughput", 0)
        rps = r.get("request_throughput", 0)
        duration = r.get("duration", 0)
        input_len = c.get("BENCHMARK_INPUT_LEN", c.get("input_len", ""))
        output_len = c.get("BENCHMARK_OUTPUT_LEN", c.get("output_len", ""))
        num_prompts = c.get("BENCHMARK_NUM_PROMPTS", c.get("num_prompts", ""))
        print(
            f"| {name} | {runner} | {tp} "
            f"| {input_len} | {output_len} "
            f"| {num_prompts} | {output_tps:.2f} "
            f"| {total_tps:.2f} | {rps:.2f} | {duration:.2f} |"
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
