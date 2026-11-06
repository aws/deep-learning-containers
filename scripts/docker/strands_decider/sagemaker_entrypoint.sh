#!/bin/bash
# SageMaker runs `docker run IMAGE serve`. Options after `serve`, or arguments that start with `-`, go to
# `strands-decider serve`; any other command runs as given (e.g. `bash`).
# strands-decider serves GET /ping and POST /invocations (and /health, /v1/systemone) on port 8080; /ping fails
# until the model is loaded and warmed up. The environment maps to `strands-decider serve` options:
#   DECIDER_CHECKPOINT  checkpoint folder. Default: SM_MODEL_DIR (/opt/ml/model, where SageMaker puts the model)
#                       if it holds the checkpoint, else its one subfolder that does (e.g. <model>/ next to base/).
#                       With a base/ folder (the base model, as the image runs offline), the checkpoint's config
#                       must name it: "base_model": "/opt/ml/model/base"
#   DECIDER_DEVICE      cuda (default) or cpu
#   DECIDER_WARMUP      0 skips the warm-up (--no-warmup)
#   DECIDER_MAX_CONCURRENCY  requests run on the model at once, the others wait (--max-concurrency). Default 1:
#                       a request's questions are already batched on the GPU, and without a cap a burst of large
#                       requests runs the GPU out of memory. 0 removes the cap (up to 40 at once)
#   DECIDER_DEVICE_MAP  e.g. auto: split a model larger than one GPU across the visible GPUs (--device-map)
#   DECIDER_ARGS        any other options, split like a shell command line, e.g. '--model-name "fraud triage"'
set -e

if [ "${1:-serve}" = "serve" ]; then
  shift || true
elif [ "${1#-}" = "$1" ]; then
  exec "$@"
fi

# CUDA forward compatibility, for hosts whose driver is older than the image's CUDA.
if [ -f /usr/local/bin/start_cuda_compat.sh ]; then
  source /usr/local/bin/start_cuda_compat.sh >&2 || true
fi
# Serving does not start a login shell, so send telemetry explicitly.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

is_checkpoint() {
  [ -f "$1/strands_decider_config.json" ] || [ -f "$1/hobson_config.json" ]
}
if [ -z "${DECIDER_CHECKPOINT:-}" ]; then
  model_dir=${SM_MODEL_DIR:-/opt/ml/model}
  if is_checkpoint "${model_dir}"; then
    DECIDER_CHECKPOINT=${model_dir}
  else
    found=()
    for dir in "${model_dir}"/*/; do
      if is_checkpoint "${dir%/}"; then found+=("${dir%/}"); fi
    done
    if [ ${#found[@]} -ne 1 ]; then
      echo "No single strands-decider checkpoint in ${model_dir} (found ${#found[@]}); set DECIDER_CHECKPOINT" >&2
      exit 1
    fi
    DECIDER_CHECKPOINT=${found[0]}
  fi
fi

# A base/ folder next to the checkpoint is the base model for an offline load; a published checkpoint names its Hub
# id instead, which can't load offline. Say so now rather than fail inside the model load.
base_dir=$(dirname "${DECIDER_CHECKPOINT}")/base
if [ -d "${base_dir}" ]; then
  python3 - "${DECIDER_CHECKPOINT}" "${base_dir}" <<'EOF'
import json, os, sys
checkpoint, base_dir = sys.argv[1:]
for name in ("strands_decider_config.json", "hobson_config.json"):
    path = os.path.join(checkpoint, name)
    if os.path.exists(path):
        base_model = json.load(open(path)).get("base_model", "")
        if os.path.realpath(base_model) != os.path.realpath(base_dir):
            sys.exit(f"{path}: base_model is {base_model!r}; set it to {base_dir!r} to load the base model "
                     f"from {base_dir} (the image runs offline, HF_HUB_OFFLINE=1)")
        break
EOF
fi

args=()
if [ "${DECIDER_WARMUP:-1}" = "0" ]; then
  args+=(--no-warmup)
fi
if [ -n "${DECIDER_MAX_CONCURRENCY:-}" ] && [ "${DECIDER_MAX_CONCURRENCY}" != "0" ]; then
  args+=(--max-concurrency "${DECIDER_MAX_CONCURRENCY}")
fi
if [ -n "${DECIDER_DEVICE_MAP:-}" ]; then
  args+=(--device-map "${DECIDER_DEVICE_MAP}")
fi
if [ -n "${DECIDER_ARGS:-}" ]; then
  # Split like a shell command line (quotes group, nothing expands).
  mapfile -d '' -t extra < <(python3 -c 'import shlex, sys; sys.stdout.write("\0".join(shlex.split(sys.argv[1])))' "${DECIDER_ARGS}")
  args+=("${extra[@]}")
fi

exec strands-decider serve "${DECIDER_CHECKPOINT}" --host 0.0.0.0 --port 8080 --device "${DECIDER_DEVICE:-cuda}" \
  "${args[@]}" "$@"
