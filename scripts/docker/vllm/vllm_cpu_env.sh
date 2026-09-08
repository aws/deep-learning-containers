#!/usr/bin/env bash
# CPU env for the vLLM entrypoints: memory-aware VLLM_CPU_KVCACHE_SPACE default (GPU %-util is inert on CPU) + tcmalloc preload.

if [ -z "${VLLM_CPU_KVCACHE_SPACE:-}" ]; then
  # grep/tr, not awk (no gawk on minimal AL2023).
  total_kb=$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | tr -dc '0-9')
  total_gib=$(( ${total_kb:-0} / 1024 / 1024 ))
  # ~40% of RAM, floor 2 GiB.
  kv=$(( total_gib * 40 / 100 ))
  [ "${kv}" -lt 2 ] && kv=2
  export VLLM_CPU_KVCACHE_SPACE="${kv}"
  echo "INFO: VLLM_CPU_KVCACHE_SPACE defaulted to ${kv} GiB (from ${total_gib} GiB RAM); set it explicitly to override."
fi

for _lib in /usr/lib64/libtcmalloc_minimal.so.4 /usr/lib64/libtcmalloc.so.4; do
  if [ -f "${_lib}" ]; then
    export LD_PRELOAD="${_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
    break
  fi
done
