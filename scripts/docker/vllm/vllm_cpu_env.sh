#!/usr/bin/env bash
# CPU env for the vLLM entrypoints: memory-aware VLLM_CPU_KVCACHE_SPACE default (GPU %-util is inert on CPU) + tcmalloc preload.

if [ -z "${VLLM_CPU_KVCACHE_SPACE:-}" ]; then
  total_kb=$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | tr -dc '0-9')
  total_gib=$(( ${total_kb:-0} / 1024 / 1024 ))
  # KV cache shares RAM with the model weights and a roughly fixed ~10 GiB of
  # runtime overhead (PyTorch, the inductor warmup compile, framework buffers).
  # vLLM checks the requested KV against RAM still free *after* that overhead, so
  # a flat 40%-of-total request overshoots on entry-class instances: on a 16 GiB
  # box 40% is 6 GiB but only ~5.5 GiB is free post-warmup, and EngineCore aborts.
  # Take the smaller of 40%-of-total (caps KV on large boxes, leaving room for
  # bigger models) and total-minus-headroom (guarantees the fixed overhead fits
  # on small boxes). Floor at 2 GiB.
  pct=$(( total_gib * 40 / 100 ))
  headroom=$(( total_gib - 11 ))
  kv=$(( pct < headroom ? pct : headroom ))
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
