#!/usr/bin/env bash
set -uo pipefail

# Static checks for arm64 GPU vLLM images (no GPU required; runs on the Graviton fleet).
#
# Covers what arm64 GPU hosts (GB200/GH200) need that x86 CI cannot catch:
#   - every shared library loads on 64 KB page kernels (ELF LOAD alignment >= 64 KB)
#   - vLLM and the NCCL test binary carry GB200 (sm_100) kernels
#   - JIT builds can link -lcuda (CUDA driver stub on LIBRARY_PATH)
#   - Mooncake has the EFA and multi-node NVLink (MNNVL) transports, and libfabric resolves
# Functional coverage (serving, NCCL, EFA, PD disaggregation) needs GB200 hardware.

PYTHON=/opt/venv/bin/python3
# Pipelines below use "grep >/dev/null", not "grep -q": an early grep exit SIGPIPEs the
# producer, which pipefail reports as a failure.
FAILED=0

pass() { echo "PASS: $1"; }
fail() {
  echo "FAIL: $1"
  FAILED=1
}

echo "=============== 64 KB page alignment ==============="
# The kernel can only mmap a LOAD segment whose alignment is a multiple of its page size.
CHECKED=0
LOW=0
while IFS= read -r -d '' f; do
  ALIGN=$(readelf -lW "$f" 2>/dev/null | awk '$1 == "LOAD" {print $NF}' | sort -u | head -1)
  [ -z "$ALIGN" ] && continue
  CHECKED=$((CHECKED + 1))
  if [ $((ALIGN)) -lt 65536 ]; then
    LOW=$((LOW + 1))
    [ "$LOW" -le 20 ] && echo "  LOAD align ${ALIGN}: $f"
  fi
done < <(find / -xdev \( -path /proc -o -path /sys \) -prune -o -type f \( -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null)
if [ "$CHECKED" -eq 0 ]; then
  fail "no shared libraries found to check"
elif [ "$LOW" -eq 0 ]; then
  pass "all ${CHECKED} shared libraries have LOAD alignment >= 64 KB"
else
  fail "${LOW} of ${CHECKED} shared libraries have LOAD alignment < 64 KB"
fi

echo "=============== vLLM ==============="
VLLM_DIR=$($PYTHON -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("vllm").origin))' 2>/dev/null)
if [ -z "$VLLM_DIR" ]; then
  fail "vllm package not found"
else
  # The stable-ABI extension holds vLLM's main CUDA kernels.
  STABLE_SO=$(ls "$VLLM_DIR"/_C_stable_libtorch*.so 2>/dev/null | head -1)
  if [ -z "$STABLE_SO" ]; then
    fail "vllm._C_stable_libtorch missing"
  elif cuobjdump --list-elf "$STABLE_SO" 2>/dev/null | grep "sm_100" >/dev/null; then
    pass "vllm._C_stable_libtorch present with sm_100 kernels"
  else
    fail "vllm._C_stable_libtorch has no sm_100 kernels"
  fi
fi

echo "=============== NCCL tests ==============="
if [ -x /usr/local/bin/all_reduce_perf ]; then
  if cuobjdump --list-elf /usr/local/bin/all_reduce_perf 2>/dev/null | grep "sm_100" >/dev/null; then
    pass "all_reduce_perf has sm_100 kernels"
  else
    fail "all_reduce_perf has no sm_100 kernels"
  fi
else
  fail "/usr/local/bin/all_reduce_perf not found"
fi

echo "=============== CUDA JIT link ==============="
STUB_FOUND=0
IFS=: read -ra LIB_DIRS <<<"${LIBRARY_PATH:-}"
for d in "${LIB_DIRS[@]}"; do
  [ -n "$d" ] && [ -e "$d/libcuda.so" ] && STUB_FOUND=1
done
if [ "$STUB_FOUND" -eq 1 ]; then
  pass "libcuda.so stub on LIBRARY_PATH (${LIBRARY_PATH})"
else
  fail "no libcuda.so on LIBRARY_PATH='${LIBRARY_PATH:-}'; FlashInfer JIT cannot link -lcuda"
fi

echo "=============== Mooncake ==============="
MC_DIR=$($PYTHON -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("mooncake").origin))' 2>/dev/null)
if [ -z "$MC_DIR" ] || [ ! -f "$MC_DIR/engine.so" ]; then
  fail "mooncake engine.so not found"
else
  if nm -D "$MC_DIR/engine.so" 2>/dev/null | grep "EfaTransport" >/dev/null; then
    pass "mooncake engine.so has the EFA transport"
  else
    fail "mooncake engine.so has no EFA transport (needs mooncake-transfer-engine >= 0.3.14)"
  fi
  # Transport names also appear in Mooncake's "unsupported transport" errors, so check the class.
  if nm -DC "$MC_DIR/engine.so" 2>/dev/null | grep "mooncake::NvlinkTransport" >/dev/null; then
    pass "mooncake engine.so has the MNNVL transport"
  else
    fail "mooncake engine.so has no MNNVL transport"
  fi
fi
# engine.so dlopens libfabric rather than linking it.
if $PYTHON -c 'import ctypes; ctypes.CDLL("libfabric.so.1")' 2>/dev/null; then
  pass "libfabric.so.1 resolves"
else
  fail "libfabric.so.1 does not resolve"
fi

echo
if [ "$FAILED" -ne 0 ]; then
  echo "arm64 GPU checks FAILED"
  exit 1
fi
echo "arm64 GPU checks passed"
