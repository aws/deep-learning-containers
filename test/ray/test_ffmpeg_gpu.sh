#!/bin/bash
# Test FFmpeg GPU-accelerated video encoding/decoding.
# Runs inside the container with GPU access
#
# Usage (CI): docker exec ${CONTAINER_ID} bash /workdir/test/ray/test_ffmpeg_gpu.sh
# Usage (manual): bash test/ray/test_ffmpeg_gpu.sh
set -eo pipefail
trap 'rm -f /tmp/gpu_enc_* /tmp/gpu_src* /tmp/gpu_dec_* /tmp/gpu_pipe_* /tmp/bench_*' EXIT

PASS=0; FAIL=0; WARN=0
pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN: $1"; WARN=$((WARN + 1)); }

echo "=========================================="
echo "FFmpeg GPU Acceleration Test"
echo "=========================================="
echo

# 1. Build flags
echo "[1/7] Checking GPU build configuration..."
BUILD_CONF=$(ffmpeg -buildconf 2>&1)

for flag in enable-cuda-nvcc; do
    if echo "${BUILD_CONF}" | grep -q "${flag}"; then
        pass "--${flag}"
    else
        fail "--${flag} MISSING -- FFmpeg was not compiled with GPU support"
    fi
done
echo

# 2. NVIDIA GPU visible
echo "[2/7] Checking NVIDIA GPU is accessible..."
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)

if [ -n "$GPU_INFO" ]; then
    pass "GPU detected: ${GPU_INFO}"
else
    fail "nvidia-smi failed -- no GPU visible"
    echo "  Make sure the container was started with --gpus all"
    exit 1
fi
echo

# 3. GPU video encoders registered
echo "[3/7] Checking GPU video encoders (NVENC)..."
ENCODERS=$(ffmpeg -encoders 2>/dev/null || true)

GPU_ENCODERS=(
    "h264_nvenc:H.264 NVENC"
    "hevc_nvenc:HEVC NVENC"
)

for entry in "${GPU_ENCODERS[@]}"; do
    IFS=: read -r enc label <<< "$entry"
    if echo "${ENCODERS}" | grep -q "${enc}"; then
        pass "${label} (${enc})"
    else
        fail "${label} (${enc}) not registered"
    fi
done
echo

# 4. GPU video decoders registered
echo "[4/7] Checking GPU video decoders (CUVID/NVDEC)..."
DECODERS=$(ffmpeg -decoders 2>/dev/null || true)
HWACCELS=$(ffmpeg -hwaccels 2>/dev/null || true)

for method in cuda cuvid; do
    if echo "${HWACCELS}" | grep -q "${method}"; then
        pass "hwaccel: ${method}"
    else
        warn "hwaccel: ${method} not listed"
    fi
done

GPU_DECODERS=(
    "h264_cuvid:H.264 CUVID"
    "hevc_cuvid:HEVC CUVID"
)

for entry in "${GPU_DECODERS[@]}"; do
    IFS=: read -r dec label <<< "$entry"
    if echo "${DECODERS}" | grep -q "${dec}"; then
        pass "${label} (${dec})"
    else
        warn "${label} (${dec}) not registered"
    fi
done
echo

# 5. GPU encode end-to-end
echo "[5/7] Testing GPU video encoding (NVENC)..."

for entry in "${GPU_ENCODERS[@]}"; do
    IFS=: read -r enc label <<< "$entry"
    if ffmpeg -y -f lavfi -i testsrc=duration=2:size=640x480:rate=30 \
        -c:v ${enc} -f mp4 /tmp/gpu_enc_${enc}.mp4 2>/dev/null; then
        CODEC=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=codec_name -of csv=p=0 \
            /tmp/gpu_enc_${enc}.mp4 2>/dev/null || true)
        pass "${label}: encoded successfully (output codec: ${CODEC})"
    else
        fail "${label}: encoding failed"
    fi
done
echo

# 6. GPU decode end-to-end
echo "[6/7] Testing GPU video decoding (NVDEC)..."

# hwaccel cuda decode
ffmpeg -y -f lavfi -i testsrc=duration=2:size=640x480:rate=30 \
    -c:v h264_nvenc /tmp/gpu_src.mp4 2>/dev/null || true

if ffmpeg -y -hwaccel cuda -i /tmp/gpu_src.mp4 \
    -c:v rawvideo -f rawvideo /tmp/gpu_dec_hwaccel.raw 2>/dev/null; then
    BYTES=$(stat -c%s /tmp/gpu_dec_hwaccel.raw 2>/dev/null || echo 0)
    if [ "$BYTES" -gt 0 ]; then
        pass "hwaccel cuda decode -> raw frames"
    else
        fail "hwaccel cuda decode produced empty output"
    fi
else
    fail "hwaccel cuda decode failed"
fi

# Full GPU pipeline: decode -> encode
ffmpeg -y -f lavfi -i testsrc=duration=2:size=1920x1080:rate=30 \
    -c:v h264_nvenc /tmp/gpu_pipe_src.mp4 2>/dev/null || true

if ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
    -i /tmp/gpu_pipe_src.mp4 \
    -c:v h264_nvenc /tmp/gpu_pipe_out.mp4 2>/dev/null; then
    RES=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=p=0 \
        /tmp/gpu_pipe_out.mp4 2>/dev/null || true)
    pass "full GPU pipeline: nvdec -> nvenc (output: ${RES})"
else
    fail "full GPU pipeline failed"
fi
echo

# 7. GPU vs CPU benchmark
echo "[7/7] GPU vs CPU encoding benchmark (5s 1080p)..."
# Note: Compares mjpeg (CPU, intra-only) vs h264_nvenc (GPU, inter-frame).
# Demonstrates GPU offload benefit, not an apples-to-apples codec comparison.

START=$(date +%s%N)
ffmpeg -y -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 \
    -c:v mjpeg -q:v 5 -f matroska /tmp/bench_cpu.mkv 2>/dev/null
END=$(date +%s%N)
MJPEG_TIME=$(( (END - START) / 1000000 ))

START=$(date +%s%N)
ffmpeg -y -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 \
    -c:v h264_nvenc -preset p4 -f mp4 /tmp/bench_gpu.mp4 2>/dev/null
END=$(date +%s%N)
NVENC_TIME=$(( (END - START) / 1000000 ))

echo "  CPU (mjpeg):         ${MJPEG_TIME}ms"
echo "  GPU (h264_nvenc p4): ${NVENC_TIME}ms"

if [ -n "${MJPEG_TIME}" ] && [ -n "${NVENC_TIME}" ] && [ "${NVENC_TIME}" -gt 0 ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.1f\", ${MJPEG_TIME} / ${NVENC_TIME}}" 2>/dev/null || true)
    [ -n "${SPEEDUP}" ] && pass "GPU speedup: ${SPEEDUP}x"
fi
echo

echo "NOTE: Audio encoding/decoding is always CPU-based."
echo "      Use test_ffmpeg_codecs.sh to verify audio codec support."
echo

# Summary
echo "=========================================="
echo "Results: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
if [ "${FAIL}" -gt 0 ]; then
    echo "GPU ACCELERATION TEST FAILED"
    exit 1
else
    echo "GPU ACCELERATION TEST PASSED"
fi
echo "=========================================="
