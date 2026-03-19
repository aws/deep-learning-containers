#!/bin/bash
# Test FFmpeg software (CPU) video and audio encoding/decoding.
# Works on both CPU and GPU images
#
# Usage (CI): docker exec ${CONTAINER_ID} bash /workdir/test/ray/test_ffmpeg_codecs.sh
# Usage (manual): bash test/ray/test_ffmpeg_codecs.sh
set -eo pipefail
trap 'rm -f /tmp/enc_* /tmp/src_* /tmp/dec_* /tmp/asrc_* /tmp/adec_* /tmp/roundtrip.*' EXIT

PASS=0; FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=========================================="
echo "FFmpeg Codec Test (CPU encode/decode)"
echo "=========================================="
echo

# 1. Binary check
echo "[1/6] Verifying FFmpeg and ffprobe are available..."
FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 || true)
FFPROBE_VERSION=$(ffprobe -version 2>&1 | head -1 || true)

if [ -n "$FFMPEG_VERSION" ]; then
    pass "ffmpeg: ${FFMPEG_VERSION}"
else
    fail "ffmpeg not found"; exit 1
fi
if [ -n "$FFPROBE_VERSION" ]; then
    pass "ffprobe: ${FFPROBE_VERSION}"
else
    fail "ffprobe not found"; exit 1
fi
echo

# 2. Video encode (CPU)
echo "[2/6] Testing video encoding (CPU)..."
VIDEO_ENCODERS=(
    "png:rgb24:PNG"
    "mjpeg:yuvj420p:MJPEG"
)

for entry in "${VIDEO_ENCODERS[@]}"; do
    IFS=: read -r enc pixfmt label <<< "$entry"
    RESULT=$(ffmpeg -y -f lavfi -i testsrc=duration=1:size=320x240:rate=10 \
        -c:v ${enc} -pix_fmt ${pixfmt} -frames:v 10 /tmp/enc_${enc}.mkv 2>/dev/null && \
        ffprobe -v error -select_streams v:0 \
            -show_entries stream=codec_name,width,height \
            -of csv=p=0 /tmp/enc_${enc}.mkv 2>/dev/null || true)
    if echo "$RESULT" | grep -q "320"; then
        pass "encode ${label} (${enc}) -> valid output"
    else
        fail "encode ${label} (${enc})"
    fi
done
echo

# 3. Video decode (CPU)
echo "[3/6] Testing video decoding (CPU)..."
VIDEO_DECODE_TESTS=(
    "mjpeg:MJPEG"
    "png:PNG"
)

for entry in "${VIDEO_DECODE_TESTS[@]}"; do
    IFS=: read -r enc label <<< "$entry"
    ffmpeg -y -f lavfi -i testsrc=duration=1:size=320x240:rate=10 \
        -c:v ${enc} -frames:v 10 /tmp/src_${enc}.mkv 2>/dev/null || true
    ffmpeg -y -i /tmp/src_${enc}.mkv \
        -c:v rawvideo -f rawvideo /tmp/dec_${enc}.raw 2>/dev/null || true
    BYTES=$(stat -c%s /tmp/dec_${enc}.raw 2>/dev/null || echo 0)
    if [ "$BYTES" -gt 0 ]; then
        pass "decode ${label} -> raw frames"
    else
        fail "decode ${label}"
    fi
done
echo

# 4. Audio encode
echo "[4/6] Testing audio encoding..."
AUDIO_ENCODERS=(
    "aac:aac:AAC"
    "flac:flac:FLAC"
    "pcm_s16le:pcm_s16le:PCM"
)

for entry in "${AUDIO_ENCODERS[@]}"; do
    IFS=: read -r enc expected_codec label <<< "$entry"
    RESULT=$(ffmpeg -y -f lavfi -i sine=frequency=440:duration=2 \
        -c:a ${enc} /tmp/enc_${enc}.mkv 2>/dev/null && \
        ffprobe -v error -select_streams a:0 \
            -show_entries stream=codec_name,sample_rate,channels \
            -of csv=p=0 /tmp/enc_${enc}.mkv 2>/dev/null || true)
    if [ -n "$RESULT" ]; then
        pass "encode ${label} (${enc}) -> ${RESULT}"
    else
        fail "encode ${label} (${enc})"
    fi
done
echo

# 5. Audio decode
echo "[5/6] Testing audio decoding..."
AUDIO_DECODE_TESTS=(
    "aac:AAC"
    "flac:FLAC"
)

for entry in "${AUDIO_DECODE_TESTS[@]}"; do
    IFS=: read -r enc label <<< "$entry"
    ffmpeg -y -f lavfi -i sine=frequency=440:duration=2 \
        -c:a ${enc} /tmp/asrc_${enc}.mkv 2>/dev/null || true
    ffmpeg -y -i /tmp/asrc_${enc}.mkv \
        -c:a pcm_s16le -f wav /tmp/adec_${enc}.wav 2>/dev/null || true
    BYTES=$(stat -c%s /tmp/adec_${enc}.wav 2>/dev/null || echo 0)
    if [ "$BYTES" -gt 1000 ]; then
        pass "decode ${label} -> PCM wav"
    else
        fail "decode ${label}"
    fi
done
echo

# 6. Roundtrip: video+audio -> container -> probe
echo "[6/6] Testing full roundtrip (video + audio -> container -> probe)..."
CONTAINERS=(
    "mp4:mjpeg:aac"
    "mkv:mjpeg:flac"
)

for entry in "${CONTAINERS[@]}"; do
    IFS=: read -r fmt venc aenc <<< "$entry"
    ffmpeg -y \
        -f lavfi -i testsrc=duration=2:size=320x240:rate=10 \
        -f lavfi -i sine=frequency=440:duration=2 \
        -c:v ${venc} -c:a ${aenc} -shortest \
        /tmp/roundtrip.${fmt} 2>/dev/null || true
    V_COUNT=$(ffprobe -v error -select_streams v -show_entries stream=index \
        -of csv=p=0 /tmp/roundtrip.${fmt} 2>/dev/null | wc -l)
    A_COUNT=$(ffprobe -v error -select_streams a -show_entries stream=index \
        -of csv=p=0 /tmp/roundtrip.${fmt} 2>/dev/null | wc -l)
    if [ "$V_COUNT" -ge 1 ] && [ "$A_COUNT" -ge 1 ]; then
        pass "${fmt}: ${venc}+${aenc} -> mux -> probe (video+audio)"
    else
        fail "${fmt}: roundtrip failed"
    fi
done
echo

# Summary
echo "=========================================="
echo "Results: ${PASS} passed, ${FAIL} failed"
if [ "${FAIL}" -gt 0 ]; then
    echo "FAILED"
    exit 1
else
    echo "ALL CODEC TESTS PASSED"
fi
echo "=========================================="
