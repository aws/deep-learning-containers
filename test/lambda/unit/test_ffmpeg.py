"""Verify FFmpeg binary and NVENC/NVDEC codec availability — pytorch image."""

import subprocess


def test_ffmpeg_available():
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"ffmpeg not available: {result.stderr}"


def test_ffprobe_available():
    result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"ffprobe not available: {result.stderr}"


def test_h264_nvenc_encoder():
    result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
    assert "h264_nvenc" in result.stdout, "h264_nvenc encoder not compiled into FFmpeg"


def test_h264_cuvid_decoder():
    result = subprocess.run(["ffmpeg", "-decoders"], capture_output=True, text=True, timeout=5)
    assert "h264_cuvid" in result.stdout, "h264_cuvid decoder not compiled into FFmpeg"
