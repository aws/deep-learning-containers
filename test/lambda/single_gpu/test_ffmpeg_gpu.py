"""Validate FFmpeg GPU-accelerated transcode on a single GPU — pytorch image.

Covers NVIDIA Video Codec SDK FFmpeg guide sections:
  §3.1: 1:1 HWACCEL transcode (GPU decode + GPU encode)
  §3.2: HWACCEL transcode with scale_npp / scale_cuda GPU scaling
  §4.2: Standalone NVDEC decode
  §4.3: High-quality latency-tolerant preset
  §4.4: Low-latency preset
  §5.2: Spatial AQ
"""

import os
import subprocess
import tempfile

import pytest


@pytest.fixture(scope="module")
def gpu_input_video():
    """Generate a 2-second 720p H.264 input using NVENC (shared across tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "input.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-vsync",
                "0",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=1280x720:rate=25",
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        yield tmpdir, path


def _run(args, timeout=60):
    subprocess.run(args, capture_output=True, timeout=timeout, check=True)


def test_hwaccel_transcode(gpu_input_video):
    """§3.1 — 1:1 HWACCEL transcode: GPU decode → GPU encode."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_31.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-c:v",
            "h264_nvenc",
            "-b:v",
            "5M",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_scale_npp(gpu_input_video):
    """§3.2 — GPU scaling with scale_npp."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_32_npp.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-vf",
            "scale_npp=640:360",
            "-c:v",
            "h264_nvenc",
            "-b:v",
            "2M",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_scale_cuda(gpu_input_video):
    """§3.2 — GPU scaling with scale_cuda."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_32_cuda.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-vf",
            "scale_cuda=640:360",
            "-c:v",
            "h264_nvenc",
            "-b:v",
            "2M",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_nvdec_standalone(gpu_input_video):
    """§4.2 — Standalone NVDEC decode to raw YUV."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_42.yuv")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-c:v",
            "h264_cuvid",
            "-i",
            inp,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_high_quality_preset(gpu_input_video):
    """§4.3 — High-quality latency-tolerant preset (p6, tune hq, B-frames, temporal AQ, lookahead)."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_43_hq.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-tune",
            "hq",
            "-b:v",
            "5M",
            "-bufsize",
            "5M",
            "-maxrate",
            "10M",
            "-qmin",
            "0",
            "-g",
            "250",
            "-bf",
            "3",
            "-b_ref_mode",
            "middle",
            "-temporal-aq",
            "1",
            "-rc-lookahead",
            "20",
            "-i_qfactor",
            "0.75",
            "-b_qfactor",
            "1.1",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_low_latency_preset(gpu_input_video):
    """§4.4 — Low-latency preset (p2, tune ll, CBR)."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_44_ll.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p2",
            "-tune",
            "ll",
            "-b:v",
            "5M",
            "-bufsize",
            "167K",
            "-maxrate",
            "10M",
            "-qmin",
            "0",
            out,
        ]
    )
    assert os.path.getsize(out) > 0


def test_spatial_aq(gpu_input_video):
    """§5.2 — Spatial AQ."""
    tmpdir, inp = gpu_input_video
    out = os.path.join(tmpdir, "out_52_saq.mp4")
    _run(
        [
            "ffmpeg",
            "-y",
            "-vsync",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            inp,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-spatial-aq",
            "1",
            "-aq-strength",
            "8",
            "-b:v",
            "5M",
            out,
        ]
    )
    assert os.path.getsize(out) > 0
