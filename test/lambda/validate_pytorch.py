#!/usr/bin/env python3
"""Validation tests for lambda-pytorch runtime image."""

import sys


def test_python_runtime():
    """Test Python runtime."""
    print(f"✓ Python {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print(f"  Prefix: {sys.prefix}")


def test_package_imports():
    """Test all required package imports."""
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("av", "av"),
        ("boto3", "boto3"),
    ]
    for module, name in packages:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {name}: {version}")

    # Test SAM2 imports
    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401

        print("✓ SAM2: available")
    except ImportError as e:
        print(f"✗ SAM2: {e}")


def test_pytorch_cuda():
    """Test PyTorch CUDA availability."""
    import torch

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        x = torch.randn(100, 100, device="cuda")
        y = torch.matmul(x, x.T)
        print(f"  GPU tensor ops: shape={y.shape}, sum={y.sum().item():.2f}")


def test_resnet_inference():
    """Test custom PyTorch model (ResNet50)."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import models, transforms

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        output = model(input_tensor)

    pred = output.argmax(dim=1).item()
    print(f"  Model: ResNet50, Output: {output.shape}, Prediction: class {pred}")


def test_diffusers_pipeline():
    """Test Stable Diffusion image generation."""
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        print("  Skipping: CUDA not available")
        return

    # Use tiny model for validation
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-torch", torch_dtype=torch.float16
    ).to("cuda")

    image = pipe("a cat", num_inference_steps=2, guidance_scale=1.0).images[0]
    print(f"  Model: Stable Diffusion (tiny), Output: {image.size}")


def test_whisper_stt():
    """Test Whisper speech-to-text."""
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline

    # Create dummy audio (1 second at 16kHz)
    audio = np.random.randn(16000).astype(np.float32) * 0.1

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, 16000)

        # Use tiny whisper model
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if torch.cuda.is_available() else -1,
        )

        result = pipe(f.name)
        print(f"  Model: Whisper tiny, Transcription length: {len(result['text'])} chars")


def test_transformers_pipeline():
    """Test transformers pipeline."""
    from transformers import pipeline

    # Test pipeline creation (use tiny model)
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if __import__("torch").cuda.is_available() else -1,
    )
    result = classifier("This is a test")
    print(f"  Pipeline: text-classification, Result: {result[0]['label']}")


def test_opencv_operations():
    """Test OpenCV operations."""
    import cv2
    import numpy as np

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    print(f"  Image: {img.shape}, Edges: {edges.sum()} pixels")


def test_audio_libraries():
    """Test audio processing libraries."""
    import librosa
    import numpy as np
    import soundfile

    # Create dummy audio
    sr = 22050
    duration = 1
    audio = np.random.randn(sr * duration).astype(np.float32)

    # Test librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    print(f"  librosa MFCC: {mfcc.shape}")

    # Test soundfile (write/read)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        soundfile.write(f.name, audio, sr)
        data, samplerate = soundfile.read(f.name)
        print(f"  soundfile I/O: {len(data)} samples @ {samplerate}Hz")


def test_video_io():
    """Test video I/O with av."""
    import tempfile

    import av
    import numpy as np

    # Create dummy video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        container = av.open(f.name, mode="w")
        stream = container.add_stream("h264", rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"

        for i in range(10):
            frame = av.VideoFrame(320, 240, "rgb24")
            frame.planes[0].update(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Read back
        container = av.open(f.name)
        frame_count = sum(1 for _ in container.decode(video=0))
        print(f"  av video I/O: {frame_count} frames")


def test_ffmpeg_available():
    """Test FFmpeg availability and NVENC/NVDEC support."""
    import subprocess

    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise Exception(f"FFmpeg not available: {result.stderr}")
    version_line = result.stdout.split("\n")[0]
    print(f"  {version_line}")

    # Verify NVENC/NVDEC encoders/decoders are compiled in
    result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
    if "h264_nvenc" not in result.stdout:
        raise Exception("h264_nvenc encoder not available — FFmpeg not built with NVENC support")
    print("  ✓ h264_nvenc encoder available")

    result = subprocess.run(["ffmpeg", "-decoders"], capture_output=True, text=True, timeout=5)
    if "h264_cuvid" not in result.stdout:
        raise Exception("h264_cuvid decoder not available — FFmpeg not built with NVDEC support")
    print("  ✓ h264_cuvid decoder available")


def test_ffmpeg_gpu_transcode():
    """Test FFmpeg GPU-accelerated transcode using NVENC/NVDEC.

    Covers NVIDIA Video Codec SDK FFmpeg guide:
    - §3.1: 1:1 HWACCEL transcode (GPU decode + GPU encode)
    - §3.2: HWACCEL transcode with scale_npp / scale_cuda GPU scaling
    - §4.2: Standalone NVDEC decode
    - §4.3: High-quality latency-tolerant preset (p6, tune hq, B-frames, temporal AQ, lookahead)
    - §4.4: Low-latency preset (p2, tune ll, CBR)
    - §5.1: Lookahead (-rc-lookahead)
    - §5.2: Spatial AQ (-spatial-aq 1) and Temporal AQ (-temporal-aq 1)
    """
    import os
    import subprocess
    import tempfile

    def run(args, timeout=60):
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=True)
        return result

    with tempfile.TemporaryDirectory() as tmpdir:

        def path(name):
            return os.path.join(tmpdir, name)

        # Generate a 2-second 1280x720 H.264 input (CPU) to feed the GPU decoder
        run(
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
                "libx264",
                "-preset",
                "ultrafast",
                path("input.mp4"),
            ]
        )

        # §3.1 — 1:1 HWACCEL transcode: GPU decode (cuvid) → GPU encode (nvenc)
        run(
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
                path("input.mp4"),
                "-c:v",
                "h264_nvenc",
                "-b:v",
                "5M",
                path("out_31.mp4"),
            ]
        )
        assert os.path.getsize(path("out_31.mp4")) > 0
        print("  ✓ §3.1 1:1 HWACCEL transcode (NVDEC → NVENC)")

        # §3.2 — HWACCEL transcode with scale_npp GPU scaling
        run(
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
                path("input.mp4"),
                "-vf",
                "scale_npp=640:360",
                "-c:v",
                "h264_nvenc",
                "-b:v",
                "2M",
                path("out_32_npp.mp4"),
            ]
        )
        assert os.path.getsize(path("out_32_npp.mp4")) > 0
        print("  ✓ §3.2 scale_npp GPU scaling (1280x720 → 640x360)")

        # §3.2 — HWACCEL transcode with scale_cuda GPU scaling
        run(
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
                path("input.mp4"),
                "-vf",
                "scale_cuda=640:360",
                "-c:v",
                "h264_nvenc",
                "-b:v",
                "2M",
                path("out_32_cuda.mp4"),
            ]
        )
        assert os.path.getsize(path("out_32_cuda.mp4")) > 0
        print("  ✓ §3.2 scale_cuda GPU scaling (1280x720 → 640x360)")

        # §4.2 — Standalone NVDEC decode to raw YUV
        run(
            [
                "ffmpeg",
                "-y",
                "-vsync",
                "0",
                "-c:v",
                "h264_cuvid",
                "-i",
                path("input.mp4"),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "nv12",
                path("out_42.yuv"),
            ]
        )
        assert os.path.getsize(path("out_42.yuv")) > 0
        print("  ✓ §4.2 standalone NVDEC decode (h264_cuvid → NV12)")

        # §4.3 — High-quality latency-tolerant preset
        # preset p6, tune hq, B-frames, temporal AQ, lookahead (§5.1), VBR
        run(
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
                path("input.mp4"),
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
                path("out_43_hq.mp4"),
            ]
        )
        assert os.path.getsize(path("out_43_hq.mp4")) > 0
        print("  ✓ §4.3 high-quality preset (p6, tune hq, B-frames, temporal AQ, lookahead)")

        # §4.4 — Low-latency preset: preset p2, tune ll, CBR, small VBV buffer
        run(
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
                path("input.mp4"),
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
                path("out_44_ll.mp4"),
            ]
        )
        assert os.path.getsize(path("out_44_ll.mp4")) > 0
        print("  ✓ §4.4 low-latency preset (p2, tune ll, CBR)")

        # §5.2 — Spatial AQ
        run(
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
                path("input.mp4"),
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
                path("out_52_saq.mp4"),
            ]
        )
        assert os.path.getsize(path("out_52_saq.mp4")) > 0
        print("  ✓ §5.2 spatial AQ (-spatial-aq 1 -aq-strength 8)")


def test_sam2_segmentation():
    """Test SAM2 image segmentation."""
    import os

    import numpy as np
    import torch

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Use tiny model for validation
        checkpoint = "/tmp/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

        # Download if not exists
        if not os.path.exists(checkpoint):
            import urllib.request

            print("  Downloading SAM2 tiny checkpoint...")
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
            urllib.request.urlretrieve(url, checkpoint)

        # Build predictor
        sam2_model = build_sam2(
            model_cfg, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        predictor = SAM2ImagePredictor(sam2_model)

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run inference
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
        )

        print(f"  Model: SAM2 tiny, Masks: {masks.shape}, Scores: {scores.shape}")

    except Exception as e:
        raise Exception(f"SAM2 validation failed: {e}")
    """Test video I/O with av."""
    import tempfile

    import av
    import numpy as np

    # Create dummy video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        container = av.open(f.name, mode="w")
        stream = container.add_stream("h264", rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"

        for i in range(10):
            frame = av.VideoFrame(320, 240, "rgb24")
            frame.planes[0].update(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Read back
        container = av.open(f.name)
        frame_count = sum(1 for _ in container.decode(video=0))
        print(f"  av video I/O: {frame_count} frames")


def test_environment():
    """Test environment variables."""
    import os

    path = os.environ.get("PATH", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    py_path = os.environ.get("PYTHONPATH", "")
    print(f"  PATH: {path[:60]}...")
    print(f"  LD_LIBRARY_PATH: {ld_path[:60]}...")
    print(f"  PYTHONPATH: {py_path[:60]}...")

    assert os.environ.get("LAMBDA_TASK_ROOT") == "/var/task"
    assert os.environ.get("LAMBDA_RUNTIME_DIR") == "/var/runtime"
    assert os.environ.get("LANG") == "en_US.UTF-8"
    assert os.environ.get("TZ") == ":/etc/localtime"

    for p in [
        "/var/lang/lib",
        "/lib64",
        "/usr/lib64",
        "/var/runtime",
        "/var/runtime/lib",
        "/var/task",
        "/var/task/lib",
        "/opt/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/lib",  # FFmpeg shared libs
        "/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia",
    ]:
        assert p in ld_path, f"LD_LIBRARY_PATH missing {p}"
    print(f"  LD_LIBRARY_PATH entries: {len(ld_path.split(':'))}")
    print("✓ Environment configured")


def main():
    """Run all validation tests."""
    tests = [
        ("Python Runtime", test_python_runtime),
        ("Package Imports", test_package_imports),
        ("PyTorch CUDA", test_pytorch_cuda),
        ("ResNet Inference", test_resnet_inference),
        ("SAM2 Segmentation", test_sam2_segmentation),
        ("Stable Diffusion", test_diffusers_pipeline),
        ("Whisper STT", test_whisper_stt),
        ("Transformers Pipeline", test_transformers_pipeline),
        ("OpenCV Operations", test_opencv_operations),
        ("FFmpeg Available", test_ffmpeg_available),
        ("FFmpeg GPU Transcode", test_ffmpeg_gpu_transcode),
        ("Audio Libraries", test_audio_libraries),
        ("Video I/O", test_video_io),
        ("Environment", test_environment),
    ]

    print("=" * 70)
    print("lambda-pytorch Runtime Validation")
    print("=" * 70)
    print()

    failed = []
    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            test_fn()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            failed.append(name)
        print()

    print("=" * 70)
    if failed:
        print(f"✗ {len(failed)}/{len(tests)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"✓ All {len(tests)} validations passed")


if __name__ == "__main__":
    main()
