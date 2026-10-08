"""Validate PyTorch CUDA and inference on a single GPU — pytorch image."""

import numpy as np
import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device_count():
    assert torch.cuda.device_count() >= 1


def test_gpu_tensor_matmul():
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    assert c.shape == (256, 256)
    assert torch.isfinite(c).all()


def test_resnet50_inference():
    """ResNet50 forward pass produces 1000-class logits."""
    from torchvision import models, transforms

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda().eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from PIL import Image

    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))
    x = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1000)
    assert torch.isfinite(out).all()


def test_opencv_operations():
    """OpenCV image processing pipeline."""
    import cv2

    img = np.random.RandomState(42).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    assert edges.shape == (480, 640)
    assert edges.sum() > 0


def test_audio_roundtrip():
    """librosa + soundfile write/read roundtrip."""
    import tempfile

    import librosa
    import soundfile

    sr = 22050
    audio = np.random.RandomState(42).randn(sr).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    assert mfcc.shape[0] == 13

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        soundfile.write(f.name, audio, sr)
        data, rate = soundfile.read(f.name)
        assert rate == sr
        assert len(data) == sr


def test_video_io_roundtrip():
    """av video write/read roundtrip."""
    import tempfile

    import av

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        container = av.open(f.name, mode="w")
        stream = container.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame(320, 240, "rgb24")
            frame.planes[0].update(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        container = av.open(f.name)
        frames = sum(1 for _ in container.decode(video=0))
        assert frames == 10
