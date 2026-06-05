"""Validate HuggingFace model pipelines on a single GPU — pytorch image."""

import numpy as np
import torch


def test_text_classification():
    """Transformers text-classification pipeline."""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0,
    )
    result = classifier("This is a test")
    assert result[0]["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result[0]["score"] <= 1.0


def test_whisper_stt():
    """Whisper speech-to-text pipeline."""
    import tempfile

    import soundfile as sf
    from transformers import pipeline

    audio = np.random.RandomState(42).randn(16000).astype(np.float32) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, 16000)
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)
        result = pipe(f.name)
        assert isinstance(result["text"], str)


def test_stable_diffusion():
    """Stable Diffusion tiny model generates an image."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-torch", torch_dtype=torch.float16
    ).to("cuda")
    image = pipe("a cat", num_inference_steps=2, guidance_scale=1.0).images[0]
    assert image.size[0] > 0 and image.size[1] > 0


def test_sam2_segmentation():
    """SAM2 image segmentation produces masks."""
    import os
    import urllib.request

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "/tmp/sam2_hiera_tiny.pt"
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            checkpoint,
        )

    model = build_sam2("sam2_hiera_t.yaml", checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(model)

    img = np.random.RandomState(42).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[320, 240]]),
        point_labels=np.array([1]),
    )
    assert masks.shape[0] > 0
    assert scores.shape[0] > 0
