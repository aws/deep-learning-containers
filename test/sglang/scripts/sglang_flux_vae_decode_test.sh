#!/bin/bash
set -euo pipefail

# SGLang FLUX.2 VAE decode test
#
# FLUX.2-small-decoder is a distilled VAE decoder (class AutoencoderKLFlux2), a
# drop-in decoder for the FLUX.2 diffusion pipeline. It is a pipeline COMPONENT,
# not a servable text-to-image model, so this validates it the way it is used:
# load the decoder and decode a latent into an image tensor.
#
# Usage: sglang_flux_vae_decode_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"

echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Loading AutoencoderKLFlux2 and running a decode ==="
python3 - "${MODEL_DIR}" "${MODEL_NAME}" <<'PY'
import sys

import torch
from diffusers import AutoencoderKLFlux2

model_dir, model_name = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKLFlux2.from_pretrained(model_dir).to(device).eval()
n_params = sum(p.numel() for p in vae.parameters())
print(f"Loaded {type(vae).__name__} params={n_params} device={device}")

# Build a latent that matches the decoder's expected channel count, then decode.
latent_channels = vae.config.latent_channels
latent = torch.randn(1, latent_channels, 16, 16, device=device, dtype=next(vae.parameters()).dtype)
with torch.no_grad():
    image = vae.decode(latent).sample

assert image.ndim == 4 and image.shape[0] == 1, f"unexpected decode output shape {tuple(image.shape)}"
assert torch.isfinite(image).all(), "decode produced non-finite values"
print(f"Decoded latent {tuple(latent.shape)} -> image {tuple(image.shape)}")
print(f"=== PASSED: {model_name} ===")
PY
