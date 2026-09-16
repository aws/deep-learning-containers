# Changelog

Changelog for the Amazon Linux 2023-based Ray LLM DLC images (`serve-llm-cuda`).

* * *

## Ray LLM v1.0 — 2026-08-31

**Tags:** `serve-llm-cuda` · `serve-llm-cuda-v1` · `serve-llm-cuda-v1.0` · `serve-llm-cuda-v1.0.0`

### Highlights

- Initial release of the Ray LLM DLC images on Amazon Linux 2023
- Ray Serve + vLLM for OpenAI-compatible LLM serving on GPU, via `ray[llm]`'s `build_openai_app`
- Ray 2.58.0 and vLLM 0.26.0 on PyTorch 2.11.0, CUDA 13.0.2, Python 3.13
- Single-GPU serving on {{ ec2_short }}; multi-node tensor-parallel serving on {{ eks_short }} via KubeRay
- EFA support for multi-node networking
- Shares the `ray` ECR repository with the Ray Serve (`serve-ml`) and Ray Train (`train-ml`) images
