# Changelog

Changelog for the Amazon Linux 2023-based NVIDIA Dynamo images.

* * *

## v1.0.1 — 2026-11-13

**Dynamo:** 1.6.2

**Tags:** `1.6.2-<flavor>-runtime-v1.0.1` for each of the six images

- Weekly security patches. No change to Dynamo or the backends.

* * *

## v1.0.0 — 2026-11-12

**Dynamo:** 1.6.2

**Tags:** `1.6.2-<flavor>-runtime` · `-v1` · `-v1.0` · `-v1.0.0`, where `<flavor>` is `vllm`, `efa-vllm`, `sglang`, `efa-sglang`, `trtllm`, or
`efa-trtllm`

**Dynamo source:** [v1.6.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.2)

### Highlights

- Initial release of the NVIDIA Dynamo containers on Amazon Linux 2023, for disaggregated LLM serving on {{ hyperpod }} ({{ eks_short }}).
- Six images: vLLM 0.31.0, SGLang 0.5.21, and TensorRT-LLM 1.3.0rc29, each with and without EFA.
- Dynamo 1.6.2 built from source; each backend's Python environment is fully locked to the versions Dynamo 1.6.2 supports.
- KV cache transfer through NIXL over EFA (LIBFABRIC) in the EFA images; without EFA, NIXL over UCX for vLLM and SGLang, and TensorRT-LLM's own cache
  transceiver.
- CUDA 13.0, Python 3.12.
