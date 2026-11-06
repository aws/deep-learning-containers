# NVIDIA Dynamo DLC

`Dockerfile.amzn2023` builds the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) runtime with one
inference backend, with or without EFA. Image configs: `.github/config/image/dynamo/`.

| Path                       | What                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `Dockerfile.amzn2023`      | Shared base and EFA stages, Dynamo built from source, one target per backend (vllm, sglang, trtllm) |
| `<backend>/pyproject.toml` | The backend's Python environment: `ai-dynamo[<backend>]==<version>`, index and exclusions           |
| `<backend>/uv.lock`        | Every package pinned; the image installs it with `uv sync --frozen`                                 |

## How dependencies are controlled

Each Dynamo release pins the exact backend version it supports (for example `ai-dynamo[vllm]==1.6.1`
requires `vllm==0.31.0` and `nixl[cu13]==1.4.1`). uv resolves that into `uv.lock`, so the image gets
one consistent, reproducible environment:

- torch, torchvision and torchaudio come from the PyTorch CUDA 13.0 index (`tool.uv.sources`).
- `exclude-dependencies` drops what the image doesn't need (`aisimulate`, and the CUDA 12 libraries
  that the `nixl` / `nvidia-cutlass-dsl` meta-packages always pull in).
- `override-dependencies` lifts an upstream cap only for a security fix, with the reason next to it
  (SGLang: `pillow>=13.0.0` over moviepy's `pillow<12`).
- `ai-dynamo`, `ai-dynamo-runtime` and `gpu-memory-service` are built from source at
  `build.dynamo_ref` and installed over the locked versions. The build fails if their version
  differs from the lock, and `uv pip check` must pass apart from the declared exclusions and
  overrides (`scripts/docker/dynamo/check_environment.py`).
- CI (`dynamo.pr.yml`) runs `uv lock --check`, and `test/dynamo/test_image.py` checks that the image
  matches the lock exactly.

## Backend notes

- **TensorRT-LLM** (`trtllm`): the wheel comes from NVIDIA's index (`pypi.nvidia.com`; PyPI only has a placeholder
  sdist). It bundles UCX and NIXL (`tensorrt_llm/libs/nixl`, without a LIBFABRIC plugin), so the EFA flavor builds
  that plugin for the bundled NIXL version (`build.bundled_nixl_version`, from TensorRT-LLM's
  `docker/common/install_nixl.sh`). It imports `mpi4py`: the EFA flavor uses the EFA installer's Open MPI, the
  non-EFA flavor installs AL2023's. Over EFA, set `cache_transceiver_config.backend: NIXL` and
  `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC`. x86_64 only (the aarch64 wheel needs glibc 2.39).
- **KVBM** is not included for any backend (deprecated upstream; its wheel pins a NIXL that conflicts with the locks).

## Build locally

```bash
docker build -f docker/dynamo/Dockerfile.amzn2023 --target vllm \
  --build-arg BACKEND=vllm --build-arg EFA_FLAVOR=efa \
  --build-arg DYNAMO_REF=<commit> -t dynamo-vllm .
```

The target and `BACKEND` must match. CI passes every `build:` key of the config as a build arg.

## Update Dynamo (or a backend's dependencies)

1. Set the new `ai-dynamo[<backend>]==<version>` in each `<backend>/pyproject.toml`.
1. Re-lock: `uv lock --directory docker/dynamo/<backend>` (uv >= 0.13, same version as
   `UV_VERSION` in the Dockerfile). To refresh other packages within their constraints, use
   `uv lock --upgrade` (or `--upgrade-package <name>`).
1. In each config, update `metadata.framework_version` (Dynamo), `metadata.backend_version` (from the
   lock), `metadata.prod_image` and `build.dynamo_ref` (the release's commit).
1. Recheck each `override-dependencies` entry (drop it once upstream allows the fixed version) and
   the image-scan allowlist `test/security/data/ecr_scan_allowlist/dynamo/dynamo-<version>.json`
   (CVEs that a backend's exact pins block; start a new file for the new version).
1. If the release moves NIXL to a version that needs a LIBFABRIC fix, add
   `scripts/docker/dynamo/nixl/patches/<nixl-version>/`; drop it once a NIXL release includes the fix.

## Add a backend

Add `<backend>/pyproject.toml` (copy an existing one and change the extra), lock it, add a
`FROM dynamo AS <backend>` target, and add configs.
