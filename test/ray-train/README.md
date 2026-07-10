# RayTrain DLC tests

Current coverage (basic):

- **Sanity** — `test/sanity/scripts/test_sanity_ray_train.py`, wired into
  `_reusable.sanity-tests.yml` on `framework == ray && job_type == training`.
  Its own script (like the Ray Serve image keeps its own suite) because RayTrain
  diverges from the PyTorch/TF training contract in two ways — it uses EFA's
  bundled OpenMPI (no from-source double-wrap) and a passive/KubeRay entrypoint
  (no `/usr/local/bin/entrypoint.sh`). Covers the shared training-cluster contract
  it *does* honor (env, PATH, EFA/NCCL, CUDA, SSH, venv, OSS, nccl-tests binary)
  plus Ray specifics (Ray version, `ray[train,tune,data]` imports, CUDA torch
  build, Ray Serve extra absent). Runs on a CPU sanity runner.
- **Security** — shared Ray ECR scan allowlist at
  `test/security/data/ecr_scan_allowlist/ray/` (RayTrain uses `framework: ray`, so it
  shares the allowlist with the Ray Serve DLC).
- **Telemetry** — cross-framework `_reusable.telemetry-tests.yml` (no per-framework code).

Deferred (tracked in `Design/raytrain-dlc-design.md`):

- **Multi-node EFA / NCCL** cross-node all-reduce (needs ≥2 EFA-capable GPU nodes).
- **KubeRay on EKS** — stand up a `RayCluster` (head + GPU workers) on a plain EKS
  managed node group, `ray job submit` a small FSDP/Lightning job, assert convergence.
- **Single-node multi-GPU** Ray Train smoke (real intra-node NCCL + FSx mount).
