# RayTrain DLC tests

Current coverage (basic):

- **Sanity** — reuses the shared training sanity test
  `test/sanity/scripts/test_sanity_training.py` (RayTrain is added to
  `training_cluster_only`, like the other training frameworks — no per-framework
  file, matching repo convention). It verifies the shared training contract
  (env, PATH, EFA/NCCL, CUDA, cuDNN, SSH, venv, OSS) plus a RayTrain-only
  `TestRayTrain` class (Ray version, `ray[train,tune,data]` importable, CUDA torch
  build, Ray Serve extra absent). Wired in `_reusable.sanity-tests.yml` on
  `framework == ray && job_type == training`. Runs on a CPU sanity runner.
- **Security** — shared Ray ECR scan allowlist at
  `test/security/data/ecr_scan_allowlist/ray/` (RayTrain uses `framework: ray`, so it
  shares the allowlist with the Ray Serve DLC).
- **Telemetry** — cross-framework `_reusable.telemetry-tests.yml` (no per-framework code).

Deferred (tracked in `Design/raytrain-dlc-design.md`):

- **Multi-node EFA / NCCL** cross-node all-reduce (needs ≥2 EFA-capable GPU nodes).
- **KubeRay on EKS** — stand up a `RayCluster` (head + GPU workers) on a plain EKS
  managed node group, `ray job submit` a small FSDP/Lightning job, assert convergence.
- **Single-node multi-GPU** Ray Train smoke (real intra-node NCCL + FSx mount).
