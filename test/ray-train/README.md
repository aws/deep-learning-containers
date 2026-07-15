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

Automated CI (in this PR):

- **Multi-node EFA / NCCL** — `_reusable.efa-tests.yml` (2x EFA GPU instances,
  `all_reduce_perf` across nodes).

Validated manually on HyperPod-EKS (2026-07-11..14, see
`Design/raytrain-dlc-requirements-from-hyperpod-guide.md`) — the single GPU image,
run via KubeRay as both head and worker:

- Multi-node DDP (ResNet-18/50) + multi-node FSDP (BERT/GLUE) across 2x g6.12xlarge,
  world_size=8, real 2-level NCCL (intra-node SHM + inter-node EFA).
- Checkpointing to FSx; fault tolerance (kill a worker → Ray recovers + resumes
  from checkpoint); production CPU-head + GPU-worker topology.

Deferred / still to automate:

- **KubeRay-on-EKS functional test in CI** — stand up a `RayCluster`, `ray job submit` a small job, assert convergence. Needs the KubeRay operator on the shared
  EKS cluster (not yet installed). The single image already covers head + worker.
- `ray.tune` HP tuning; LoRA/PEFT (ties to peft/trl open question).
