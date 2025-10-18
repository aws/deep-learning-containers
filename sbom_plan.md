# SBOM Integration Prep

## Buildspec files we updated
- [x] `buildspec.yml` — PR and nightly builds install Trivy, export scan defaults, and add SBOM artifacts.
- [x] `release_buildspec.yml` — Release job mirrors Trivy install/env defaults and publishes `sbom/*.sbom.json`.
- [x] `bjs_release_buildspec.yml` — China release pipeline aligned with the same tooling.
- [x] `extended_release_buildspec.yml` — Staging builds now share the same scanner setup.
- [x] Verified additional buildspecs delegate to these paths; no extra edits needed.

## Build orchestration hook
- [x] `src/image.py` (`DockerImage.build`) runs `scripts/security/scan_image.sh` post-build unless `SKIP_VULN_SCAN=true`, covering all DockerImage consumers.
- [x] Confirmed `image_builder.py`, common/autopatch stages, and CLI helpers reuse `DockerImage.build`; no duplicate hooks required.

## Hook strategy (implemented)
- [x] Install Trivy during `install` phases of image-building CodeBuild jobs.
- [x] Centralized scan execution in `DockerImage.build` using the shared wrapper and `SKIP_VULN_SCAN` guard.
- [x] SBOMs emitted via the wrapper (default `spdx-json`) into `sbom/`.
- [x] Artifact configs include `sbom/*.sbom.json` for retrieval.
- [ ] Future enhancement: if more structured SBOM aggregation is required, consider recording paths in a manifest.

## Open questions / follow-ups
- [ ] Decide whether to retain SBOM artifacts long term or publish to a persistent store.
- [ ] Validate CodeBuild time/disk impact of Trivy database downloads; explore caching if needed.
