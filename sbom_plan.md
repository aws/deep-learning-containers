# SBOM Integration Prep

## Buildspec files we likely need to touch
- [ ] `buildspec.yml` — PR and nightly builds invoke `python src/main.py`; add scanner install in `pre_build`, pipe SBOM artifacts into secondary outputs.
- [ ] `release_buildspec.yml` — Release job runs `publish_dlc_images`; inject scanner install before build phase and ensure artifacts section (if any) publishes SBOMs.
- [ ] `bjs_release_buildspec.yml` — Mirrors release workflow for China region; mirror tool install + SBOM handling.
- [ ] `extended_release_buildspec.yml` — Verify whether `stage_extended_release` builds images or only stages metadata; if it builds, add scanner install + SBOM export.
- [ ] (Confirm) Any other CodeBuild spec that actually builds and pushes images; current grep shows no additional builder specs, but re-check once we locate `publish_dlc_images*` implementations.

## Build orchestration code to inspect/hook
- [ ] `src/image.py` (`DockerImage.build` / `docker_build`) — direct place to invoke scanner right after a successful build, before size checks.
- [ ] `src/image_builder.py` — orchestrates image objects; good place if we need to collect SBOM paths or bubble up failures.
- [ ] `patch_helper.py` / `common_stage_image.py` — ensure pre-push images (autopatch/common stage) also trigger scanning if they call the shared build routine.
- [ ] Locate implementations for CLI helpers (`publish_dlc_images`, `publish_dlc_images_to_bjs`, `stage_extended_release`) to confirm they reuse `DockerImage.build`; if not, introduce a shared scanner utility they call.

## Recommended hook strategy (minimal change)
- [ ] Install Trivy (or chosen scanner) during `install`/`pre_build` phase of every CodeBuild job that builds images; cache DB in `/tmp/trivy` to avoid repeated downloads.
- [ ] Extend `DockerImage.build` to run `trivy image --severity CRITICAL --exit-code 1 --ignore-unfixed` (configurable) once `docker_build` succeeds; capture stdout for logs.
- [ ] Generate SBOM via `trivy sbom -f spdx-json` (or CycloneDX) into a deterministic path (e.g., `$CODEBUILD_SRC_DIR/sbom/<repo>-<tag>.json`) and track filenames in Python for artifact upload.
- [ ] Surface SBOM paths via a lightweight registry (e.g., append to `constants.TEST_TYPE_IMAGES_PATH` or a new JSON) so buildspec artifact section can include `sbom/*.json`.
- [ ] Update CodeBuild artifact configuration (or add new secondary artifact) to upload all generated SBOM files.
- [ ] Document severity policy and any future allowlist expectations (e.g., fail on CRITICAL, warn on HIGH) so maintainers can tune thresholds.

## Open questions / follow-ups
- [ ] Confirm where `publish_dlc_images*` commands are defined; ensure they go through the same build path or plan separate hooks.
- [ ] Decide whether to emit SBOMs for intermediate/pre-push images or only final tags.
- [ ] Validate that CodeBuild environments have enough disk/time budget for scanner DB downloads; consider offline cache if needed.
