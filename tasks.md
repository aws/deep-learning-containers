# Implementation Plan: Auto Upstream Release Tracker

## Overview

Implement a GitHub Actions workflow that automatically detects new upstream framework releases (vLLM, SGLang) and creates PRs to update version configurations. The implementation uses GitHub Actions YAML workflows, bash scripts for version logic, and `yq` for YAML manipulation. All utility functions live in a shared bash script sourced by the workflow.

## Tasks

- [ ] 1. Create the framework tracker registry config
  - [ ] 1.1 Create `.github/config/tracker.yml` with vLLM and SGLang entries
    - Define the `frameworks` top-level mapping
    - Add top-level `notifications` section with `slack.enabled: true` (webhook URL stored as GitHub Actions secret `SLACK_WEBHOOK_URL`, not in this file)
    - Add `vllm` entry with `github_repo: "vllm-project/vllm"`, `tag_prefix: "v"`, and `config_files` for vllm-ec2 and vllm-sagemaker with their `prod_image_template` values
    - Add `sglang` entry with `github_repo: "sgl-project/sglang"`, `tag_prefix: "v"`, and `config_files` for sglang-ec2 and sglang-sagemaker
    - Add `exclude_configs` for vllm-rayserve (independent lifecycle)
    - Add `reviewers` for each framework
    - Add `dockerfiles` for vllm (`docker/vllm/Dockerfile` with `base_image_template: "vllm/vllm-openai:v{version}"`) and sglang (`docker/sglang/Dockerfile` with `base_image_template: "lmsysorg/sglang:v{version}-cu129-amd64"`)
    - Add `docker_hub_image` for vllm (`"vllm/vllm-openai"`) and sglang (`"lmsysorg/sglang"`)
    - Add `test_setup_script` for vllm with `pattern: "scripts/vllm/vllm_{version_underscored}_test_setup.sh"` and `workflow_files` pointing to `auto-release-vllm-sagemaker.yml` and `auto-release-vllm-ec2.yml`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 13.1, 13.4, 14.1, 15.1, 15.4, 16.3, 16.5_

- [ ] 2. Implement version comparison and utility functions
  - [ ] 2.1 Create `scripts/tracker-utils.sh` with core bash functions
    - Implement `strip_tag_prefix(tag, prefix)` — strips configured prefix from release tag, defaults to `"v"`
    - Implement `is_rc_version(version)` — returns true if version string contains RC suffix (case-insensitive match for `rc\d+`, `-rc.\d+`); e.g., `"0.17.0rc1"` → true, `"0.17.0"` → false
    - Implement `is_newer_version(upstream, current)` — numeric segment-by-segment semver comparison, pads to 3 segments; caller must ensure RC versions are filtered before calling
    - Implement `get_current_version(config_file)` — reads `common.framework_version` from a config YAML using `yq`
    - Implement `render_prod_image(template, version)` — substitutes `{major}`, `{minor}`, `{patch}` placeholders in prod_image_template
    - All functions should `set -euo pipefail` and validate inputs
    - _Requirements: 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 5.2, 12.1, 12.2_

  - [ ]* 2.2 Write unit tests for `is_newer_version()`
    - **Property 1: Version comparison correctness**
    - **Validates: Requirements 3.1, 3.2, 3.3**
    - Test cases: equal versions return false, major bump detected, minor bump detected, patch bump detected
    - Test edge cases: `"0.9.0"` vs `"0.16.0"` (numeric not lexicographic), `"0.16"` vs `"0.16.0"` (padding)
    - Test that older upstream returns false

  - [ ]* 2.3 Write unit tests for `strip_tag_prefix()`
    - **Property 2: Tag prefix stripping**
    - **Validates: Requirements 2.3, 2.4**
    - Test stripping `"v"` from `"v0.17.0"` → `"0.17.0"`
    - Test default prefix when none configured
    - Test tag that doesn't match prefix (passthrough)

  - [ ]* 2.4 Write unit tests for `render_prod_image()`
    - Test template rendering with `{major}` and `{minor}` placeholders
    - Test with version `"0.17.0"` and template `"vllm:{major}.{minor}-gpu-py312-ec2"` → `"vllm:0.17-gpu-py312-ec2"`
    - _Requirements: 5.2_

  - [ ]* 2.5 Write unit tests for `is_rc_version()`
    - **Property 3b: RC tag filtering**
    - **Validates: Requirements 12.1, 12.2**
    - Test that `"0.17.0rc1"` returns true
    - Test that `"0.17.0RC2"` returns true (case-insensitive)
    - Test that `"0.17.0-rc.1"` returns true
    - Test that `"0.17.0"` returns false
    - Test that `"0.17.0.post1"` returns false (not an RC)
    - Test that `"1.0.0rc0"` returns true

- [ ] 3. Checkpoint - Ensure utility function tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement config file update logic
  - [ ] 4.1 Create `scripts/update-configs.sh` with config update functions
    - Implement `update_config_files(framework_key, new_version, config_entries_json)` — iterates over config file entries, updates `common.framework_version` and `common.prod_image` using `yq`
    - Source `tracker-utils.sh` for `render_prod_image` and helper functions
    - Validate that each config file exists before updating
    - Ensure `yq` preserves all non-target fields (use `yq eval -i` for in-place updates)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 4.2 Write tests for config file update logic
    - **Property 4: Config update correctness**
    - **Property 5: Config integrity (non-target fields preserved)**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    - Create a sample config YAML, run update, verify `framework_version` and `prod_image` changed
    - Verify all other fields remain identical after update
    - Test with multiple config files to verify all are updated

  - [ ]* 4.3 Write test for framework isolation
    - **Property 6: Framework isolation**
    - **Validates: Requirements 5.5, 7.1, 7.2**
    - Create config files for two frameworks, update one, verify the other is untouched
    - Verify files in `exclude_configs` are not modified

- [ ] 4b. Implement Dockerfile update logic
  - [ ] 4b.1 Add `update_dockerfiles()` function to `scripts/update-configs.sh`
    - Implement `update_dockerfiles(new_version, dockerfile_entries_json)` — iterates over Dockerfile entries, renders `base_image_template` with `{version}` replaced, uses `sed` to update `ARG BASE_IMAGE=` line
    - Only modify the `ARG BASE_IMAGE=` default value line; preserve all other Dockerfile content
    - Validate that each Dockerfile exists before updating
    - _Requirements: 13.1, 13.2, 13.3_

  - [ ]* 4b.2 Write tests for Dockerfile update logic
    - **Property 13: Dockerfile BASE_IMAGE update correctness**
    - **Validates: Requirements 13.1, 13.2, 13.3**
    - Create a sample Dockerfile with `ARG BASE_IMAGE=vllm/vllm-openai:v0.16.0`, run update with version `"0.17.0"`, verify `ARG BASE_IMAGE=vllm/vllm-openai:v0.17.0`
    - Verify all other Dockerfile lines remain unchanged (FROM, RUN, COPY, etc.)
    - Test with multiple Dockerfiles to verify all are updated

- [ ] 4c. Implement CUDA/Python version detection logic
  - [ ] 4c.1 Add `detect_cuda_python_changes()` function to `scripts/update-configs.sh`
    - Implement `detect_cuda_python_changes(docker_hub_image, new_version, config_entries_json)` — queries Docker Hub API for image tag metadata, extracts CUDA/Python versions from labels or tag string
    - If CUDA version differs from `common.cuda_version`, update all config files using `yq`
    - If Python version differs from `common.python_version`, update all config files using `yq`
    - If Docker Hub API fails or metadata is missing, return a warning string (do not fail)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ]* 4c.2 Write tests for CUDA/Python version detection
    - **Property 14: CUDA/Python version detection graceful degradation**
    - **Validates: Requirements 14.2, 14.3, 14.4, 14.5**
    - Mock Docker Hub API response with changed CUDA version, verify config files updated
    - Mock Docker Hub API failure, verify warning string returned and no config changes
    - Test that detection failure does not prevent the function from returning successfully

- [ ] 4d. Implement test setup script renaming logic
  - [ ] 4d.1 Add `rename_test_setup_script()` function to `scripts/update-configs.sh`
    - Implement `rename_test_setup_script(old_version, new_version, test_setup_config_json)` — converts versions to underscored format, resolves old/new script paths from pattern, uses `git mv` to rename, uses `sed` to update workflow file references
    - Validate that the old script file exists before renaming
    - Validate that each workflow file exists before updating references
    - _Requirements: 15.1, 15.2, 15.3_

  - [ ]* 4d.2 Write tests for test setup script renaming
    - **Property 15: Test setup script rename correctness**
    - **Validates: Requirements 15.1, 15.2, 15.3**
    - Create a mock script file and workflow files, run rename from `"0.16.0"` to `"0.17.0"`, verify old file removed and new file exists
    - Verify workflow files reference the new script path
    - Test version-to-underscore conversion: `"0.16.0"` → `"0_16_0"`, `"0.17.0"` → `"0_17_0"`

- [ ] 5. Implement the main GitHub Actions workflow
  - [ ] 5.1 Create `.github/workflows/check-upstream-releases.yml` workflow skeleton
    - Define `on.schedule` with `cron: '0 * * * *'` (every 60 minutes)
    - Define `on.workflow_dispatch` with `framework` (string, optional) and `dry-run` (boolean, default false) inputs
    - Set `permissions: contents: write, pull-requests: write`
    - Define single job `check-releases` running on `ubuntu-latest`
    - Checkout repository, install `yq`
    - _Requirements: 1.1, 1.3, 1.4, 8.1, 8.2_

  - [ ] 5.2 Implement tracker registry parsing step
    - Read `.github/config/tracker.yml` using `yq`
    - Extract framework keys into a list
    - If `framework` input is provided, filter to only that framework
    - If filtered framework doesn't exist in registry, fail with descriptive error
    - _Requirements: 1.2, 1.3, 1.4, 9.1, 9.3_

  - [ ] 5.3 Implement upstream version check loop
    - For each framework: call GitHub Releases API `GET /repos/{owner}/{repo}/releases/latest`
    - Check `prerelease` and `draft` fields — skip if either is true
    - Strip tag prefix using `strip_tag_prefix()`
    - Check for RC suffix using `is_rc_version()` — skip if version contains RC suffix (e.g., `rc1`, `rc2`)
    - Read current version from first config file using `get_current_version()`
    - Compare using `is_newer_version()` — skip if not newer
    - Log results for each framework
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 12.1, 12.2, 12.3_

  - [ ] 5.4 Implement branch existence check and idempotency guard
    - Check if branch `auto-update/{framework}-{version}` exists using `git ls-remote`
    - If branch exists, log and skip PR creation
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 5.5 Implement config update and PR creation steps
    - Source `scripts/update-configs.sh` to update config files
    - Call `update_dockerfiles()` if framework has `dockerfiles` configured
    - Call `detect_cuda_python_changes()` if framework has `docker_hub_image` configured; capture warning string if detection fails
    - Call `rename_test_setup_script()` if framework has `test_setup_script` configured
    - Create branch `auto-update/{framework}-{version}` from `main`
    - Commit all updated files (configs, Dockerfiles, renamed scripts, workflow files) with message `[Auto-Update] {framework} {version}`
    - Push branch and create PR using `gh pr create` with populated description template
    - Include CUDA/Python detection warning in PR description if applicable
    - Assign reviewers from framework entry if specified
    - Include upstream release link, version diff, changed files list, and release notes summary in PR body
    - After PR creation, trigger Slack notification step (task 5.8)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 11.1, 11.2, 13.1, 13.2, 14.1, 14.4, 15.1, 15.2_

  - [ ] 5.6 Implement dry-run mode
    - When `dry-run` is true, log detected version differences but skip branch creation, commits, and PR creation
    - Ensure no side effects in dry-run mode
    - _Requirements: 8.1, 8.2_

  - [ ] 5.7 Implement error handling and failure isolation
    - Wrap each framework's processing in error handling (use `|| true` pattern or `set +e` per framework)
    - On API 404: log warning, continue to next framework
    - On API 403 (rate limit): log error with rate-limit headers, mark framework as failed
    - On config update failure: log error, continue to next framework
    - On permission error: fail with descriptive message
    - Track per-framework success/failure and output summary at end
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 5.8 Implement Slack webhook notification step
    - Add `send_slack_notification()` function to `scripts/tracker-utils.sh`
    - Read `notifications.slack.enabled` from tracker.yml using `yq`
    - If enabled and `SLACK_WEBHOOK_URL` secret is available, construct a simple JSON payload with key-value pairs: `framework_name`, `framework_version`, `pr_url`, `release_notes_url`, and `changed_files` (comma-separated string)
    - POST the payload to the Slack Workflow webhook URL using `curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json"`
    - The Slack Workflow on the receiving end handles message formatting and channel routing — our workflow only provides raw data
    - If HTTP 200, log success; otherwise log warning with status code
    - If `SLACK_WEBHOOK_URL` secret is missing or empty, log info and skip (do not warn)
    - Handle failure gracefully: log warning, do NOT fail the workflow — the PR was already created successfully
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

  - [ ]* 5.9 Write tests for Slack notification
    - **Property 16: Slack notification failure isolation**
    - **Validates: Requirements 16.1, 16.3, 16.4**
    - Test that `send_slack_notification()` constructs correct key-value JSON payload with all required fields (`framework_name`, `framework_version`, `pr_url`, `release_notes_url`, `changed_files`)
    - Test that the payload does NOT contain any Block Kit or rich formatting — only simple key-value pairs
    - Test that HTTP non-200 response returns false and logs warning without failing
    - Test that missing/empty webhook URL returns false and logs info without warning
    - Test that network timeout returns false and logs warning without failing

- [ ] 6. Checkpoint - Verify workflow syntax and dry-run
  - Ensure all tests pass, ask the user if questions arise.
  - Validate workflow YAML syntax with `actionlint` or manual review
  - Recommend user test with: `gh workflow run check-upstream-releases.yml -f dry-run=true`

- [ ] 7. Integration testing and validation
  - [ ]* 7.1 Write integration test script for dry-run mode
    - **Property 10: Dry-run produces no side effects**
    - **Property 9: Framework filter correctness**
    - **Validates: Requirements 1.3, 1.4, 8.2**
    - Create a test script that sets up mock config files and runs the tracker logic in dry-run mode
    - Verify no branches or PRs are created
    - Verify framework filter processes only the specified framework

  - [ ]* 7.2 Write validation test for tracker registry schema
    - **Property 11: Registry validation**
    - **Validates: Requirement 9.3**
    - Test that missing `github_repo` produces a validation error
    - Test that empty `config_files` list produces a validation error
    - Test that valid entries parse successfully

  - [ ]* 7.3 Write test for idempotency
    - **Property 7: Idempotency**
    - **Validates: Requirements 4.2, 4.3**
    - Simulate branch already existing, verify workflow skips PR creation

- [ ] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify tracker.yml is valid and complete
  - Verify workflow file has correct permissions and cron schedule
  - Confirm all requirements are covered by implementation tasks

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Implementation language is GitHub Actions YAML + Bash (determined by the design)
- `yq` is already available in the repo's CI environment (used by `load-config` action)
- `gh` CLI is pre-installed on GitHub Actions `ubuntu-latest` runners
- The 2-week scope prioritizes: registry config → utility functions → workflow → testing
- Property tests validate universal correctness properties from the design document
- Each task references specific requirements for traceability
- Tasks 4b, 4c, 4d extend the update scope beyond config files to Dockerfiles, CUDA/Python detection, and test script renaming
- CUDA/Python detection (4c) is best-effort — failure degrades gracefully to a PR warning
- Test script renaming (4d) only applies to frameworks with `test_setup_script` in tracker.yml (currently vLLM only)
- Slack notification (5.8) is best-effort — failure does not block PR creation or fail the workflow
- The Slack webhook URL is a secret (not committed to the repo); the receiving Slack Workflow handles message formatting and channel routing — our workflow only sends key-value pairs
