# Requirements Document

## Introduction

The Auto Upstream Release Tracker automates the detection of new upstream framework releases (starting with vLLM and SGLang) and creates pull requests to update version configurations in the `deep-learning-containers` repository. This eliminates the manual process of monitoring GitHub repositories, updating config files, and creating PRs — reducing the time between an upstream release and customer availability on AWS infrastructure.

## Glossary

- **Tracker_Workflow**: The GitHub Actions workflow (`check-upstream-releases.yml`) that runs on a cron schedule to detect new upstream releases and create update PRs.
- **Tracker_Registry**: The central configuration file (`.github/config/tracker.yml`) that defines which upstream frameworks to monitor, their GitHub coordinates, and which config files to update.
- **Config_Updater**: The component responsible for modifying `framework_version` and `prod_image` fields in target YAML config files using `yq`.
- **Version_Comparator**: The component that performs semantic numeric comparison of version strings to determine if an upstream release is newer than the current config version.
- **Framework_Entry**: A single framework definition within the Tracker_Registry, containing the GitHub repo, tag prefix, config file mappings, and reviewer assignments.
- **Upstream_Release**: A non-draft, non-prerelease GitHub release from a tracked framework's repository.
- **Config_File**: A `.github/config/*.yml` file containing `common.framework_version` and `common.prod_image` fields for a specific framework image variant.
- **Update_Branch**: A Git branch named `auto-update/{framework}-{version}` created to hold config file changes for a detected upstream release.
- **PR_Description**: The pull request body populated with release context including upstream release link, version diff, changed files, and review guidance.
- **Dockerfile_Updater**: The component responsible for updating `ARG BASE_IMAGE=` lines in Dockerfiles to match the new upstream version.
- **CUDA_Python_Detector**: The component that inspects Docker Hub image metadata to detect changes in CUDA and Python versions between releases.
- **Test_Script_Renamer**: The component that renames version-named test setup scripts and updates workflow file references.
- **Slack_Notifier**: The component that sends key-value data to an existing Slack Workflow via webhook after a PR is created, providing targeted notifications for auto-update PRs. The Slack Workflow handles message formatting and channel routing.

## Requirements

### Requirement 1: Scheduled Upstream Release Detection

**User Story:** As an engineer, I want the system to automatically check for new upstream framework releases on a regular schedule, so that I don't have to manually monitor GitHub repositories.

#### Acceptance Criteria

1. THE Tracker_Workflow SHALL execute on a cron schedule of every 60 minutes.
2. WHEN the Tracker_Workflow executes, THE Tracker_Workflow SHALL load all Framework_Entries from the Tracker_Registry.
3. WHEN a `framework` input filter is provided via manual trigger, THE Tracker_Workflow SHALL process only the matching Framework_Entry.
4. WHEN no `framework` input filter is provided, THE Tracker_Workflow SHALL process all Framework_Entries in the Tracker_Registry.

### Requirement 2: Upstream Version Retrieval

**User Story:** As an engineer, I want the system to correctly retrieve the latest stable release version from upstream GitHub repositories, so that only production-ready versions trigger updates.

#### Acceptance Criteria

1. WHEN checking a Framework_Entry, THE Tracker_Workflow SHALL call the GitHub Releases API endpoint `GET /repos/{owner}/{repo}/releases/latest` for the configured `github_repo`.
2. WHEN the GitHub Releases API returns a release where `prerelease` is true or `draft` is true, THE Tracker_Workflow SHALL treat the framework as having no new stable release.
3. WHEN the GitHub Releases API returns a valid release, THE Version_Comparator SHALL strip the configured `tag_prefix` from the `tag_name` to produce a clean version string.
4. IF the `tag_prefix` is not specified in the Framework_Entry, THEN THE Version_Comparator SHALL default to stripping the prefix `"v"`.

### Requirement 3: Semantic Version Comparison

**User Story:** As an engineer, I want version comparisons to be numerically correct, so that the system never creates false update PRs or misses real updates.

#### Acceptance Criteria

1. THE Version_Comparator SHALL compare version strings segment-by-segment using numeric (not lexicographic) comparison across MAJOR, MINOR, and PATCH segments.
2. WHEN the upstream version is strictly greater than the current config version, THE Version_Comparator SHALL indicate a new version is available.
3. WHEN the upstream version is equal to or less than the current config version, THE Version_Comparator SHALL indicate no update is needed.
4. WHEN a version string has fewer than three segments, THE Version_Comparator SHALL pad missing segments with zero (e.g., `"0.16"` becomes `"0.16.0"`).

### Requirement 4: Idempotent PR Creation

**User Story:** As an engineer, I want the system to avoid creating duplicate PRs for the same upstream version, so that I am not overwhelmed with redundant notifications.

#### Acceptance Criteria

1. WHEN a new upstream version is detected, THE Tracker_Workflow SHALL check whether a branch named `auto-update/{framework_key}-{version}` already exists.
2. WHILE an Update_Branch for a given framework and version already exists, THE Tracker_Workflow SHALL skip PR creation for that framework.
3. WHEN no Update_Branch exists for the detected version, THE Tracker_Workflow SHALL create a new branch named `auto-update/{framework_key}-{version}` from `main`.

### Requirement 5: Config File Update

**User Story:** As an engineer, I want the system to correctly update only the version-related fields in config files, so that all other configuration remains intact.

#### Acceptance Criteria

1. WHEN updating a Config_File, THE Config_Updater SHALL set the `common.framework_version` field to the new upstream version string.
2. WHEN updating a Config_File, THE Config_Updater SHALL render the `prod_image_template` from the Framework_Entry using the new version's major and minor segments and write the result to the `common.prod_image` field.
3. WHEN updating a Config_File, THE Config_Updater SHALL preserve all fields other than `common.framework_version` and `common.prod_image` unchanged.
4. THE Config_Updater SHALL update all Config_Files listed in the Framework_Entry's `config_files` list for the given framework.
5. WHEN a Config_File is listed in the Framework_Entry's `exclude_configs` list, THE Config_Updater SHALL not modify that file.

### Requirement 6: Pull Request Creation with Context

**User Story:** As an engineer, I want auto-created PRs to include enough context (release notes link, changed files, review guidance), so that I can review them efficiently.

#### Acceptance Criteria

1. WHEN config files have been updated for a new version, THE Tracker_Workflow SHALL create a pull request targeting the `main` branch.
2. THE PR_Description SHALL include a link to the upstream release page (`html_url` from the GitHub Releases API).
3. THE PR_Description SHALL include the previous version and the new version.
4. THE PR_Description SHALL include the list of changed Config_Files.
5. THE PR_Description SHALL include a summary or link to the upstream release notes.
6. WHEN reviewers are specified in the Framework_Entry, THE Tracker_Workflow SHALL assign those reviewers to the created pull request.

### Requirement 7: Framework Isolation

**User Story:** As an engineer, I want updates to one framework to never affect another framework's config files, so that independent release cycles are maintained.

#### Acceptance Criteria

1. WHEN processing a Framework_Entry, THE Config_Updater SHALL modify only the Config_Files explicitly listed in that Framework_Entry's `config_files` mapping.
2. THE Config_Updater SHALL not modify any Config_File belonging to a different Framework_Entry, regardless of shared field names or framework identifiers.

### Requirement 8: Dry Run Support

**User Story:** As an engineer, I want to run the tracker in dry-run mode, so that I can verify detection logic without creating actual PRs.

#### Acceptance Criteria

1. WHEN the `dry-run` input is set to `true`, THE Tracker_Workflow SHALL detect version differences and log the results.
2. WHILE the `dry-run` input is `true`, THE Tracker_Workflow SHALL not create any branches, commits, or pull requests.

### Requirement 9: Extensible Framework Registry

**User Story:** As an engineer, I want to add new frameworks to track by editing a single configuration file, so that no workflow code changes are needed.

#### Acceptance Criteria

1. THE Tracker_Registry SHALL define each tracked framework as a keyed entry under a top-level `frameworks` mapping.
2. WHEN a new Framework_Entry is added to the Tracker_Registry, THE Tracker_Workflow SHALL process the new entry on its next scheduled run without any workflow file modifications.
3. THE Tracker_Registry SHALL require each Framework_Entry to specify `github_repo` and at least one entry in `config_files`.
4. THE Tracker_Registry SHALL support optional fields: `tag_prefix`, `exclude_configs`, and `reviewers`.

### Requirement 10: Error Handling and Resilience

**User Story:** As an engineer, I want the system to handle API failures and configuration errors gracefully, so that one failure does not block updates for other frameworks.

#### Acceptance Criteria

1. IF the GitHub Releases API returns a 404 for a Framework_Entry, THEN THE Tracker_Workflow SHALL log a warning and continue processing the remaining frameworks.
2. IF the GitHub Releases API returns a 403 rate-limit response, THEN THE Tracker_Workflow SHALL log an error including rate-limit headers and fail the job for that framework.
3. IF a Config_File update fails (e.g., `yq` error due to schema change), THEN THE Tracker_Workflow SHALL report the failure and continue processing other frameworks.
4. IF the `GITHUB_TOKEN` lacks required permissions (`contents: write`, `pull-requests: write`), THEN THE Tracker_Workflow SHALL fail with a descriptive error message.

### Requirement 11: Email Notification via GitHub PR System

**User Story:** As an engineer, I want to receive email notifications when automated PRs are created, so that I can review and merge them promptly.

#### Acceptance Criteria

1. WHEN a pull request is created by the Tracker_Workflow, THE Tracker_Workflow SHALL rely on GitHub's built-in PR notification system to send email notifications to assigned reviewers.
2. WHEN reviewers are configured in the Framework_Entry, THE Tracker_Workflow SHALL assign them to the PR so that GitHub triggers email notifications.

### Requirement 12: RC Tag Filtering

**User Story:** As an engineer, I want the system to reject release candidate (RC) versions by inspecting the version string itself, so that RC releases are filtered even when upstream repos don't properly mark them as pre-releases in the GitHub API.

#### Acceptance Criteria

1. WHEN the version string (after tag prefix stripping) contains an RC suffix pattern (e.g., `rc1`, `rc2`, `RC1`, `-rc.1`), THE Version_Comparator SHALL reject the version as non-stable.
2. THE RC suffix check SHALL be case-insensitive and match patterns: `rc\d+`, `RC\d+`, `-rc.\d+`.
3. WHEN a version is rejected due to an RC suffix, THE Tracker_Workflow SHALL log the rejection reason and continue to the next framework.

### Requirement 13: Dockerfile BASE_IMAGE Update

**User Story:** As an engineer, I want the system to automatically update the `ARG BASE_IMAGE=` line in Dockerfiles when a new upstream version is detected, so that Dockerfiles stay in sync with the framework version.

#### Acceptance Criteria

1. WHEN a Framework_Entry has a `dockerfiles` list configured, THE Config_Updater SHALL update the `ARG BASE_IMAGE=` default value in each listed Dockerfile.
2. THE Config_Updater SHALL render the `base_image_template` from the Framework_Entry by substituting `{version}` with the new upstream version string.
3. THE Config_Updater SHALL modify only the `ARG BASE_IMAGE=` line in each Dockerfile; all other Dockerfile content SHALL remain unchanged.
4. WHEN a Framework_Entry does not have a `dockerfiles` list, THE Config_Updater SHALL skip Dockerfile updates for that framework.

### Requirement 14: CUDA/Python Version Detection

**User Story:** As an engineer, I want the system to detect CUDA and Python version changes in upstream images and update config files accordingly, so that version metadata stays accurate without manual inspection.

#### Acceptance Criteria

1. WHEN a Framework_Entry has a `docker_hub_image` configured, THE Tracker_Workflow SHALL query the Docker Hub API to inspect the upstream image metadata for the new version tag.
2. IF the upstream image's CUDA version differs from `common.cuda_version` in the config files, THE Config_Updater SHALL update `common.cuda_version` in all config files for that framework.
3. IF the upstream image's Python version differs from `common.python_version` in the config files, THE Config_Updater SHALL update `common.python_version` in all config files for that framework.
4. IF the Docker Hub API is unreachable or the image metadata does not contain CUDA/Python version information, THE Tracker_Workflow SHALL add a warning to the PR description indicating that CUDA/Python versions could not be verified.
5. CUDA/Python detection failure SHALL NOT prevent PR creation.

### Requirement 15: Test Setup Script Renaming

**User Story:** As an engineer, I want the system to automatically rename version-named test setup scripts and update all workflow references, so that test infrastructure stays in sync with the framework version.

#### Acceptance Criteria

1. WHEN a Framework_Entry has a `test_setup_script` configured, THE Tracker_Workflow SHALL rename the test setup script from the old version pattern to the new version pattern using `git mv`.
2. THE Tracker_Workflow SHALL update all files listed in `test_setup_script.workflow_files` to replace references to the old script path with the new script path.
3. THE version in the script filename SHALL use underscored format (e.g., `0.16.0` → `0_16_0`, `0.17.0` → `0_17_0`).
4. WHEN a Framework_Entry does not have a `test_setup_script` configured, THE Tracker_Workflow SHALL skip test script renaming for that framework.

### Requirement 16: Slack Webhook Notification

**User Story:** As an engineer, I want to receive targeted Slack notifications when auto-update PRs are created, so that I get timely alerts without the noise of GitHub's general notification system.

#### Acceptance Criteria

1. WHEN a pull request is created by the Tracker_Workflow AND `notifications.slack.enabled` is `true` in the Tracker_Registry AND the `SLACK_WEBHOOK_URL` GitHub Actions secret is available, THE Slack_Notifier SHALL POST a JSON payload of key-value pairs to the Slack Workflow webhook URL.
2. THE Slack notification payload SHALL include the following key-value pairs: `framework_name`, `framework_version`, `pr_url`, `release_notes_url`, and `changed_files` (comma-separated string of modified file paths).
3. THE Slack webhook URL SHALL be stored as a GitHub Actions secret named `SLACK_WEBHOOK_URL` and SHALL NOT appear in the Tracker_Registry or any committed file.
4. IF the Slack webhook POST fails (HTTP error, network timeout, invalid URL), THE Tracker_Workflow SHALL log a warning but SHALL NOT fail the workflow or roll back the PR creation.
5. Message formatting and channel routing SHALL be handled by the receiving Slack Workflow, not by the Tracker_Workflow. THE Tracker_Workflow SHALL only provide raw data as key-value pairs.
