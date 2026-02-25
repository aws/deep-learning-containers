## Purpose

## Test Plan

## Test Result

______________________________________________________________________

<details>
<summary>Toggle if you are merging into master Branch</summary>

By default, docker image builds and tests are disabled. Two ways to run builds and tests:

1. Using dlc_developer_config.toml
1. Using this PR description (currently only supported for PyTorch, TensorFlow, vllm, and base images)

<details>
<summary>How to use the helper utility for updating dlc_developer_config.toml</summary>

Assuming your remote is called `origin` (you can find out more with `git remote -v`)...

- Run default builds and tests for a particular buildspec - also commits and pushes changes to remote; Example:

`python src/prepare_dlc_dev_environment.py -b </path/to/buildspec.yml> -cp origin`

- Enable specific tests for a buildspec or set of buildspecs - also commits and pushes changes to remote; Example:

`python src/prepare_dlc_dev_environment.py -b </path/to/buildspec.yml> -t sanity_tests -cp origin`

- Restore TOML file when ready to merge

`python src/prepare_dlc_dev_environment.py -rcp origin`

**NOTE: If you are creating a PR for a new framework version, please ensure success of the local, standard, rc, and efa sagemaker tests by updating the dlc_developer_config.toml file:**

- [ ] `sagemaker_remote_tests = true`
- [ ] `sagemaker_efa_tests = true`
- [ ] `sagemaker_rc_tests = true`
- [ ] `sagemaker_local_tests = true`

</details>

<details>
<summary>How to use PR description</summary>
Use the code block below to uncomment commands and run the PR CodeBuild jobs. There are two commands available:

- `# /buildspec <buildspec_path>`
  - e.g.: `# /buildspec pytorch/training/buildspec.yml`
  - If this line is commented out, dlc_developer_config.toml will be used.
- `# /tests <test_list>`
  - e.g.: `# /tests sanity security ec2`
  - If this line is commented out, it will run the default set of tests (same as the defaults in dlc_developer_config.toml): `sanity, security, ec2, ecs, eks, sagemaker, sagemaker-local`.

</details>

```
# /buildspec <buildspec_path>
# /tests <test_list>
```

</details>

<details>
<summary>Toggle if you are merging into main Branch</summary>

## PR Checklist

- [] I ran `pre-commit run --all-files` locally before creating this PR. (Read [DEVELOPMENT.md](https://github.com/aws/deep-learning-containers/blob/main/DEVELOPMENT.md) for details).

</details>
