*GitHub Issue #, if available:*

**Note**: 
- If merging this PR should also close the associated Issue, please also add that Issue # to the Linked Issues section on the right. 

- All PR's are checked weekly for staleness. This PR will be closed if not updated in 30 days.

### Description

### Tests run

<details>
<summary>DevToml</summary>

```toml
[dev]
# Set to "huggingface", for example, if you are a huggingface developer. Default is ""
partner_developer = ""
# Please only set it to true if you are preparing an EI related PR
# Do remember to revert it back to false before merging any PR (including EI dedicated PR)
ei_mode = false
# Please only set it to true if you are preparing a NEURON related PR
# Do remember to revert it back to false before merging any PR (including NEURON dedicated PR)
neuron_mode = false
# Please only set it to true if you are preparing a NEURONX related PR
# Do remember to revert it back to false before merging any PR (including NEURONX dedicated PR)
neuronx_mode = false
# Please only set it to true if you are preparing a GRAVITON related PR
# Do remember to revert it back to false before merging any PR (including GRAVITON dedicated PR)
graviton_mode = false
# Please only set it to True if you are preparing a HABANA related PR
# Do remember to revert it back to False before merging any PR (including HABANA dedicated PR)
habana_mode = false
# Please only set it to True if you are preparing a HUGGINGFACE TRCOMP related PR
# Do remember to revert it back to False before merging any PR (including HUGGINGFACE TRCOMP dedicated PR)
# This mode is used to build TF 2.6 and PT1.11 DLC
huggingface_trcomp_mode = false
# Please only set it to True if you are preparing a TRCOMP related PR
# Do remember to revert it back to False before merging any PR (including TRCOMP dedicated PR)
# This mode is used to build PT1.12 and above DLC
trcomp_mode = false
# Set deep_canary_mode to true to simulate Deep Canary Test conditions on PR for all frameworks in the
# build_frameworks list below. This will cause all image builds and non-deep-canary tests on the PR to be skipped,
# regardless of whether they are enabled or disabled below.
# Set graviton_mode to true to run Deep Canaries on Graviton images.
# Do remember to revert it back to false before merging any PR.
deep_canary_mode = false

[build]
# Add in frameworks you would like to build. By default, builds are disabled unless you specify building an image.
# available frameworks - ["autogluon", "huggingface_tensorflow", "huggingface_pytorch", "huggingface_tensorflow_trcomp", "huggingface_pytorch_trcomp", "pytorch_trcomp", "tensorflow", "mxnet", "pytorch", "stabilityai_pytorch"]
build_frameworks = []

# By default we build both training and inference containers. Set true/false values to determine which to build.
build_training = true
build_inference = true

# Set to false in order to remove datetime tag on PR builds
datetime_tag = true
# Note: Need to build the images at least once with datetime_tag = false
# before disabling new builds, or tests will fail
do_build = true
autopatch_build = false

[notify]
### Notify on test failures
### Off by default
notify_test_failures = false
  # Valid values: medium or high
  notification_severity = "medium"

[test]
### On by default
sanity_tests = true
  safety_check_test = false
  ecr_scan_allowlist_feature = false
ecs_tests = true
eks_tests = true
ec2_tests = true
# Set it to true if you are preparing a Benchmark related PR
ec2_benchmark_tests = false

### Set ec2_tests_on_heavy_instances = true to be able to run any EC2 tests that use large/expensive instance types by
### default. If false, these types of tests will be skipped while other tests will run as usual.
### These tests are run in EC2 test jobs, so ec2_tests must be true if ec2_tests_on_heavy_instances is true.
### Off by default (set to false)
ec2_tests_on_heavy_instances = false

### SM specific tests
### Off by default
sagemaker_local_tests = false

# run standard sagemaker remote tests from test/sagemaker_tests
sagemaker_remote_tests = false
# run efa sagemaker tests
sagemaker_efa_tests = false
# run release_candidate_integration tests
sagemaker_rc_tests = false
# run sagemaker benchmark tests
sagemaker_benchmark_tests = false

# SM remote EFA test instance type
sagemaker_remote_efa_instance_type = ""

# Run CI tests for nightly images
# false by default
nightly_pr_test_mode = false

use_scheduler = false

[buildspec_override]
# Assign the path to the required buildspec file from the deep-learning-containers folder
# For example:
# dlc-pr-tensorflow-2-habana-training = "habana/tensorflow/training/buildspec-2-10.yml"
# dlc-pr-pytorch-inference = "pytorch/inference/buildspec-1-12.yml"
# Setting the buildspec file path to "" allows the image builder to choose the default buildspec file.

### TRAINING PR JOBS ###

# Standard Framework Training
dlc-pr-mxnet-training = ""
dlc-pr-pytorch-training = ""
dlc-pr-tensorflow-2-training = ""
dlc-pr-autogluon-training = ""

# HuggingFace Training
dlc-pr-huggingface-tensorflow-training = ""
dlc-pr-huggingface-pytorch-training = ""

# Training Compiler
dlc-pr-huggingface-pytorch-trcomp-training = ""
dlc-pr-huggingface-tensorflow-2-trcomp-training = ""
dlc-pr-pytorch-trcomp-training = ""

# Neuron Training
dlc-pr-mxnet-neuron-training = ""
dlc-pr-pytorch-neuron-training = ""
dlc-pr-tensorflow-2-neuron-training = ""

# Stability AI Training
dlc-pr-stabilityai-pytorch-training = ""

# Habana Training
dlc-pr-pytorch-habana-training = ""
dlc-pr-tensorflow-2-habana-training = ""

### INFERENCE PR JOBS ###

# Standard Framework Inference
dlc-pr-mxnet-inference = ""
dlc-pr-pytorch-inference = ""
dlc-pr-tensorflow-2-inference = ""
dlc-pr-autogluon-inference = ""

# Neuron Inference
dlc-pr-mxnet-neuron-inference = ""
dlc-pr-pytorch-neuron-inference = ""
dlc-pr-tensorflow-1-neuron-inference = ""
dlc-pr-tensorflow-2-neuron-inference = ""

# HuggingFace Inference
dlc-pr-huggingface-tensorflow-inference = ""
dlc-pr-huggingface-pytorch-inference = ""
dlc-pr-huggingface-pytorch-neuron-inference = ""

# Stability AI Inference
dlc-pr-stabilityai-pytorch-inference = ""

# Graviton Inference
dlc-pr-mxnet-graviton-inference = ""
dlc-pr-pytorch-graviton-inference = ""
dlc-pr-tensorflow-2-graviton-inference = ""

# EIA Inference
dlc-pr-mxnet-eia-inference = ""
dlc-pr-pytorch-eia-inference = ""
dlc-pr-tensorflow-2-eia-inference = ""
```
</details>

**NOTE: By default, docker builds are disabled. In order to build your container, please update dlc_developer_config.toml and specify the framework to build in "build_frameworks"**
- [ ] I have run builds/tests on commit <INSERT COMMIT ID> for my changes.

**NOTE: If you are creating a PR for a new framework version, please ensure success of the standard, rc, and efa sagemaker remote tests by updating the dlc_developer_config.toml file:**
<details>
<summary>Expand</summary>

- [ ] `sagemaker_remote_tests = true`
- [ ] `sagemaker_efa_tests = true`
- [ ] `sagemaker_rc_tests = true`

**Additionally, please run the sagemaker local tests in at least one revision:**
- [ ] `sagemaker_local_tests = true`

</details>

### Formatting
- [ ] I have run `black -l 100` on my code (formatting tool: https://black.readthedocs.io/en/stable/getting_started.html)

### DLC image/dockerfile

#### Builds to Execute
<details>
<summary>Expand</summary>

Click the checkbox to enable a build to execute upon merge.

*Note: By default, pipelines are set to "latest". Replace with major.minor framework version if you do not want "latest".*

- [ ] build_pytorch_training_latest
- [ ] build_pytorch_inference_latest
- [ ] build_tensorflow_training_latest
- [ ] build_tensorflow_inference_latest

</details>

### Additional context

### PR Checklist 
<details>
<summary>Expand</summary>

- [ ] I've prepended PR tag with frameworks/job this applies to : [mxnet, tensorflow, pytorch] | [ei/neuron/graviton] | [build] | [test] | [benchmark] | [ec2, ecs, eks, sagemaker]
- [ ] If the PR changes affects SM test, I've modified dlc_developer_config.toml in my PR branch by setting sagemaker_tests = true and efa_tests = true
- [ ] If this PR changes existing code, the change fully backward compatible with pre-existing code. (Non backward-compatible changes need special approval.)
- [ ] (If applicable) I've documented below the DLC image/dockerfile this relates to
- [ ] (If applicable) I've documented below the tests I've run on the DLC image
- [ ] (If applicable) I've reviewed the licenses of updated and new binaries and their dependencies to make sure all licenses are on the Apache Software Foundation Third Party License Policy Category A or Category B license list.  See [https://www.apache.org/legal/resolved.html](https://www.apache.org/legal/resolved.html).
- [ ] (If applicable) I've scanned the updated and new binaries to make sure they do not have vulnerabilities associated with them.

#### NEURON/GRAVITON Testing Checklist
* When creating a PR:
- [ ] I've modified `dlc_developer_config.toml` in my PR branch by setting `neuron_mode = true` or `graviton_mode = true`

#### Benchmark Testing Checklist
* When creating a PR:
- [ ] I've modified `dlc_developer_config.toml` in my PR branch by setting `ec2_benchmark_tests = true` or `sagemaker_benchmark_tests = true`
</details>

### Pytest Marker Checklist
<details>
<summary>Expand</summary>

- [ ] (If applicable) I have added the marker `@pytest.mark.model("<model-type>")` to the new tests which I have added, to specify the Deep Learning model that is used in the test (use `"N/A"` if the test doesn't use a model)
- [ ] (If applicable) I have added the marker `@pytest.mark.integration("<feature-being-tested>")` to the new tests which I have added, to specify the feature that will be tested
- [ ] (If applicable) I have added the marker `@pytest.mark.multinode(<integer-num-nodes>)` to the new tests which I have added, to specify the number of nodes used on a multi-node test
- [ ] (If applicable) I have added the marker `@pytest.mark.processor(<"cpu"/"gpu"/"eia"/"neuron">)` to the new tests which I have added, if a test is specifically applicable to only one processor type
</details>


By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license. I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
