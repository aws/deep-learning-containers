*GitHub Issue #, if available:*

Note: If merging this PR should also close the associated Issue, please also add that Issue # to the Linked Issues section on the right. 



### Description

### Tests run
**NOTE: If you are creating a PR for a new framework version, please ensure success of the standard, rc, and efa sagemaker remote tests by updating the dlc_developer_config.toml file:**

- [ ] Revision A: `sagemaker_remote_tests = "standard"`
- [ ] Revision B: `sagemaker_remote_tests = "rc"`
- [ ] Revision C: `sagemaker_remote_tests = "efa"`

**Additionally, please run the sagemaker local tests in at least one revision:**
- [ ] `sagemaker_local_tests = true`

### DLC image/dockerfile

### Additional context

## Label Checklist
- [ ] I have added the project label for this PR (*<project_name>* or "Improvement")

## PR Checklist
- [ ] I've prepended PR tag with frameworks/job this applies to : [mxnet, tensorflow, pytorch] | [ei/neuron/graviton] | [build] | [test] | [benchmark] | [ec2, ecs, eks, sagemaker]
- [ ] If the PR changes affects SM test, I've modified dlc_developer_config.toml in my PR branch by setting sagemaker_tests = true and efa_tests = true
- [ ] If this PR changes existing code, the change fully backward compatible with pre-existing code. (Non backward-compatible changes need special approval.)
- [ ] (If applicable) I've documented below the DLC image/dockerfile this relates to
- [ ] (If applicable) I've documented below the tests I've run on the DLC image
- [ ] (If applicable) I've reviewed the licenses of updated and new binaries and their dependencies to make sure all licenses are on the Apache Software Foundation Third Party License Policy Category A or Category B license list.  See [https://www.apache.org/legal/resolved.html](https://www.apache.org/legal/resolved.html).
- [ ] (If applicable) I've scanned the updated and new binaries to make sure they do not have vulnerabilities associated with them.

## Pytest Marker Checklist
- [ ] (If applicable) I have added the marker `@pytest.mark.model("<model-type>")` to the new tests which I have added, to specify the Deep Learning model that is used in the test (use `"N/A"` if the test doesn't use a model)
- [ ] (If applicable) I have added the marker `@pytest.mark.integration("<feature-being-tested>")` to the new tests which I have added, to specify the feature that will be tested
- [ ] (If applicable) I have added the marker `@pytest.mark.multinode(<integer-num-nodes>)` to the new tests which I have added, to specify the number of nodes used on a multi-node test
- [ ] (If applicable) I have added the marker `@pytest.mark.processor(<"cpu"/"gpu"/"eia"/"neuron"/"graviton">)` to the new tests which I have added, if a test is specifically applicable to only one processor type

#### EIA/NEURON/GRAVITON Testing Checklist
* When creating a PR:
- [ ] I've modified `dlc_developer_config.toml` in my PR branch by setting `ei_mode = true`, `neuron_mode = true` or `graviton_mode = true`

#### Benchmark Testing Checklist
* When creating a PR:
- [ ] I've modified `dlc_developer_config.toml` in my PR branch by setting `benchmark_mode = true`

By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license. I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
