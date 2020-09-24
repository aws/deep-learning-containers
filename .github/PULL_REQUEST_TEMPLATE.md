*Issue #, if available:*

## PR Checklist
- [ ] I've prepended PR tag with frameworks/job this applies to : [mxnet, tensorflow, pytorch] | [ei/neuron] | [build] | [test] | [benchmark] | [ec2, ecs, eks, sagemaker]
- [ ] (If applicable) I've documented below the DLC image/dockerfile this relates to
- [ ] (If applicable) I've documented below the tests I've run on the DLC image
- [ ] (If applicable) I've reviewed the licenses of updated and new binaries and their dependencies to make sure all licenses are on the Apache Software Foundation Third Party License Policy Category A or Category B license list.  See [https://www.apache.org/legal/resolved.html](https://www.apache.org/legal/resolved.html).
- [ ] (If applicable) I've scanned the updated and new binaries to make sure they do not have vulnerabilities associated with them.

## Pytest Marker Checklist
- [ ] (If applicable) I have added the marker `@pytest.mark.model("<model-type>")` to the new tests which I have added, to specify the Deep Learning model that is used in the test (use `"N/A"` if the test doesn't use a model)
- [ ] (If applicable) I have added the marker `@pytest.mark.integration("<feature-being-tested>")` to the new tests which I have added, to specify the feature that will be tested
- [ ] (If applicable) I have added the marker `@pytest.mark.multinode(<integer-num-nodes>)` to the new tests which I have added, to specify the number of nodes used on a multi-node test
- [ ] (If applicable) I have added the marker `@pytest.mark.processor(<"cpu"/"gpu"/"eia"/"neuron">)` to the new tests which I have added, if a test is specifically applicable to only one processor type

#### EIA/NEURON Checklist
* When creating a PR:
- [ ] I've modified `src/config/build_config.py` in my PR branch by setting `ENABLE_EI_MODE = True` or `ENABLE_NEURON_MODE = True`
* When PR is reviewed and ready to be merged:
- [ ] I've reverted the code change on the config file mentioned above
#### Benchmark Checklist
* When creating a PR:
- [ ] I've modified `src/config/test_config.py` in my PR branch by setting `ENABLE_BENCHMARK_DEV_MODE = True`
* When PR is reviewed and ready to be merged:
- [ ] I've reverted the code change on the config file mentioned above

## Reviewer Checklist
* For reviewer, before merging, please cross-check:
- [ ] I've verified the code change on the config file mentioned above has already been reverted

*Description:*

*Tests run:*

*DLC image/dockerfile:*

*Additional context:*


By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license. I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.

