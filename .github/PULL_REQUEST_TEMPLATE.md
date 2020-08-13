*Issue #, if available:*

## Checklist
- [ ] I've prepended PR tag with frameworks/job this applies to : [mxnet, tensorflow, pytorch] | [build] | [test] | [build, test] | [ec2, ecs, eks, sagemaker]
- [ ] (If applicable) I've documented below the DLC image/dockerfile this relates to
- [ ] (If applicable) I've documented below the tests I've run on the DLC image
- [ ] (If applicable) I've reviewed the licenses of updated and new binaries and their dependencies to make sure all licenses are on the Apache Software Foundation Third Party License Policy Category A or Category B license list.  See [https://www.apache.org/legal/resolved.html](https://www.apache.org/legal/resolved.html).
- [ ] (If applicable) I've scanned the updated and new binaries to make sure they do not have vulnerabilities associated with them.

#### EIA Checklist
* When creating a PR:
- [ ] I've modified ```src/config/build_config.py``` in my PR branch by setting ```ENABLE_EI_MODE = True```
* When PR is reviewed and ready to be merged:
- [ ] I've reverted the code change on the config file mentioned above

* For reviewer, before merging, please cross-check:
- [ ] I've verified the code change on the config file mentioned above has already been reverted

*Description:*

*Tests run:*

*DLC image/dockerfile:*

*Additional context:*


By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license. I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.

