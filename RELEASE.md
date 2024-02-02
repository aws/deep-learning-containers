## Creating a release

Follow the Github official [documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) to create release.

* Step 4:
    * Create a new tag for each release.
    * The version number should be same as the one of packaged [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html).
* Step 5:
    * The target should always be "main" branch.
* Step 6:
    * Choose "auto".
* Step 7:
    * Title is same as version number.
* Step 8:
    * Automatically generate release notes.
* Steps 9 - 12 can be skipped for now:
    * No need to attach binaries. The Neuron DLC images are released in ECR. The releases of this repo are only pointing Dockerfiles.
