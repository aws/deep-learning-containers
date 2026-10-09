# SGLang AL2023 local patches

`*.patch` files here are applied with `git apply` to the `SGLANG_REF` checkout in the
builder stage of `docker/sglang/Dockerfile.amzn2023`, in filename order, right after the
clone. An empty directory is the normal state; a patch that no longer applies to
`SGLANG_REF` fails the build rather than silently producing an unpatched image.

Patches come from two places:

- **Committed here** — applied to every build, including releases.
- **S3** — `scripts/ci/build/sglang_server/pre_build.sh` copies
  `s3://dlc-cicd-models/build-patches/sglang_server/<sglang_ref>/*.patch` into this
  directory before every CI build, releases included. The patch stays out of the
  repository but ships in released images. It is keyed by `sglang_ref`, so it silently stops applying once the ref moves.
  Uploading a patch does not trigger a build; push a commit that touches a build path.

Either way, use a patch only to carry a change the pinned `SGLANG_REF` predates, and drop
it in the same PR that moves `SGLANG_REF` past it.

Generate a patch against the exact pinned ref:

```
git -C <sglang-clone> diff <SGLANG_REF> <your-work> > <topic>.patch
```

Do not re-wrap or reformat a patch (no `git apply --whitespace=fix`) — a patch that itself
contains a patch may carry a checksum over its own bytes. The build discards `git apply`
output, so reproduce an apply failure locally against `SGLANG_REF`.

Two caveats when adding a patch:

- The upstream test suite is not patched. `.github/workflows/sglang.tests-upstream.yml`
  checks out pristine `SGLANG_REF` and puts it on `PYTHONPATH` ahead of the image's
  install, so those tests exercise unpatched code. Apply the patch to that checkout too,
  gated on the AL2023 image — the workflow is shared with the Ubuntu and Lambda SGLang
  images, which build without these patches.
- The image does not record what was applied. `SETUPTOOLS_SCM_PRETEND_VERSION` stamps the
  version from `SGLANG_REF` alone, and `.git` is stripped from the runtime stage, so a
  patched image reports the same version as an unpatched one.
