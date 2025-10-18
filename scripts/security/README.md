# Security Utilities

`scan_image.sh` wraps Trivy so build jobs can generate SBOMs and enforce vulnerability policy with a single call.

## Environment knobs
- `VULN_SEVERITY` (default `CRITICAL`): Comma-separated severity list for the scan.
- `VULN_FAIL_ON` (default `true`): When truthy, the script exits non-zero if the scan reports findings at the chosen severity.
- `GENERATE_SBOM` (default `true`): Toggle SBOM creation.
- `SBOM_FORMAT` (default `spdx-json`): Passed to `trivy sbom -f`.
- `SBOM_DIR` (default `sbom`): Directory where SBOM files are written.
- `TRIVY_BIN` (optional): Override the scanner binary path if needed.
- `DLC_VULN_POLICY` (default unset): if set to `warn`, the buildspecs export `VULN_FAIL_ON=false` so scans warn but do not fail.

Invoke with a fully qualified image tag, for example:

```sh
scripts/security/scan_image.sh 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5-gpu-py311-ubuntu22.04
```
