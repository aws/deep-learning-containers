# Local Scanning Quickstart

Verify vulnerability scans and SBOM generation locally on a Mac (including Apple Silicon) without needing any AWS account access.

## 1. Install Docker Desktop
- Download and install Docker Desktop for Mac from <https://www.docker.com/products/docker-desktop/>.
- Start Docker Desktop and ensure `docker version` works from a terminal.

## 2. Install Trivy (scanner)
```sh
make trivy-install
```
If Homebrew is not installed, follow the instructions at <https://brew.sh/> and rerun the target, or manually run:
```sh
brew install trivy
```

## 3. Pull a sample AWS DLC image
Grab a small CPU-only training image from the public DLC registry:
```sh
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py38-ubuntu20.04
```

## 4. Run the vulnerability scan + SBOM generator
Use the Makefile wrapper so environment knobs match the CI integration:
```sh
make scan-image \
  IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py38-ubuntu20.04
```
Optional parameters (pass on the command line) include `VULN_SEVERITY`, `VULN_FAIL_ON`, `GENERATE_SBOM`, and `SBOM_DIR`.

## 5. Inspect results
- SBOM artifacts land under `sbom/` by default (one JSON per image tag).
- Trivy scan output appears in the terminal; non-zero exit indicates findings at or above the configured severity.

That’s it—this mirrors the planned CI behavior, making it easy to iterate locally before opening a pull request.
