VULN_SEVERITY ?= CRITICAL
VULN_FAIL_ON ?= true
GENERATE_SBOM ?= true
SBOM_DIR ?= sbom

.PHONY: trivy-install scan-image example

trivy-install:
	@if command -v brew >/dev/null 2>&1; then \
		echo "Installing Trivy via Homebrew..."; \
		brew install trivy; \
	else \
		echo "Homebrew is not installed. Install it from https://brew.sh/ and rerun 'make trivy-install'."; \
		exit 1; \
	fi

scan-image:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make scan-image IMAGE=<image_tag> [VULN_SEVERITY=CRITICAL] [VULN_FAIL_ON=true] [GENERATE_SBOM=true]"; \
		exit 1; \
	fi
	@VULN_SEVERITY="$(VULN_SEVERITY)" \
		VULN_FAIL_ON="$(VULN_FAIL_ON)" \
		GENERATE_SBOM="$(GENERATE_SBOM)" \
		SBOM_DIR="$(SBOM_DIR)" \
		scripts/security/scan_image.sh "$(IMAGE)"

example:
	@echo "docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py38-ubuntu20.04"
	@echo "make scan-image IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py38-ubuntu20.04"
