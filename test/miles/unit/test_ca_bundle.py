"""Verify the CA bundle that Python HTTPS clients in the image trust."""

import certifi


def test_ca_bundle_omits_the_compromised_emsign_root():
    # Its intermediate CA keys were compromised (GHSA-6q48-6c7w-9gw2); certifi
    # 2026.10.19 removed it.
    with open(certifi.where()) as f:
        assert "CN=emSign ECC Root CA - G3" not in f.read()
