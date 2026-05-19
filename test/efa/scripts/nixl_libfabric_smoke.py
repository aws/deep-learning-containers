"""NIXL LIBFABRIC backend packaging smoke test.

Catches regressions where the nixl-cu* wheel ships without the LIBFABRIC plugin
or where libfabric on the system has no EFA provider compiled in. Standalone
Python file so the orchestrator can invoke it through `docker exec bash -c`
without heredoc/quoting headaches.
"""

import os
import sys

os.environ["FI_PROVIDER"] = "efa"

from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402

agent = nixl_agent("efa-smoke", nixl_agent_config(backends=[]))
# Raises nixlNotFoundError if plugin .so missing,
# nixlBackendError if libfabric has no EFA provider.
agent.create_backend("LIBFABRIC", {"provider": "efa"})
print("LIBFABRIC backend created with provider=efa")
sys.exit(0)
