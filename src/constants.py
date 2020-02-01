"""
Define the constants used by the build system
"""

# Function Status Codes
SUCCESS = 0
FAIL = 1

# Left and right padding between text and margins in output
PADDING = 1

# Docker connections
DOCKER_URL="unix://var/run/docker.sock"

STATUS_MESSAGE = {SUCCESS: "Success", FAIL: "Failed"}
