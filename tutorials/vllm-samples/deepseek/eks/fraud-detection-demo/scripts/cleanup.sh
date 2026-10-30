#!/bin/bash
docker ps -q | xargs docker stop 2>/dev/null || echo "No running containers"
docker ps -aq | xargs docker rm 2>/dev/null || echo "No containers to remove"
echo "Cleanup complete!"
