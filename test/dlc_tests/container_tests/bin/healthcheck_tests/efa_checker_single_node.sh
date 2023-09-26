#!/bin/bash
set -e

echo "Start EFA checker single node test"
efa_checker_single_node --no_instance_id --verbose
echo "Complete EFA checker single node test"
