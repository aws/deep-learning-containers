#!/bin/bash
set -e

echo "Start EFA checker test"
efa_checker_single_node --no_instance_id --verbose
echo "Complete EFA checker test"
