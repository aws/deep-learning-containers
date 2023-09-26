#!/bin/bash
set -e

echo "Check whether EFA checker binary exists"
ls -lt /usr/bin/efa_checker_single_node
ls -lt /usr/bin/efa_checker_multi_node
