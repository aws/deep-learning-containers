#!/bin/bash
set -e

echo "Start DCGM test"
dcgmi diag -r 1
echo "Complete DCGM test"