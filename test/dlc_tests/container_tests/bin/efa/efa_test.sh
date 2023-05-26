#!/bin/bash
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

set -ex

# Script to run fi_pingpong locally over EFA to test connectivity.

if ! command -v fi_info >/dev/null 2>&1 || ! command -v fi_pingpong >/dev/null 2>&1; then
        echo "Error: required libfabric binaries not found."
        exit 1
fi

if ! fi_info -p efa >/dev/null 2>&1; then
        echo "Error: EFA libfabric provider not detected." >&2
        exit 1
fi

echo "Starting server..."
FI_EFA_ENABLE_SHM_TRANSFER=0 fi_pingpong -e rdm -p efa >/dev/null 2>&1 &
sleep 0.5

echo "Starting client..."
FI_EFA_ENABLE_SHM_TRANSFER=0 timeout 8 fi_pingpong -e rdm -p efa localhost
ret=$?
if [ $ret -ne 0 ]; then
        if [ $ret -eq 124 ]; then
                echo "Error: fi_pingpong test timed out." >&2
        else
                echo "Error: fi_pingpong test returned $ret." >&2
        fi
fi
kill %1
exit $ret
