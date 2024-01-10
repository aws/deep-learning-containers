#!/bin/bash

# Copyright 2018-2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

neuron_monitor_running=0
if [[ ! -z "${NEURON_MONITOR_CW_REGION}" ]]; then
    # Start neuron monitor. If namespace/region variable is set then use it.
    if [[ ! -z "${NEURON_MONITOR_CONFIG_FILE}" ]]; then
        config="--config-file ${NEURON_MONITOR_CONFIG_FILE:+ $NEURON_MONITOR_CONFIG_FILE}"
    fi
    if [[ ! -z "${NEURON_MONITOR_CW_NAMESPACE}" ]]; then
        mnnamespace="--namespace ${NEURON_MONITOR_CW_NAMESPACE:+ $NEURON_MONITOR_CW_NAMESPACE}"
    fi
    region="--region ${NEURON_MONITOR_CW_REGION:+ $NEURON_MONITOR_CW_REGION}"
    /opt/aws/neuron/bin/neuron-monitor ${config:+ $config} | /opt/aws/neuron/bin/neuron-monitor-cloudwatch.py ${mnnamespace:+ $mnnamespace} ${region:+$region} >> /tmp/nm.log 2>&1 &
    nm_pid=$!
    echo "Neuron Monitor Started"
    neuron_monitor_running=1
fi
