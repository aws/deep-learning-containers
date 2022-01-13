#!/bin/bash

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