#!/bin/bash

MODEL_STORE=/opt/ml/model
TS_CONFIG=/home/model-server/config.properties
MODEL_PATH=""

while getopts ":m:t:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG"
    ;;
    t) TS_CONFIG="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Model path: %s\n" "$MODEL_PATH"
printf "TS_CONFIG: %s\n" "$TS_CONFIG"
# Start the Model Server
if [[ -z "$MODEL_PATH" ]]; then
  torchserve --start --ts-config /home/model-server/config.properties --model-store /opt/ml/model &
else
  torchserve --start --ts-config $TS_CONFIG --models $MODEL_PATH &
fi
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start TF Model Server: $status"
  exit $status
fi