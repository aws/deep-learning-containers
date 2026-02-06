#!/bin/bash

source fsdp.conf

# Login to DLC registry
echo " creating fsdp.yaml file "

cat fsdp.yaml-template | envsubst > fsdp.yaml