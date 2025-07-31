#!/bin/bash

# setup_fsx_vllm.sh
# Script to mount FSx and setup VLLM environment


# Get FSx DNS name from argument
FSX_DNS_NAME=$1
MOUNT_NAME=$2

# Function to log messages with hostname
log() {
    local HOSTNAME=$(hostname)
    echo "[Host ${HOSTNAME}] $1"
}


# Function to check if command was successful
check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

if [ -z "$FSX_DNS_NAME" ] || [ -z "$MOUNT_NAME" ]; then
    echo "Usage: $0 <FSX_DNS_NAME> <MOUNT_NAME>"
    exit 1
fi

# Install required packages
log "Installing required packages..."
sudo yum install -y nfs-utils git
check_error "Failed to install base packages"


# Install the latest Lustre client
log "Installing latest Lustre client..."
sudo yum install -y lustre-client
check_error "Failed to install Lustre client"


# Create FSx mount directory
log "Creating FSx mount directory..."
sudo mkdir -p /fsx
check_error "Failed to create /fsx directory"


# Modify mount command to include verbose output
sudo mount -t lustre -o relatime,flock ${FSX_DNS_NAME}@tcp:/${MOUNT_NAME} /fsx

# Create VLLM directory in FSx
log "Creating VLLM directory..."
sudo mkdir -p /fsx/vllm-dlc

check_error "Failed to create /fsx/vllm-dlc directory"

# Set proper permissions
log "Setting proper permissions..."
sudo chown -R ec2-user:ec2-user /fsx/vllm-dlc
check_error "Failed to set permissions"

cd /fsx/vllm-dlc
git clone https://github.com/vllm-project/vllm.git

# Download ShareGPT dataset
log "Downloading ShareGPT dataset..."
cd /fsx/vllm-dlc && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
check_error "Failed to download ShareGPT dataset"

log "Setup completed successfully!"

    