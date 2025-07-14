#!/bin/bash

# setup_fsx_vllm.sh
# Script to mount FSx and setup VLLM environment

# Function to check if command was successful
check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Get FSx DNS name from argument
FSX_DNS_NAME=$1
MOUNT_NAME=$2

if [ -z "$FSX_DNS_NAME" ] || [ -z "$MOUNT_NAME" ]; then
    echo "Usage: $0 <FSX_DNS_NAME> <MOUNT_NAME>"
    exit 1
fi

# Install required packages
echo "Installing required packages..."
sudo yum install -y nfs-utils git
check_error "Failed to install required packages"

# Create FSx mount directory
echo "Creating FSx mount directory..."
sudo mkdir -p /fsx
check_error "Failed to create /fsx directory"

# Add filesystem check before mounting
echo "Checking FSx DNS name resolution..."
nslookup ${FSX_DNS_NAME}
check_error "Failed to resolve FSx DNS name"

# Add network connectivity test
echo "Testing network connectivity..."
ping -c 1 ${FSX_DNS_NAME}
check_error "Failed to ping FSx endpoint"

# Modify mount command to include verbose output
sudo mount -t lustre -o relatime,flock,_netdev,verbose ${FSX_DNS_NAME}@tcp:${MOUNT_NAME} /fsx

# Create VLLM directory in FSx
echo "Creating VLLM directory..."
sudo mkdir -p /fsx/vllm
check_error "Failed to create /fsx/vllm directory"

# Set proper permissions
echo "Setting proper permissions..."
sudo chown -R ec2-user:ec2-user /fsx/vllm
check_error "Failed to set permissions"

# Clone VLLM repository
echo "Cloning VLLM repository..."
cd /fsx/vllm && git clone https://github.com/vllm-project/vllm/
check_error "Failed to clone VLLM repository"

# Download ShareGPT dataset
echo "Downloading ShareGPT dataset..."
cd /fsx/vllm && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
check_error "Failed to download ShareGPT dataset"

echo "Setup completed successfully!"
