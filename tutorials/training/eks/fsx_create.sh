#!/bin/bash

##Create an FSx for Lustre file system
# Create a security group for FSx Lustre
FSX_SG_ID=$(aws ec2 create-security-group --group-name fsx-lustre-sg \
  --description "Security group for FSx Lustre" \
  --vpc-id $(aws eks describe-cluster --name eks-p4d \
  --query "cluster.resourcesVpcConfig.vpcId" --output text) \
  --query "GroupId" --output text)

echo "Created security group: $FSX_SG_ID"

# Add inbound rules for FSx Lustre
aws ec2 authorize-security-group-ingress --group-id $FSX_SG_ID \
  --protocol tcp --port 988-1023 \
  --source-group $(aws eks describe-cluster --name eks-p4d \
  --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

aws ec2 authorize-security-group-ingress --group-id $FSX_SG_ID \
     --protocol tcp --port 988-1023 \
     --source-group $FSX_SG_ID

# Create the FSx Lustre filesystem
SUBNET_ID=$(aws eks describe-cluster --name eks-p4d \
  --query "cluster.resourcesVpcConfig.subnetIds[0]" --output text)

echo "Using subnet: $SUBNET_ID"

FSX_ID=$(aws fsx create-file-system --file-system-type LUSTRE \
  --storage-capacity 1200 --subnet-ids $SUBNET_ID \
  --security-group-ids $FSX_SG_ID --lustre-configuration DeploymentType=SCRATCH_2 \
  --tags Key=Name,Value=vllm-model-storage \
  --query "FileSystem.FileSystemId" --output text)

echo "Created FSx filesystem: $FSX_ID"

# Wait for the filesystem to be available (typically takes 5-10 minutes)
echo "Waiting for filesystem to become available..."
aws fsx describe-file-systems --file-system-id $FSX_ID \
  --query "FileSystems[0].Lifecycle" --output text

# You can run the above command periodically until it returns "AVAILABLE"
# Example: watch -n 30 "aws fsx describe-file-systems --file-system-id $FSX_ID --query FileSystems[0].Lifecycle --output text"

# Get the DNS name and mount name
FSX_DNS=$(aws fsx describe-file-systems --file-system-id $FSX_ID \
  --query "FileSystems[0].DNSName" --output text)

FSX_MOUNT=$(aws fsx describe-file-systems --file-system-id $FSX_ID \
  --query "FileSystems[0].LustreConfiguration.MountName" --output text)

echo "FSx DNS: $FSX_DNS"
echo "FSx Mount Name: $FSX_MOUNT"

export FSX_ID=$FSX_ID

export FSX_DNS=$FSX_DNS

export FSX_MOUNT=$FSX_MOUNT

cat fsx-pvc-static.yaml-template | envsubst > fsx-pvc-static.yaml

