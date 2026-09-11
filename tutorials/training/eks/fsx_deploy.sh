#!/bin/bash

# This script intergrates your EKS cluster with Amazon FSx for Lustre.
# It requres all values in fsx.conf to be configured.
# To learn more about FSx and the steps involved in setting it up, please refer to the following links
# https://aws.amazon.com/blogs/opensource/using-fsx-lustre-csi-driver-amazon-eks/
# https://docs.aws.amazon.com/eks/latest/userguide/fsx-csi.html
# https://github.com/kubernetes-sigs/aws-fsx-csi-driver
# If dynamic provisioning of FSx volumes through the storage class does not work,
# you can also use static provisioning using the manifests in this folder or following instructions from here:
# https://github.com/kubernetes-sigs/aws-fsx-csi-driver/blob/master/examples/kubernetes/static_provisioning/README.md

source fsx.conf

# FSx Policy
echo ""
echo "Checking if FSx Policy ${FSX_POLICY_NAME} exists ..."
POLICY_ARN=$(aws iam list-policies --query Policies[?PolicyName=="'${FSX_POLICY_NAME}'"].{Arn:Arn} --output text)
echo ""
if [ "$POLICY_ARN" == "" ]; then
        echo "Policy does not exist."
        echo "Creating FSX Policy ${FSX_POLICY_NAME} ..."
        POLICY_ARN=$(aws iam create-policy --policy-name ${FSX_POLICY_NAME} --policy-document $FSX_POLICY_DOC --query "Policy.Arn" --output text)
else
        echo "Policy ${FSX_POLICY_NAME} found"
fi
echo "POLICY_ARN=$POLICY_ARN"

# EKS Instance Profiles
echo ""
echo "Checking instance profiles ..."
for index in ${!EKS_INSTANCE_PROFILE_NAMES[@]}; do
        EKS_INSTANCE_PROFILE_NAME=${EKS_INSTANCE_PROFILE_NAMES[$index]}
        echo ""
        echo "Instance profile ${EKS_INSTANCE_PROFILE_NAME} ..."
        INSTANCE_PROFILE=$(aws iam list-instance-profiles --query InstanceProfiles[?InstanceProfileName=="'${EKS_INSTANCE_PROFILE_NAME}'"].{InstanceProfileName:InstanceProfileName} --output text)
        if [ "$INSTANCE_PROFILE" == "" ]; then
                echo "Not found."
                echo "Please check fsx.conf and try again."
                echo "The configured instance profile must exist"
                echo "Describe one of the EKS instances in each node group that should have access to FSx "
                echo "and find its attached instance profile. Update the array in fsx.conf and try again."
                exit 1
        else
                echo "Found."
                echo "Getting instance profile role ..."
                ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name ${INSTANCE_PROFILE} --query InstanceProfile.Roles[0].RoleName --output text)
                echo "Attaching FSx Policy to role ${ROLE_NAME} ..."
                aws iam attach-role-policy --policy-arn ${POLICY_ARN} --role-name ${ROLE_NAME}
        fi
done

# FSx CSI driver
echo ""
echo "Installing FSx CSI driver ..."
kubectl apply -k "github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"
echo "FSx pods in kube-system namespace ..."
kubectl -n kube-system get pods | grep fsx

# Subnet CIDR
echo ""
echo "Getting CIDR for subnet ${FSX_SUBNET_ID} ..."
SUBNET_CIDR=$(aws ec2 describe-subnets --query Subnets[?SubnetId=="'${FSX_SUBNET_ID}'"].{CIDR:CidrBlock} --output text)
if [ "$SUBNET_CIDR" == "" ]; then
        echo "Subnet not found."
        echo "Please check the subnet id provided in fsx.conf and try again"
        exit 1
else
        echo "Subnet found"
        echo "CIDR=$SUBNET_CIDR"
        VPC_ID=$(aws ec2 describe-subnets --query Subnets[?SubnetId=="'${FSX_SUBNET_ID}'"].{VpcId:VpcId} --output text)
fi

# Security Group
echo ""
echo "Checking security group ${FSX_SECURITY_GROUP_NAME} ..."
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --query SecurityGroups[?GroupName=="'${FSX_SECURITY_GROUP_NAME}'"].{GroupId:GroupId} --output text)
if [ "$SECURITY_GROUP_ID" == "" ]; then
        echo "Not found. Creating ..."
        SECURITY_GROUP_ID=$(aws ec2 create-security-group --vpc-id ${VPC_ID} --group-name ${FSX_SECURITY_GROUP_NAME} --description "FSx for Lustre Security Group" --query "GroupId" --output text)
        echo "Authorizing FSx ..."
        aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 988 --cidr ${SUBNET_CIDR}
else
        echo "Found."
fi
echo "SECURITY_GROUP_ID=${SECURITY_GROUP_ID}"

# Storage Class
echo ""
echo "Creating storage class ${FSX_STORAGE_CLASS_NAME} ..."
cat > fsx-storage-class.yaml <<EOF
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: ${FSX_STORAGE_CLASS_NAME}
provisioner: fsx.csi.aws.com
parameters:
  subnetId: ${FSX_SUBNET_ID}
  securityGroupIds: ${SECURITY_GROUP_ID}
EOF
kubectl apply -f fsx-storage-class.yaml
kubectl get sc
