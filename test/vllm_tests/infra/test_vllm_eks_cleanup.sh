#!/bin/bash
# Cleanup script for vLLM DeepSeek 32B deployment on EKS
# This script deletes all resources created for the vLLM deployment
# with appropriate wait times to ensure proper deletion

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# AWS Profile to use
REGION="us-west-2"
CLUSTER_NAME="vllm-cluster"
NODEGROUP_NAME="vllm-p4d-nodes-efa"

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to wait for a resource to be deleted
wait_for_deletion() {
    local check_command="$1"
    local resource_name="$2"
    local timeout_seconds="$3"
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout_seconds))
    
    echo -e "${YELLOW}Waiting for $resource_name to be deleted (timeout: ${timeout_seconds}s)...${NC}"
    
    while true; do
        if ! eval "$check_command" &>/dev/null; then
            print_success "$resource_name deleted successfully"
            return 0
        fi
        
        current_time=$(date +%s)
        if [ $current_time -gt $end_time ]; then
            print_warning "$resource_name deletion timed out after ${timeout_seconds}s"
            return 1
        fi
        
        echo -n "."
        sleep 10
    done
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check for required tools
for cmd in kubectl aws eksctl helm; do
    if ! command_exists $cmd; then
        print_error "Required command '$cmd' not found. Please install it and try again."
        exit 1
    fi
done

# Confirm with the user
echo -e "${RED}WARNING: This script will delete all resources related to the vLLM deployment.${NC}"
echo -e "${RED}This action is irreversible and will result in data loss.${NC}"
read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Store security group IDs for later use
print_section "Retrieving security group IDs"
echo "Getting ALB security group ID..."
ALB_SG=$(kubectl get ingress vllm-deepseek-32b-lws-ingress -o jsonpath='{.metadata.annotations.alb\.ingress\.kubernetes\.io/security-groups}' 2>/dev/null || echo "")
if [ -z "$ALB_SG" ]; then
    print_warning "Could not retrieve ALB security group ID from ingress. Will try to find it later."
fi



echo "Getting FSx security group ID..."
FSX_ID=$(kubectl get pv fsx-lustre-pv -o jsonpath='{.spec.csi.volumeHandle}' 2>/dev/null | cut -d'/' -f1 || echo "")
if [ -n "$FSX_ID" ]; then
    echo "Found FSx filesystem ID: $FSX_ID"
    SG_ID=$(aws fsx describe-file-systems --file-system-id $FSX_ID --query "FileSystems[0].NetworkInterfaceIds[0]" --output text 2>/dev/null | xargs -I{} aws ec2 describe-network-interfaces --network-interface-ids {} --query "NetworkInterfaces[0].Groups[0].GroupId" --output text 2>/dev/null || echo "")
    if [ -n "$SG_ID" ]; then
        echo "Found FSx security group ID: $SG_ID"
    else
        print_warning "Could not retrieve FSx security group ID."
    fi
else
    print_warning "Could not retrieve FSx filesystem ID."
fi

echo "Getting Node security group ID..."
NODE_SG=$(aws ec2 describe-security-groups --filters "Name=tag:aws:cloudformation:logical-id,Values=NodeSecurityGroup" "Name=tag:eks:cluster-name,Values=$CLUSTER_NAME" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")
if [ -n "$NODE_SG" ]; then
    echo "Found Node security group ID: $NODE_SG"
else
    print_warning "Could not retrieve Node security group ID."
fi

echo "Getting VPC ID from the EKS cluster..."
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text 2>/dev/null || echo "")
if [ -n "$VPC_ID" ]; then
    echo "Found VPC ID: $VPC_ID"
    

else
    print_warning "Could not retrieve VPC ID from the EKS cluster."
fi

# 1. Delete Kubernetes Resources
print_section "Deleting Kubernetes Resources"

echo "Deleting vLLM ingress..."
kubectl delete -f vllm-deepseek-32b-lws-ingress.yaml --ignore-not-found
print_success "Ingress deletion initiated"

echo "Waiting 30 seconds for ingress controller to process deletion..."
sleep 30

echo "Deleting vLLM LeaderWorkerSet..."
kubectl delete -f vllm-deepseek-32b-lws.yaml --ignore-not-found
print_success "LeaderWorkerSet deletion initiated"

echo "Waiting 60 seconds for pods to terminate..."
sleep 60

echo "Deleting FSx Lustre PVC..."
kubectl delete -f fsx-lustre-pvc.yaml --ignore-not-found
print_success "PVC deletion initiated"

echo "Waiting 10 seconds for PVC deletion to process..."
sleep 10

echo "Deleting FSx Lustre PV..."
kubectl delete -f fsx-lustre-pv.yaml --ignore-not-found
print_success "PV deletion initiated"

echo "Waiting 10 seconds for PV deletion to process..."
sleep 10

echo "Deleting storage class..."
kubectl delete -f fsx-storage-class.yaml --ignore-not-found
print_success "Storage class deletion initiated"

echo "Deleting AWS Load Balancer Controller..."
helm uninstall aws-load-balancer-controller -n kube-system --ignore-not-found
print_success "AWS Load Balancer Controller deletion initiated"

echo "Waiting 60 seconds for controller termination..."
sleep 60

echo "Verifying all resources are deleted..."
kubectl get pods,svc,ingress,pv,pvc
print_success "Kubernetes resource deletion completed"

# 2. Delete the IAM Service Account CloudFormation Stack
print_section "Deleting IAM Service Account CloudFormation Stack"

STACK_NAME="eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-kube-system-aws-load-balancer-controller"
echo "Deleting CloudFormation stack: $STACK_NAME"
aws cloudformation delete-stack --stack-name $STACK_NAME 2>/dev/null || true

wait_for_deletion "aws cloudformation describe-stacks --stack-name $STACK_NAME" "IAM Service Account CloudFormation Stack" 300
print_success "IAM Service Account CloudFormation Stack deletion completed"

# 3. Delete the IAM Policy
print_section "Deleting IAM Policy"

echo "Getting the ARN of the IAM policy..."
POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='AWSLoadBalancerControllerIAMPolicy'].Arn" --output text)

if [ -n "$POLICY_ARN" ] && [ "$POLICY_ARN" != "None" ]; then
    echo "Deleting IAM policy: $POLICY_ARN"
    aws iam delete-policy --policy-arn $POLICY_ARN
    print_success "IAM policy deleted"
else
    print_warning "IAM policy not found or already deleted"
fi

# 4. Delete the FSx Lustre Filesystem
print_section "Deleting FSx Lustre Filesystem"

if [ -n "$FSX_ID" ]; then
    echo "Deleting FSx Lustre filesystem: $FSX_ID"
    aws fsx delete-file-system --file-system-id $FSX_ID 2>/dev/null || true
    
    wait_for_deletion "aws fsx describe-file-systems --file-system-id $FSX_ID" "FSx Lustre filesystem" 600
    print_success "FSx Lustre filesystem deletion completed"
else
    print_warning "FSx Lustre filesystem ID not found or already deleted"
fi

# 5. Check for Any Remaining Load Balancers
print_section "Checking for Remaining Load Balancers"

echo "Checking for ALBs and NLBs..."
aws elbv2 describe-load-balancers --query "LoadBalancers[?contains(DNSName, '${CLUSTER_NAME}')].LoadBalancerArn" --output text | while read -r lb_arn; do
    if [ -n "$lb_arn" ]; then
        echo "Deleting load balancer: $lb_arn"
        aws elbv2 delete-load-balancer --load-balancer-arn $lb_arn
    fi
done

echo "Checking for Classic ELBs..."
aws elb describe-load-balancers --query "LoadBalancerDescriptions[?contains(DNSName, '${CLUSTER_NAME}')].LoadBalancerName" --output text | while read -r lb_name; do
    if [ -n "$lb_name" ]; then
        echo "Deleting classic load balancer: $lb_name"
        aws elb delete-load-balancer --load-balancer-name $lb_name
    fi
done

print_success "Load balancer cleanup completed"

# 6. Delete the Node Group
print_section "Deleting Node Group"

# Check if node group exists before attempting to delete it
echo "Checking if node group exists: $NODEGROUP_NAME"
if eksctl get nodegroup --cluster=$CLUSTER_NAME --name=$NODEGROUP_NAME --region=$REGION &>/dev/null; then
    echo "Node group exists. Deleting node group: $NODEGROUP_NAME"
    eksctl delete nodegroup --cluster=$CLUSTER_NAME --name=$NODEGROUP_NAME --region=$REGION --drain=false
    
    wait_for_deletion "eksctl get nodegroup --cluster=$CLUSTER_NAME --name=$NODEGROUP_NAME --region=$REGION" "Node group" 1100
    print_success "Node group deletion completed"
else
    print_warning "Node group $NODEGROUP_NAME not found or already deleted"
fi

# 7. Delete the Security Groups
print_section "Deleting Security Groups"

# Delete security groups in the recommended order: FSx SG -> Node SG -> ALB SG

if [ -n "$SG_ID" ]; then
    echo "Deleting FSx security group: $SG_ID"
    aws ec2 delete-security-group --group-id $SG_ID 2>/dev/null || print_warning "Failed to delete FSx security group"
    if [ $? -eq 0 ]; then
        print_success "FSx security group deleted"
    fi
else
    print_warning "FSx security group ID not found or already deleted"
fi

echo "Waiting 30 seconds after FSx security group deletion"
sleep 30

if [ -n "$NODE_SG" ]; then
    echo "Deleting Node security group: $NODE_SG"
    aws ec2 delete-security-group --group-id $NODE_SG 2>/dev/null || print_warning "Failed to delete Node security group"
    if [ $? -eq 0 ]; then
        print_success "Node security group deleted"
    fi
else
    print_warning "Node security group ID not found or already deleted"
fi

echo "Waiting 30 seconds after Node security group deletion"
sleep 30


if [ -n "$ALB_SG" ]; then
    echo "Deleting ALB security group: $ALB_SG"
    aws ec2 delete-security-group --group-id $ALB_SG 2>/dev/null || print_warning "Failed to delete ALB security group"
    if [ $? -eq 0 ]; then
        print_success "ALB security group deleted"
    fi
else
    print_warning "ALB security group ID not found or already deleted"
fi

echo "Waiting 30 seconds after ALB security group deletion"
sleep 30

# 8. Delete the EKS Cluster
print_section "Deleting EKS Cluster"

echo "Deleting EKS cluster: $CLUSTER_NAME"
eksctl delete cluster --name=$CLUSTER_NAME --region=$REGION

wait_for_deletion "aws eks describe-cluster --name $CLUSTER_NAME" "EKS cluster" 1100
print_success "EKS cluster deletion completed"

# 9. Final Verification
print_section "Final Verification"

echo "Checking for any remaining CloudFormation stacks..."
REMAINING_STACKS=$(aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE DELETE_FAILED --query "StackSummaries[?contains(StackName, '${CLUSTER_NAME}')].StackName" --output text)

if [ -n "$REMAINING_STACKS" ]; then
    print_warning "Some CloudFormation stacks still exist:"
    echo "$REMAINING_STACKS"
    echo
    echo "You may need to manually delete these stacks or troubleshoot deletion failures."
    echo "See the README.md section on 'Troubleshooting CloudFormation Stack Deletion Failures'."
else
    print_success "No remaining CloudFormation stacks found"
fi

print_section "Cleanup Complete"
echo "All resources related to the vLLM deployment have been deleted or cleanup has been initiated."
echo "Some AWS resources may still be in the process of being deleted."
echo "Please check the AWS Management Console to verify all resources have been properly cleaned up."