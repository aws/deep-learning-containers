#!/bin/bash
set -e

# Configuration
PROFILE="vllm-profile"
REGION="us-west-2"
CLUSTER_NAME="fraud-detection-cluster"
VPC_ID="vpc-0e4bacb3d48f425d3"
PRIVATE_SUBNETS="subnet-009cb0c756d91d094,subnet-0663244be68d7a5cb,subnet-0eadd502700500bbb"
PUBLIC_SUBNETS="" # Get public subnets if needed for ALB
NAMESPACE="fraud-detection.local"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile $PROFILE)
VLLM_ENDPOINT="${VLLM_ENDPOINT:-}" # Set this to your DeepSeek EKS endpoint

echo "===================================================================="
echo "Financial Fraud Detection System - ECS Deployment"
echo "===================================================================="
echo "VPC: $VPC_ID"
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "===================================================================="

# Step 1: Create ECS Cluster
echo
echo "Step 1: Creating ECS Cluster..."
aws ecs create-cluster \
  --cluster-name $CLUSTER_NAME \
  --region $REGION \
  --profile $PROFILE \
  --capacity-providers FARGATE FARGATE_SPOT \
  --default-capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
  2>/dev/null || echo "Cluster already exists"

# Step 2: Create Cloud Map Namespace for Service Discovery
echo
echo "Step 2: Creating Cloud Map namespace..."
NAMESPACE_ID=$(aws servicediscovery create-private-dns-namespace \
  --name $NAMESPACE \
  --vpc $VPC_ID \
  --region $REGION \
  --profile $PROFILE \
  --query 'OperationId' \
  --output text 2>/dev/null || echo "")

if [ ! -z "$NAMESPACE_ID" ]; then
  echo "Waiting for namespace creation..."
  sleep 10
  NAMESPACE_ID=$(aws servicediscovery list-namespaces \
    --filters Name=TYPE,Values=DNS_PRIVATE \
    --query "Namespaces[?Name=='$NAMESPACE'].Id" \
    --output text \
    --profile $PROFILE \
    --region $REGION)
fi

echo "Cloud Map Namespace ID: $NAMESPACE_ID"

# Step 3: Get Your IP for ALB Security Group
echo
echo "Step 3: Getting your IP address..."
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Your IP: $MY_IP"

# Step 4: Create Security Groups
echo
echo "Step 4: Creating Security Groups..."

# ALB Security Group
ALB_SG=$(aws ec2 create-security-group \
  --group-name fraud-detection-alb-sg \
  --description "Security group for fraud detection ALB" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --profile $PROFILE \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=fraud-detection-alb-sg" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --profile $PROFILE \
    --region $REGION)

echo "ALB Security Group: $ALB_SG"

# Add ingress rule for YOUR IP only (no 0.0.0.0/0!)
aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG \
  --protocol tcp \
  --port 80 \
  --cidr ${MY_IP}/32 \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Rule already exists"

# ECS UI Security Group
UI_SG=$(aws ec2 create-security-group \
  --group-name fraud-detection-ui-sg \
  --description "Security group for fraud detection UI" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --profile $PROFILE \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=fraud-detection-ui-sg" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --profile $PROFILE \
    --region $REGION)

echo "UI Security Group: $UI_SG"

# Allow ALB to UI on port 8501
aws ec2 authorize-security-group-ingress \
  --group-id $UI_SG \
  --protocol tcp \
  --port 8501 \
  --source-group $ALB_SG \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Rule already exists"

# MCP Servers Security Group
MCP_SG=$(aws ec2 create-security-group \
  --group-name fraud-detection-mcp-sg \
  --description "Security group for MCP servers" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --profile $PROFILE \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=fraud-detection-mcp-sg" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --profile $PROFILE \
    --region $REGION)

echo "MCP Security Group: $MCP_SG"

# Allow UI to MCP servers on port 8080
aws ec2 authorize-security-group-ingress \
  --group-id $MCP_SG \
  --protocol tcp \
  --port 8080 \
  --source-group $UI_SG \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Rule already exists"

# Allow MCP servers to talk to each other
aws ec2 authorize-security-group-ingress \
  --group-id $MCP_SG \
  --protocol tcp \
  --port 8080 \
  --source-group $MCP_SG \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Rule already exists"

echo
echo "===================================================================="
echo "Security Groups Created (NO 0.0.0.0/0 rules!)"
echo "ALB SG: $ALB_SG (allows ${MY_IP}/32 only)"
echo "UI SG: $UI_SG (allows ALB only)"
echo "MCP SG: $MCP_SG (allows UI and self only)"
echo "===================================================================="

# Step 5: Create ECR Repositories
echo
echo "Step 5: Creating ECR repositories..."
for service in transaction-risk identity-verifier email-alerts fraud-logger geolocation-checker report-generator ui; do
  aws ecr create-repository \
    --repository-name fraud-detection/$service \
    --region $REGION \
    --profile $PROFILE 2>/dev/null || echo "Repository fraud-detection/$service already exists"
done

echo
echo "===================================================================="
echo "Next Steps:"
echo "===================================================================="
echo "1. Build and push Docker images:"
echo "   ./build-and-push-images.sh"
echo
echo "2. Create ECS task definitions and services:"
echo "   ./deploy-ecs-services.sh"
echo
echo "3. Update UI with service discovery URLs"
echo "   Service URLs will be: http://<service-name>.$NAMESPACE:8080/sse"
echo
echo "Environment variables to set:"
echo "  VPC_ID=$VPC_ID"
echo "  PRIVATE_SUBNETS=$PRIVATE_SUBNETS"
echo "  ALB_SG=$ALB_SG"
echo "  UI_SG=$UI_SG"
echo "  MCP_SG=$MCP_SG"
echo "  NAMESPACE_ID=$NAMESPACE_ID"
echo "  VLLM_ENDPOINT=$VLLM_ENDPOINT"
echo "===================================================================="
