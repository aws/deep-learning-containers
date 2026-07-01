#!/bin/bash
set -e

# Configuration
PROFILE="vllm-profile"
REGION="us-west-2"
CLUSTER_NAME="fraud-detection-cluster"
VPC_ID="vpc-0e4bacb3d48f425d3"
PRIVATE_SUBNETS="subnet-009cb0c756d91d094,subnet-0663244be68d7a5cb,subnet-0eadd502700500bbb"
MCP_SG="sg-089ccdfacad3456ea"
UI_SG="sg-06210836510399012"
ALB_SG="sg-00d988b0ef59fb774"
NAMESPACE_ID="ns-4vhyfr36fdygm5zs"
ACCOUNT_ID="239857122763"
VLLM_ENDPOINT="http://k8s-default-vllmdeep-066fc65d38-1002306241.us-west-2.elb.amazonaws.com"

echo "========================================================================"
echo "Deploying Fraud Detection System to ECS"
echo "========================================================================"
echo "vLLM Endpoint: $VLLM_ENDPOINT"
echo "========================================================================"

# Step 1: Create IAM execution role for ECS tasks
echo
echo "Step 1: Creating IAM execution role..."

ROLE_NAME="ecsTaskExecutionRole-fraud-detection"

# Check if role exists
if ! /usr/local/bin/aws iam get-role --role-name $ROLE_NAME --profile $PROFILE 2>/dev/null; then
  # Create trust policy
  cat > /tmp/ecs-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

  # Create role
  /usr/local/bin/aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file:///tmp/ecs-trust-policy.json \
    --profile $PROFILE
  
  # Attach managed policy
  /usr/local/bin/aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    --profile $PROFILE
  
  echo "✓ IAM role created"
else
  echo "✓ IAM role already exists"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Step 2: Deploy MCP Servers
echo
echo "Step 2: Deploying 6 MCP servers..."

MCP_SERVICES=("transaction-risk" "identity-verifier" "email-alerts" "fraud-logger" "geolocation-checker" "report-generator")

for SERVICE in "${MCP_SERVICES[@]}"; do
  echo
  echo "Deploying: $SERVICE"
  
  # Create task definition
  cat > /tmp/${SERVICE}-task.json <<EOF
{
  "family": "fraud-detection-${SERVICE}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "${ROLE_ARN}",
  "containerDefinitions": [{
    "name": "${SERVICE}",
    "image": "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/fraud-detection/${SERVICE}:latest",
    "essential": true,
    "portMappings": [{
      "containerPort": 8080,
      "protocol": "tcp"
    }],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/fraud-detection",
        "awslogs-region": "${REGION}",
        "awslogs-stream-prefix": "${SERVICE}",
        "awslogs-create-group": "true"
      }
    }
  }]
}
EOF

  # Register task definition
  TASK_ARN=$(/usr/local/bin/aws ecs register-task-definition \
    --cli-input-json file:///tmp/${SERVICE}-task.json \
    --region $REGION \
    --profile $PROFILE \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)
  
  echo "  Task definition: $TASK_ARN"
  
  # Create service with Cloud Map
  /usr/local/bin/aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE \
    --task-definition $TASK_ARN \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$MCP_SG]}" \
    --service-registries "registryArn=arn:aws:servicediscovery:${REGION}:${ACCOUNT_ID}:service/srv-$(uuidgen | cut -d'-' -f1)" \
    --region $REGION \
    --profile $PROFILE 2>/dev/null || echo "  Service may already exist"
  
  echo "  ✓ $SERVICE deployed"
done

# Step 3: Get public subnets for ALB
echo
echo "Step 3: Getting public subnets for ALB..."

PUBLIC_SUBNETS=$(/usr/local/bin/aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query 'Subnets[].SubnetId' \
  --output text \
  --profile $PROFILE \
  --region $REGION | tr '\t' ',')

if [ -z "$PUBLIC_SUBNETS" ]; then
  echo "❌ No public subnets found! Using private subnets (this won't work for internet-facing ALB)"
  PUBLIC_SUBNETS=$PRIVATE_SUBNETS
fi

echo "Public subnets: $PUBLIC_SUBNETS"

# Step 4: Create ALB
echo
echo "Step 4: Creating Application Load Balancer..."

ALB_ARN=$(/usr/local/bin/aws elbv2 create-load-balancer \
  --name fraud-detection-alb \
  --subnets $(echo $PUBLIC_SUBNETS | tr ',' ' ') \
  --security-groups $ALB_SG \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4 \
  --region $REGION \
  --profile $PROFILE \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text 2>/dev/null || \
  /usr/local/bin/aws elbv2 describe-load-balancers \
    --names fraud-detection-alb \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text \
    --profile $PROFILE \
    --region $REGION)

echo "ALB ARN: $ALB_ARN"

# Get ALB DNS name
ALB_DNS=$(/usr/local/bin/aws elbv2 describe-load-balancers \
  --load-balancer-arns $ALB_ARN \
  --query 'LoadBalancers[0].DNSName' \
  --output text \
  --profile $PROFILE \
  --region $REGION)

echo "ALB DNS: $ALB_DNS"

# Step 5: Create target group
echo
echo "Step 5: Creating target group..."

TG_ARN=$(/usr/local/bin/aws elbv2 create-target-group \
  --name fraud-detection-ui-tg \
  --protocol HTTP \
  --port 8501 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path / \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --region $REGION \
  --profile $PROFILE \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text 2>/dev/null || \
  /usr/local/bin/aws elbv2 describe-target-groups \
    --names fraud-detection-ui-tg \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text \
    --profile $PROFILE \
    --region $REGION)

echo "Target Group: $TG_ARN"

# Step 6: Create listener
echo
echo "Step 6: Creating ALB listener..."

/usr/local/bin/aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Listener may already exist"

# Step 7: Deploy UI
echo
echo "Step 7: Deploying UI with environment variables..."

cat > /tmp/ui-task.json <<EOF
{
  "family": "fraud-detection-ui",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "${ROLE_ARN}",
  "containerDefinitions": [{
    "name": "ui",
    "image": "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/fraud-detection/ui:latest",
    "essential": true,
    "portMappings": [{
      "containerPort": 8501,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "VLLM_ENDPOINT", "value": "${VLLM_ENDPOINT}"},
      {"name": "MCP_TRANSACTION_RISK_URL", "value": "http://transaction-risk.fraud-detection.local:8080/sse"},
      {"name": "MCP_IDENTITY_VERIFIER_URL", "value": "http://identity-verifier.fraud-detection.local:8080/sse"},
      {"name": "MCP_EMAIL_ALERTS_URL", "value": "http://email-alerts.fraud-detection.local:8080/sse"},
      {"name": "MCP_FRAUD_LOGGER_URL", "value": "http://fraud-logger.fraud-detection.local:8080/sse"},
      {"name": "MCP_GEOLOCATION_URL", "value": "http://geolocation-checker.fraud-detection.local:8080/sse"},
      {"name": "MCP_REPORT_GENERATOR_URL", "value": "http://report-generator.fraud-detection.local:8080/sse"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/fraud-detection",
        "awslogs-region": "${REGION}",
        "awslogs-stream-prefix": "ui",
        "awslogs-create-group": "true"
      }
    }
  }]
}
EOF

UI_TASK_ARN=$(/usr/local/bin/aws ecs register-task-definition \
  --cli-input-json file:///tmp/ui-task.json \
  --region $REGION \
  --profile $PROFILE \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "UI Task definition: $UI_TASK_ARN"

# Create UI service
/usr/local/bin/aws ecs create-service \
  --cluster $CLUSTER_NAME \
  --service-name ui \
  --task-definition $UI_TASK_ARN \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$UI_SG]}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=ui,containerPort=8501" \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "UI service may already exist"

echo "✓ UI deployed"

echo
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo
echo "🌐 Access your application at:"
echo "   http://$ALB_DNS"
echo
echo "📊 System Components:"
echo "   - vLLM DeepSeek R1 on EKS: $VLLM_ENDPOINT"
echo "   - 6 MCP Servers on ECS (Fargate)"
echo "   - Streamlit UI on ECS (Fargate)"
echo "   - Application Load Balancer"
echo
echo "Wait 2-3 minutes for services to start, then access the UI!"
echo "========================================================================"
