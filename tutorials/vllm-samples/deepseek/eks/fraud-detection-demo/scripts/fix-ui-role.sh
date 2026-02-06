#!/bin/bash
set -e

echo "Registering task definition v4 with correct IAM role..."

cat > /tmp/ui-v4.json << 'ENDOFJSON'
{
  "family": "fraud-detection-ui",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::239857122763:role/ecsTaskExecutionRole-fraud-detection",
  "containerDefinitions": [
    {
      "name": "ui",
      "image": "239857122763.dkr.ecr.us-west-2.amazonaws.com/fraud-detection/ui:latest",
      "essential": true,
      "portMappings": [{"containerPort": 8501, "protocol": "tcp"}],
      "environment": [
        {"name": "VLLM_ENDPOINT", "value": "http://k8s-default-vllmdeep-066fc65d38-1002306241.us-west-2.elb.amazonaws.com"},
        {"name": "MCP_TRANSACTION_RISK_URL", "value": "http://192.168.177.13:8080/sse"},
        {"name": "MCP_IDENTITY_VERIFIER_URL", "value": "http://192.168.150.184:8080/sse"},
        {"name": "MCP_EMAIL_ALERTS_URL", "value": "http://192.168.174.170:8080/sse"},
        {"name": "MCP_FRAUD_LOGGER_URL", "value": "http://192.168.110.66:8080/sse"},
        {"name": "MCP_GEOLOCATION_URL", "value": "http://192.168.155.26:8080/sse"},
        {"name": "MCP_REPORT_GENERATOR_URL", "value": "http://192.168.154.51:8080/sse"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fraud-detection",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ui"
        }
      }
    }
  ]
}
ENDOFJSON

/usr/local/bin/aws ecs register-task-definition \
  --cli-input-json file:///tmp/ui-v4.json \
  --region us-west-2 \
  --profile vllm-profile \
  --query 'taskDefinition.revision' \
  --output text

echo ""
echo "Updating service to use v4..."

/usr/local/bin/aws ecs update-service \
  --cluster fraud-detection-cluster \
  --service ui \
  --task-definition fraud-detection-ui:4 \
  --force-new-deployment \
  --region us-west-2 \
  --profile vllm-profile \
  --query 'service.serviceName' \
  --output text

echo ""
echo "========================================================================="
echo "✅ UI service updated with CORRECT IAM role and NEW MCP IPs!"
echo "========================================================================="
echo "Wait 1-2 minutes for task to start, then test at:"
echo "http://fraud-detection-alb-1436787808.us-west-2.elb.amazonaws.com"
echo "========================================================================="
