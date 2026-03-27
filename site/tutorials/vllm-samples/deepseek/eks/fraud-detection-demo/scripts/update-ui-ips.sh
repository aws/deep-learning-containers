#!/bin/bash
set -e

echo "Creating UI task definition with updated MCP server IPs..."

cat > /tmp/ui-task-def-v3.json << 'EOFX'
{
  "family": "fraud-detection-ui",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::239857122763:role/ecsTaskExecutionRole",
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
EOFX

echo "Registering task definition..."
/usr/local/bin/aws ecs register-task-definition \
  --cli-input-json file:///tmp/ui-task-def-v3.json \
  --region us-west-2 \
  --profile vllm-profile \
  --query 'taskDefinition.revision' \
  --output text

echo ""
echo "Updating UI service..."
/usr/local/bin/aws ecs update-service \
  --cluster fraud-detection-cluster \
  --service ui \
  --task-definition fraud-detection-ui:3 \
  --force-new-deployment \
  --region us-west-2 \
  --profile vllm-profile \
  --query 'service.serviceName' \
  --output text

echo ""
echo "========================================================================="
echo "✅ UI updated with NEW MCP server IPs!"
echo "========================================================================="
echo ""
echo "NEW IP addresses:"
echo "  - transaction-risk: 192.168.177.13:8080"
echo "  - identity-verifier: 192.168.150.184:8080"
echo "  - email-alerts: 192.168.174.170:8080"
echo "  - fraud-logger: 192.168.110.66:8080"
echo "  - geolocation-checker: 192.168.155.26:8080"
echo "  - report-generator: 192.168.154.51:8080"
echo ""
echo "Wait 1-2 minutes, then test at:"
echo "http://fraud-detection-alb-1436787808.us-west-2.elb.amazonaws.com"
echo "========================================================================="
