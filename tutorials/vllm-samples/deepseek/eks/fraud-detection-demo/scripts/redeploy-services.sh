#!/bin/bash
set -e

echo "Redeploying all ECS services..."
echo ""

SERVICES="transaction-risk identity-verifier email-alerts fraud-logger geolocation-checker report-generator ui"

for svc in $SERVICES; do
  echo "Updating $svc..."
  /usr/local/bin/aws ecs update-service \
    --cluster fraud-detection-cluster \
    --service $svc \
    --force-new-deployment \
    --region us-west-2 \
    --profile vllm-profile \
    --query 'service.serviceName' \
    --output text
  echo "  ✓ Done"
done

echo ""
echo "========================================================================="
echo "✅ All services redeploying!"
echo "========================================================================="
echo ""
echo "Wait 2-3 minutes, then access your system at:"
echo "http://fraud-detection-alb-1436787808.us-west-2.elb.amazonaws.com"
echo ""
echo "========================================================================="
