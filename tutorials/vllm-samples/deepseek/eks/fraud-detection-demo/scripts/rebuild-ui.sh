#!/bin/bash
set -e

echo "Rebuilding UI with debug output..."
cd /home/vsnak/riv-aim352-chalktalk/ui

docker build -t fraud-detection/ui:latest .
docker tag fraud-detection/ui:latest 239857122763.dkr.ecr.us-west-2.amazonaws.com/fraud-detection/ui:latest
docker push 239857122763.dkr.ecr.us-west-2.amazonaws.com/fraud-detection/ui:latest

echo ""
echo "Redeploying UI service..."
/usr/local/bin/aws ecs update-service \
  --cluster fraud-detection-cluster \
  --service ui \
  --force-new-deployment \
  --region us-west-2 \
  --profile vllm-profile \
  --query 'service.serviceName' \
  --output text

echo ""
echo "========================================================================="
echo "✅ UI rebuilt and redeploying with debug output!"
echo "========================================================================="
echo "Wait 1-2 minutes, then test at:"
echo "http://fraud-detection-alb-1436787808.us-west-2.elb.amazonaws.com"
echo ""
echo "The debug output will show:"
echo "  - Response type"
echo "  - Response length"
echo "  - This will help us fix the empty display issue"
echo "========================================================================="
