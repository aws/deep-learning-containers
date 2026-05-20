#!/bin/bash
set -e

PROFILE="vllm-profile"
REGION="us-west-2"
ACCOUNT_ID=239857122763
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "===================================================================="
echo "Building and Pushing Docker Images to ECR"
echo "===================================================================="
echo "Registry: $REGISTRY"
echo "===================================================================="

# Login to ECR
echo
echo "Logging in to ECR..."
/usr/local/bin/aws ecr get-login-password --region $REGION --profile $PROFILE | \
  docker login --username AWS --password-stdin $REGISTRY

# Build and push MCP servers
echo
echo "Building MCP servers..."
cd mcp-servers

for service in transaction-risk identity-verifier email-alerts fraud-logger geolocation-checker report-generator; do
  echo
  echo "===================================================================="
  echo "Building: $service"
  echo "===================================================================="
  
  # Create build context
  cp Dockerfile $service/
  cp requirements.txt $service/
  
  # Build
  docker build -t fraud-detection/$service:latest $service/
  
  # Tag
  docker tag fraud-detection/$service:latest $REGISTRY/fraud-detection/$service:latest
  
  # Push
  docker push $REGISTRY/fraud-detection/$service:latest
  
  # Cleanup
  rm $service/Dockerfile $service/requirements.txt
  
  echo "✓ $service pushed successfully"
done

cd ..

# Build and push UI
echo
echo "===================================================================="
echo "Building: ui"
echo "===================================================================="
cd ui
docker build -t fraud-detection/ui:latest .
docker tag fraud-detection/ui:latest $REGISTRY/fraud-detection/ui:latest
docker push $REGISTRY/fraud-detection/ui:latest
echo "✓ ui pushed successfully"

cd ..

echo
echo "===================================================================="
echo "All images built and pushed successfully!"
echo "===================================================================="
echo
echo "Images available:"
for service in transaction-risk identity-verifier email-alerts fraud-logger geolocation-checker report-generator ui; do
  echo "  - $REGISTRY/fraud-detection/$service:latest"
done
echo "===================================================================="
