#!/bin/bash
set -e

echo "Starting MCP servers locally..."
echo ""

# Build MCP image
cd /home/vsnak/riv-aim352-chalktalk/mcp-servers
cp requirements.txt transaction-risk/
cp Dockerfile transaction-risk/
docker build -t mcp-server:test transaction-risk/
rm transaction-risk/requirements.txt transaction-risk/Dockerfile

# Start 6 MCP servers
echo "Starting transaction-risk on port 8081..."
docker run -d --name mcp-transaction-risk -p 8081:8080 mcp-server:test

echo "Starting identity-verifier on port 8082..."
docker run -d --name mcp-identity-verifier -p 8082:8080 mcp-server:test

echo "Starting email-alerts on port 8083..."
docker run -d --name mcp-email-alerts -p 8083:8080 mcp-server:test

echo "Starting fraud-logger on port 8084..."
docker run -d --name mcp-fraud-logger -p 8084:8080 mcp-server:test

echo "Starting geolocation-checker on port 8085..."
docker run -d --name mcp-geolocation -p 8085:8080 mcp-server:test

echo "Starting report-generator on port 8086..."
docker run -d --name mcp-report-generator -p 8086:8080 mcp-server:test

echo ""
echo "Waiting for servers to start..."
sleep 5

echo ""
echo "Starting UI connected to local MCPs..."
cd /home/vsnak/riv-aim352-chalktalk/ui

docker run -d --name ui-test-local \
  -p 8501:8501 \
  -e VLLM_ENDPOINT="http://k8s-default-vllmdeep-066fc65d38-1002306241.us-west-2.elb.amazonaws.com" \
  -e MCP_TRANSACTION_RISK_URL="http://host.docker.internal:8081/sse" \
  -e MCP_IDENTITY_VERIFIER_URL="http://host.docker.internal:8082/sse" \
  -e MCP_EMAIL_ALERTS_URL="http://host.docker.internal:8083/sse" \
  -e MCP_FRAUD_LOGGER_URL="http://host.docker.internal:8084/sse" \
  -e MCP_GEOLOCATION_URL="http://host.docker.internal:8085/sse" \
  -e MCP_REPORT_GENERATOR_URL="http://host.docker.internal:8086/sse" \
  fraud-detection/ui:test

echo ""
echo "========================================================================="
echo "✅ Local test stack running!"
echo "========================================================================="
echo ""
echo "MCP Servers:"
echo "  - transaction-risk: http://localhost:8081/sse"
echo "  - identity-verifier: http://localhost:8082/sse"
echo "  - email-alerts: http://localhost:8083/sse"
echo "  - fraud-logger: http://localhost:8084/sse"
echo "  - geolocation-checker: http://localhost:8085/sse"
echo "  - report-generator: http://localhost:8086/sse"
echo ""
echo "UI: http://localhost:8501"
echo ""
echo "========================================================================="
echo ""
echo "To stop all:"
echo "  docker stop ui-test-local mcp-transaction-risk mcp-identity-verifier mcp-email-alerts mcp-fraud-logger mcp-geolocation mcp-report-generator"
echo "  docker rm ui-test-local mcp-transaction-risk mcp-identity-verifier mcp-email-alerts mcp-fraud-logger mcp-geolocation mcp-report-generator"
