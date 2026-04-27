#!/bin/bash
# Image generation via OpenAI-compatible /v1/images/generations endpoint
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red apple on a white table", "size": "512x512", "n": 1}'
