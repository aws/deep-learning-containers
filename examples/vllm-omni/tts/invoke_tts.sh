#!/bin/bash
# Text-to-speech via OpenAI-compatible /v1/audio/speech endpoint
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?", "voice": "vivian", "language": "English"}' \
  --output speech.wav
