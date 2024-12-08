#!/bin/bash

# Start Ollama service
nohup ollama serve > ollama.log 2>&1 &

# Wait for Ollama to start
sleep 10

# Pull the model
ollama pull eas/dragon-mistral-v0

# Start the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000