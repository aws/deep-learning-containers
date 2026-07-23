#!/bin/bash
  
# Choose the appropriate image based on your instance type
# For CPU
IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310-ubuntu22.04-ec2-v1.30"

# Download abalone dataset if it doesn't exist
echo "Checking for abalone dataset..."
if [ ! -f "./data/abalone.data" ]; then
    echo "Downloading abalone dataset..."
    curl -o ./data/abalone.data https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
    echo "Dataset downloaded successfully."
else
    echo "Dataset already exists, skipping download."
fi

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# For GPU (uncomment if using GPU instance)
# IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-gpu-py39-cuda11.2-ubuntu20.04-ec2"

# Run the container
docker run --rm \
  -v $(pwd):/opt/ml/code \
  -v $(pwd)/output:/opt/ml/model \
  -v $(pwd)/data:/opt/ml/data \
  ${IMAGE} \
  bash -c "pip install -r /opt/ml/code/requirements.txt && python /opt/ml/code/src/train.py"
  