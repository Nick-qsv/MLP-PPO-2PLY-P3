#!/bin/bash

# Define the Docker image name and tag
IMAGE_NAME="backgammon_ai_rl_ppoagent"
IMAGE_TAG="latest"

# Build the Docker image
echo "Building Docker image '$IMAGE_NAME:$IMAGE_TAG'..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME:$IMAGE_TAG' built successfully."
else
    echo "Docker image build failed."
    exit 1
fi
