#!/bin/bash

# Define the Docker image name and tag
IMAGE_NAME="mlp_ppo_v1"
IMAGE_TAG="latest"

# Define the test script path inside the container
TEST_SCRIPT="./src/testing/test_env.py"

# Run the Docker container and execute the test script
echo "Running tests inside the Docker container..."
docker run --rm \
    -v "$(pwd)":/app \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    python3 ${TEST_SCRIPT}

# Check if the test script ran successfully
if [ $? -eq 0 ]; then
    echo "Environment tests ran successfully."
else
    echo "Environment tests failed."
    exit 1
fi
