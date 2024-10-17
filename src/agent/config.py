import os

# Hyperparameters
NUM_ENVS = 1
NUM_UPDATES = 10_000  # Adjust as needed
T_HORIZON = 2048  # Number of steps to collect before an update
NUM_EPOCHS = 4
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPS_CLIP = 0.20
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF_START = 0.10
ENTROPY_COEF_END = 0.01
ENTROPY_ANNEAL_EPISODES = 500_000
MAX_TIMESTEPS = 500

# S3 Configuration
S3_BUCKET_NAME = "bgppomodels"
S3_MODEL_PREFIX = "models/"
S3_LOG_PREFIX = "logs/"
