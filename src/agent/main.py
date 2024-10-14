# main.py

import torch
from datetime import datetime
from bg_environment import BackgammonEnv
from backgammon_v2 import Player
from agents.ppo_agent import PPOAgent
from training.train import train_agent
from config import (
    HIDDEN_SIZE,
    ENTROPY_COEF_START,
    ENTROPY_COEF_END,
    ENTROPY_ANNEAL_EPISODES,
    S3_LOG_PREFIX,
)
from custom.record_writer import record_writer  # Ensure custom RecordWriter is loaded

if __name__ == "__main__":
    # Initialize the environment
    env = BackgammonEnv(match_length=5)
    OBSERVATION_SPACE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n
    MAX_LEGAL_MOVES = env.max_legal_moves  # e.g., 200
    MOVE_FEATURE_LENGTH = env.move_feature_length  # e.g., 12

    # Get the S3 bucket name from environment variable or set a default
    s3_bucket_name = "bgppomodels"
    # s3_bucket_name = None

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'}")

    # Initialize the agent
    agent = PPOAgent(
        board_feature_size=198,
        move_feature_size=MOVE_FEATURE_LENGTH,
        hidden_size=HIDDEN_SIZE,
        action_size=ACTION_SIZE,
        entropy_coef_start=ENTROPY_COEF_START,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_anneal_episodes=ENTROPY_ANNEAL_EPISODES,
        log_dir=None,  # Set to None to use S3 for logs
        s3_bucket_name=s3_bucket_name,
        s3_log_prefix=S3_LOG_PREFIX,
    )

    # Optionally load a saved model
    # To load from S3
    # agent.load_model("ppo_backgammon_episode_500500.pth", from_s3=True, training=True)

    # Set agent to training mode
    agent.set_training_mode(training=True)
    # Train the agent
    train_agent(env, agent)
