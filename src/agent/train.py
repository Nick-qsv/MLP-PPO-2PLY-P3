import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import os
import boto3
from botocore.config import Config
from tqdm import tqdm  # For progress bar

# Import your environment and agent classes
# Assuming they are defined in the same script or properly imported
# from your_module import VectorizedBackgammonEnv, BackgammonPPOAgent

# Hyperparameters (some are redefined here for clarity)
NUM_ENVS = 50
NUM_UPDATES = 10_000  # Adjust as needed
T_HORIZON = 2048  # Number of steps to collect before an update
NUM_EPOCHS = 4
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPS_CLIP = 0.25
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF_START = 0.20
ENTROPY_COEF_END = 0.01
ENTROPY_ANNEAL_EPISODES = 600_000
MAX_TIMESTEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the vectorized environment and the PPO agent
envs = VectorizedBackgammonEnv(num_envs=NUM_ENVS)
agent = BackgammonPPOAgent(
    action_size=envs.action_space.n,
    entropy_coef_start=ENTROPY_COEF_START,
    entropy_coef_end=ENTROPY_COEF_END,
    entropy_anneal_episodes=ENTROPY_ANNEAL_EPISODES,
    device=device,
)

# Training loop
total_episodes = 0
total_steps = 0

for update in tqdm(range(NUM_UPDATES), desc="Training Progress"):
    observations = envs.reset()  # Shape: (NUM_ENVS, 198)
    agent.memory = []  # Clear memory at the start of each update

    for step in range(T_HORIZON):
        # Get action masks and legal moves
        action_masks = envs.get_action_masks()  # Shape: (NUM_ENVS, max_legal_moves)
        observations_tensor = torch.tensor(
            observations, dtype=torch.float32, device=device
        )
        action_masks_tensor = torch.tensor(
            action_masks, dtype=torch.float32, device=device
        )

        # Get actions from agent
        actions, action_log_probs, state_values = agent.select_action(
            observations_tensor, action_masks_tensor
        )
        actions_np = actions.cpu().numpy()

        # Step the environment
        next_observations, rewards, dones, infos = envs.step(actions_np)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

        # Store experiences in agent's memory
        agent.memory.append(
            {
                "observations": observations_tensor,
                "action_masks": action_masks_tensor,
                "actions": actions,
                "action_log_probs": action_log_probs,
                "state_values": state_values,
                "rewards": rewards_tensor,
                "dones": dones_tensor,
            }
        )

        observations = next_observations
        total_steps += NUM_ENVS

        # Check if any environments are done
        if np.any(dones):
            total_episodes += np.sum(dones)

    # After T_HORIZON steps, optimize the model
    agent.optimize_model()

    # Optionally log progress to TensorBoard
    agent.writer.add_scalar("Loss/Policy Loss", agent.last_policy_loss, update)
    agent.writer.add_scalar("Loss/Value Loss", agent.last_value_loss, update)
    agent.writer.add_scalar("Loss/Entropy", agent.last_entropy_loss, update)
    agent.writer.add_scalar("Stats/Total Episodes", total_episodes, update)
    agent.writer.add_scalar("Stats/Total Steps", total_steps, update)
    agent.writer.add_scalar("Stats/Entropy Coefficient", agent.entropy_coef, update)

    # Save model periodically (e.g., every 100 updates)
    if update % 100 == 0 and update > 0:
        model_save_path = f"backgammon_ppo_update_{update}.pt"
        torch.save(agent.policy_network.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # If using S3, upload the model
        if agent.s3_bucket_name:
            s3_key = f"{agent.s3_model_prefix}backgammon_ppo_update_{update}.pt"
            agent.s3_client.upload_file(model_save_path, agent.s3_bucket_name, s3_key)
            print(f"Model uploaded to s3://{agent.s3_bucket_name}/{s3_key}")

# Close environments and writer
envs.close()
agent.writer.close()
