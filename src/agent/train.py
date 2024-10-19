import torch
import numpy as np
import boto3
import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from botocore.config import Config
from tqdm import tqdm  # For progress bar
from config import *
from src.environment import VectorizedBackgammonEnv
from ppo_agent import BackgammonPPOAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_agent():
    global total_episodes, total_steps
    recent_rewards = []
    recent_wins = []
    episode_rewards = torch.zeros(NUM_ENVS, device=device)
    episode_lengths = torch.zeros(NUM_ENVS, device=device)

    for update in tqdm(range(NUM_UPDATES), desc="Training Progress"):
        observations = envs.reset()  # Already a tensor on device
        agent.memory = []  # Clear memory at the start of each update

        for step in range(T_HORIZON):
            # Get action masks
            action_masks = envs.get_action_masks()  # Already a tensor on device

            # Get actions from agent
            actions = agent.select_action(observations, action_masks)

            # Step the environment
            next_observations, rewards, dones, infos = envs.step(actions)

            # Update per-environment episode rewards and lengths
            episode_rewards += rewards
            episode_lengths += 1

            # Store rewards and dones in agent's memory
            for i in range(NUM_ENVS):
                agent.memory[-NUM_ENVS + i]["reward"] = rewards[i].unsqueeze(0)
                agent.memory[-NUM_ENVS + i]["done"] = dones[i].unsqueeze(0)

            observations = next_observations
            total_steps += NUM_ENVS

            # Handle episodes ending
            for i in range(NUM_ENVS):
                if dones[i]:
                    total_episodes += 1
                    # Record the episode reward and win
                    recent_rewards.append(episode_rewards[i].item())
                    info = infos[i]
                    if (
                        "winner" in info
                        and info["winner"] == infos[i]["current_player"]
                    ):
                        recent_wins.append(1)
                    else:
                        recent_wins.append(0)

                    # Reset per-environment episode data
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

                    # Log metrics every 1000 episodes
                    if total_episodes % 10 == 0:
                        avg_reward = np.mean(recent_rewards[-1000:])
                        win_rate = np.mean(recent_wins[-1000:])
                        agent.win_rates.append(win_rate)
                        agent.log_metrics(total_episodes, avg_reward, win_rate)

        # After T_HORIZON steps, optimize the model
        agent.update()

        # Optionally log progress to TensorBoard
        agent.writer.add_scalar("Loss/Policy Loss", agent.last_policy_loss, update)
        agent.writer.add_scalar("Loss/Value Loss", agent.last_value_loss, update)
        agent.writer.add_scalar("Loss/Entropy", agent.last_entropy_loss, update)
        agent.writer.add_scalar("Loss/Total Loss", agent.last_total_loss, update)
        agent.writer.add_scalar("Stats/Total Episodes", total_episodes, update)
        agent.writer.add_scalar("Stats/Total Steps", total_steps, update)
        agent.writer.add_scalar("Stats/Entropy Coefficient", agent.entropy_coef, update)

        # Save model periodically
        if update % 50 == 0 and update > 0:
            filename = f"backgammon_ppo_update_{update}.pth"
            agent.save_model(filename=filename, to_s3=True)
            agent.save_model(filename="ppo_backgammon_s3.pth", to_s3=True)

    # Close environments and writer
    envs.close()
    agent.writer.close()


if __name__ == "__main__":
    # Initialize the vectorized environment and the PPO agent
    print("Initializing VectorizedBackgammonEnv...")
    envs = VectorizedBackgammonEnv(num_envs=NUM_ENVS, device=device)
    print("VectorizedBackgammonEnv initialized.")
    agent = BackgammonPPOAgent(
        action_size=envs.action_space.n,
        entropy_coef_start=ENTROPY_COEF_START,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_anneal_episodes=ENTROPY_ANNEAL_EPISODES,
        device=device,
        s3_bucket_name=S3_BUCKET_NAME,
        s3_model_prefix=S3_MODEL_PREFIX,
        s3_log_prefix=S3_LOG_PREFIX,
    )

    # Initialize counters
    total_episodes = 0
    total_steps = 0

    # Optionally load a saved model
    # agent.load_model(filename="ppo_backgammon_s3.pth", from_s3=True)

    # Set agent to training mode
    agent.set_training_mode(training=True)

    # Start the training loop
    train_agent()
