import os
import io
import numpy as np
import boto3
import botocore
from botocore.config import Config
from tensorboardX import record_writer
from tensorboardX.record_writer import S3RecordWriter
import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
from src.agent.policy_network import BackgammonPolicyNetwork
from src.agent.config import (
    HIDDEN_SIZE,
    LEARNING_RATE,
    GAMMA,
    EPS_CLIP,
    NUM_EPOCHS,
    VALUE_LOSS_COEF,
    ENTROPY_COEF_START,
    ENTROPY_COEF_END,
    ENTROPY_ANNEAL_EPISODES,
)
from datetime import datetime  # pylint: disable=import-error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Save the original RecordWriter
original_RecordWriter = record_writer.RecordWriter


class MyS3RecordWriter(S3RecordWriter):
    def __init__(self, logdir, *args, **kwargs):
        super(MyS3RecordWriter, self).__init__(logdir, *args, **kwargs)

    def flush(self):
        self.buffer.seek(0)
        try:
            self.s3.upload_fileobj(self.buffer, self.bucket, self.path)
        except botocore.exceptions.ClientError as e:
            print(f"S3 upload failed: {e}")
            # Optionally, log the error or take other action
        except Exception as e:
            print(f"Unexpected exception during S3 upload: {e}")
        finally:
            self.buffer.close()
            self.buffer = io.BytesIO()


def MyRecordWriter(logdir, filename_suffix=""):
    if logdir.startswith("s3://"):
        return MyS3RecordWriter(logdir)
    else:
        # Use the original RecordWriter for local directories
        return original_RecordWriter(logdir, filename_suffix)


# Monkey-patch the RecordWriter in tensorboardX
record_writer.RecordWriter = MyRecordWriter


class BackgammonPPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for Backgammon using BackgammonPolicyNetwork.
    """

    def __init__(
        self,
        input_size=198,
        hidden_size=HIDDEN_SIZE,
        action_size=10,
        entropy_coef_start=ENTROPY_COEF_START,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_anneal_episodes=ENTROPY_ANNEAL_EPISODES,
        log_dir=None,
        s3_bucket_name=None,
        s3_model_prefix="models/",
        s3_log_prefix="logs/",
        device=device,
    ):
        self.action_size = action_size
        self.device = device
        self.policy_network = BackgammonPolicyNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            action_size=action_size,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.scaler = GradScaler(device=self.device)

        # Entropy coefficient parameters
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_anneal_episodes = entropy_anneal_episodes
        self.entropy_coef = self.entropy_coef_start  # Initialize entropy coefficient

        # S3 parameters
        self.s3_bucket_name = s3_bucket_name
        self.s3_model_prefix = s3_model_prefix
        if s3_bucket_name:
            self.s3_client = boto3.client(
                "s3",
                config=Config(
                    retries={"max_attempts": 2, "mode": "standard"},
                    connect_timeout=5,
                    read_timeout=60,
                ),
            )
        else:
            self.s3_client = None

        # Define a logging interval
        self.LOG_INTERVAL = 1000

        self.total_steps = 0
        self.total_episodes = 0
        self.memory = []
        self.losses = []
        self.win_rates = []
        self.training = True  # Default to training mode

        # Initialize TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if s3_bucket_name:
            # If S3 bucket is provided, set log_dir to S3 path
            log_dir = f"s3://{s3_bucket_name}/{s3_log_prefix}{timestamp}/"
        elif log_dir is None:
            log_dir = os.path.join("runs", f"backgammon_ppo_{timestamp}")
        else:
            # Append timestamp to the provided log_dir to make it unique
            log_dir = os.path.join(log_dir, f"{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to TensorBoard at {log_dir}")

    def select_action(self, observations, action_masks=None):
        """
        Selects actions based on the current policy network and observations.
        Handles both batched and single observations.
        """
        # Convert observations to tensors
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(self.device)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)  # Add batch dimension

        # Convert action_masks to tensors
        if action_masks is not None:
            if isinstance(action_masks, np.ndarray):
                action_masks = torch.from_numpy(action_masks).float()
            action_masks = action_masks.to(self.device)
            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)  # Add batch dimension
        else:
            # If action_masks is not provided, assume all actions are legal
            action_masks = torch.ones(
                observations.size(0), self.action_size, device=self.device
            )

        # Forward pass
        logits, state_values = self.policy_network(observations)
        masked_logits = logits + (action_masks + 1e-45).log()
        action_probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(action_probs)

        if self.training:
            # Training mode
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            # Store experiences in memory
            for i in range(observations.size(0)):
                self.memory.append(
                    {
                        "observation": observations[i].unsqueeze(0),
                        "action_mask": action_masks[i].unsqueeze(0),
                        "action": actions[i].unsqueeze(0),
                        "action_log_prob": action_log_probs[i].unsqueeze(0),
                        "state_value": state_values[i].unsqueeze(0),
                        "reward": None,  # Will be filled later
                        "done": None,  # Will be filled later
                    }
                )
            return actions.cpu().numpy()
        else:
            # Inference mode
            actions = torch.argmax(action_probs, dim=-1)
            return actions.cpu().numpy()

    def update_entropy_coef(self):
        progress = min(1.0, self.total_episodes / self.entropy_anneal_episodes)
        self.entropy_coef = self.entropy_coef_start - progress * (
            self.entropy_coef_start - self.entropy_coef_end
        )
        # Log the entropy coefficient
        if self.total_episodes % self.LOG_INTERVAL == 0:
            self.writer.add_scalar(
                "Hyperparameters/Entropy_Coefficient",
                self.entropy_coef,
                self.total_episodes,
            )

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(
            reversed(rewards.cpu().numpy()), reversed(dones.cpu().numpy())
        ):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        """
        Optimizes the policy network using the collected experiences in memory.
        """
        if not self.memory:
            print("No data to update.")
            return

        observations = torch.cat(
            [item["observation"] for item in self.memory], dim=0
        ).to(self.device)
        actions = torch.cat([item["action"] for item in self.memory], dim=0).to(
            self.device
        )
        action_log_probs = torch.cat(
            [item["action_log_prob"] for item in self.memory], dim=0
        ).to(self.device)
        state_values = torch.cat(
            [item["state_value"] for item in self.memory], dim=0
        ).to(self.device)
        rewards = torch.tensor(
            [item["reward"] for item in self.memory], device=self.device
        ).float()
        dones = torch.tensor(
            [item["done"] for item in self.memory], device=self.device
        ).float()
        action_masks = torch.cat(
            [item["action_mask"] for item in self.memory], dim=0
        ).to(self.device)

        returns = self.compute_returns(rewards, dones)
        returns = torch.tensor(returns, device=self.device).float()

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Compute advantages
        advantages = returns - state_values.detach()

        # Optimize policy and value network
        total_loss = 0
        for _ in range(NUM_EPOCHS):
            with autocast(device_type=self.device.type):
                logits, new_state_values = self.policy_network(observations)
                masked_logits = logits + (action_masks + 1e-45).log()
                action_probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(action_probs)
                new_action_log_probs = dist.log_prob(actions.squeeze(-1))

                ratios = torch.exp(
                    new_action_log_probs - action_log_probs.detach().squeeze(-1)
                )
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_state_values.squeeze(-1), returns)

                # Update the total loss to include entropy term
                loss = (
                    policy_loss
                    + VALUE_LOSS_COEF * value_loss
                    - self.entropy_coef * dist.entropy().mean()
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Logging Additional Metrics at Reduced Frequency
            if self.total_steps % self.LOG_INTERVAL == 0:
                # Policy Entropy
                policy_entropy = dist.entropy().mean().item()
                # Value Estimates Statistics
                value_estimates = new_state_values.detach()
                value_mean = value_estimates.mean().item()
                value_std = value_estimates.std().item()
                # Advantages Statistics
                adv_mean = advantages.mean().item()
                adv_std = advantages.std().item()
                # Learning Rate Tracking
                current_lr = self.optimizer.param_groups[0]["lr"]
                # Logging to TensorBoard
                self.writer.add_scalar(
                    "Loss/Policy Loss", policy_loss.item(), self.total_steps
                )
                self.writer.add_scalar(
                    "Loss/Value Loss", value_loss.item(), self.total_steps
                )
                self.writer.add_scalar("Loss/Total Loss", loss.item(), self.total_steps)
                self.writer.add_scalar(
                    "Policy/Entropy", policy_entropy, self.total_steps
                )
                self.writer.add_scalar(
                    "Value/Estimate Mean", value_mean, self.total_steps
                )
                self.writer.add_scalar(
                    "Value/Estimate Std", value_std, self.total_steps
                )
                self.writer.add_scalar("Advantage/Mean", adv_mean, self.total_steps)
                self.writer.add_scalar("Advantage/Std", adv_std, self.total_steps)
                self.writer.add_scalar("Learning Rate", current_lr, self.total_steps)

            # Increment total steps
            self.total_steps += 1

        avg_loss = total_loss / NUM_EPOCHS
        self.losses.append(avg_loss)
        self.memory = []

        # Update entropy coefficient
        self.update_entropy_coef()

    def set_training_mode(self, training=True):
        """
        Sets the agent to training or evaluation mode.
        """
        self.training = training
        self.policy_network.train(training)
        mode = "training" if training else "evaluation"
        print(f"Agent set to {mode} mode.")

    def save_model_local(self, filename=None):
        if not os.path.exists("models"):
            os.makedirs("models")
        if filename is None:
            filename = os.path.join("models", "ppo_backgammon.pth")
        else:
            filename = os.path.join("models", filename)
        torch.save(self.policy_network.state_dict(), filename)
        print(f"Model saved locally to {filename}")

    def load_model_local(self, filename="ppo_backgammon.pth", training=True):
        if os.path.isfile(filename):
            self.policy_network.load_state_dict(
                torch.load(filename, map_location=self.device)
            )
            self.policy_network.to(self.device)
            if training:
                self.policy_network.train()  # Set to training mode
                self.training = True  # Update the agent's training flag
                print(f"Model loaded locally from {filename} and set to training mode.")
            else:
                self.policy_network.eval()  # Set to evaluation mode
                self.training = False  # Update the agent's training flag
                print(
                    f"Model loaded locally from {filename} and set to evaluation mode."
                )
        else:
            print(f"No saved model found locally at {filename}")

    def save_model_s3(self, filename=None):
        if not self.s3_client:
            print("S3 client not initialized. Cannot save to S3.")
            return

        if filename is None:
            filename = "ppo_backgammon_s3.pth"
        print("Saving model state_dict to buffer...")
        buffer = io.BytesIO()
        torch.save(self.policy_network.state_dict(), buffer)
        buffer.seek(0)  # Reset buffer position to the beginning
        print("Model saved to buffer.")

        # Define S3 key
        s3_key = os.path.join(self.s3_model_prefix, filename)
        print(f"Uploading model to s3://{self.s3_bucket_name}/{s3_key}...")

        # Check AWS credentials
        try:
            credentials = self.s3_client._request_signer._credentials
            if not credentials or not credentials.access_key:
                print("No AWS credentials found.")
                return
            else:
                print("AWS credentials found.")
        except Exception as e:
            print(f"Error retrieving AWS credentials: {e}")
            return

        try:
            self.s3_client.upload_fileobj(buffer, self.s3_bucket_name, s3_key)
            print(f"Model saved to s3://{self.s3_bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Failed to save model to S3: {e}")

    def load_model_s3(self, filename="ppo_backgammon_s3.pth", training=True):
        if not self.s3_client:
            print("S3 client not initialized. Cannot load from S3.")
            return

        s3_key = os.path.join(self.s3_model_prefix, filename)
        buffer = io.BytesIO()
        try:
            self.s3_client.download_fileobj(self.s3_bucket_name, s3_key, buffer)
            buffer.seek(0)
            self.policy_network.load_state_dict(
                torch.load(buffer, map_location=self.device)
            )
            self.policy_network.to(self.device)
            if training:
                self.policy_network.train()  # Set to training mode
                self.training = True
                print(
                    f"Model loaded from s3://{self.s3_bucket_name}/{s3_key} and set to training mode."
                )
            else:
                self.policy_network.eval()  # Set to evaluation mode
                self.training = False
                print(
                    f"Model loaded from s3://{self.s3_bucket_name}/{s3_key} and set to evaluation mode."
                )
        except Exception as e:
            print(
                f"Failed to load model from s3://{self.s3_bucket_name}/{s3_key}. Error: {e}"
            )

    def save_model(self, filename=None, to_s3=False):
        """
        A generic save method that can save either locally or to S3 based on the 'to_s3' flag.
        """
        if to_s3:
            self.save_model_s3(filename)
        else:
            self.save_model_local(filename)

    def load_model(self, filename=None, from_s3=False, training=True):
        """
        A generic load method that can load either locally or from S3 based on the 'from_s3' flag.
        """
        if from_s3:
            self.load_model_s3(filename, training)
        else:
            self.load_model_local(filename, training)

    def log_metrics(self, episode, episode_reward, win):
        win_rate = sum(self.win_rates[-100:]) / min(len(self.win_rates), 100)
        avg_loss = sum(self.losses[-100:]) / min(len(self.losses), 100)
        print(
            f"*** Episode {episode} | Reward: {episode_reward} | Loss: {avg_loss:.4f} | Win Rate: {win_rate*100:.2f}% ***"
        )

        # Log win rate to TensorBoard at reduced frequency
        if self.total_episodes % self.LOG_INTERVAL == 0:
            self.writer.add_scalar("Win Rate", win_rate, self.total_episodes)
