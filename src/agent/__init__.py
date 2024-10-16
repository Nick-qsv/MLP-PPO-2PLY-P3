from .policy_network import BackgammonPolicyNetwork
from .ppo_agent import BackgammonPPOAgent
from .config import *
from .train import train_agent

__all__ = [
    "BackgammonPolicyNetwork",
    "BackgammonPPOAgent",
    "HIDDEN_SIZE",
    "NUM_ENVS",
    "NUM_UPDATES",
    "T_HORIZON",
    "NUM_EPOCHS",
    "HIDDEN_SIZE",
    "LEARNING_RATE",
    "GAMMA",
    "EPS_CLIP",
    "VALUE_LOSS_COEF",
    "ENTROPY_COEF_START",
    "ENTROPY_COEF_END",
    "ENTROPY_ANNEAL_EPISODES",
    "MAX_TIMESTEPS",
    "S3_BUCKET_NAME",
    "S3_MODEL_PREFIX",
    "S3_LOG_PREFIX",
    "train_agent",
]
