# src/__init__.py
from .agent.train import train_agent
from .testing import build, run_tests
from .constants import BAR_INDEX, BEAR_OFF_INDEX, NUMBER_OF_POINTS, PLAYER_TO_INDEX

__all__ = [
    "train_agent",
    "build",
    "run_tests",
    "BAR_INDEX",
    "BEAR_OFF_INDEX",
    "NUMBER_OF_POINTS",
    "PLAYER_TO_INDEX",
]
