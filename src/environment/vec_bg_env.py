import torch
from src.environment.backgammon_env import BackgammonEnv


class VectorizedBackgammonEnv:
    def __init__(self, num_envs=50, match_length=15, max_legal_moves=500, device=None):
        self.num_envs = num_envs
        self.device = device if device is not None else torch.device("cpu")
        self.envs = [
            BackgammonEnv(match_length, max_legal_moves, device=self.device)
            for _ in range(num_envs)
        ]

        # Observation and action spaces (assumed to be the same across all envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        observations = []
        for env in self.envs:
            obs = env.reset()  # Should return a tensor on the correct device
            observations.append(obs)
        observations = torch.stack(observations).to(self.device)
        return observations

    def step(self, actions):
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        observations = torch.stack(observations).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return (
            observations,
            rewards,
            dones,
            infos,
        )

    def get_action_masks(self):
        action_masks = []
        for env in self.envs:
            action_masks.append(env.action_mask)
        action_masks = torch.stack(action_masks).to(self.device)
        return action_masks

    def get_legal_board_features(self):
        legal_board_features = []
        for env in self.envs:
            legal_board_features.append(env.legal_board_features)
        legal_board_features = torch.stack(legal_board_features).to(self.device)
        return legal_board_features

    def render(self):
        for env in self.envs:
            env.render()

    def close(self):
        for env in self.envs:
            env.close()
