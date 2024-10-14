import numpy as np  # pylint: disable=import-error
from src.environment.backgammon_env import BackgammonEnv


class VectorizedBackgammonEnv:
    def __init__(self, num_envs=50, match_length=15, max_legal_moves=500):
        self.num_envs = num_envs
        self.envs = [
            BackgammonEnv(match_length, max_legal_moves) for _ in range(num_envs)
        ]

        # Observation and action spaces (assumed to be the same across all envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        observations = []
        for env in self.envs:
            obs = env.reset()
            observations.append(obs)
        return np.stack(observations)

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
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            infos,
        )

    def get_action_masks(self):
        action_masks = []
        for env in self.envs:
            action_masks.append(env.action_mask)
        return np.stack(action_masks)

    def get_legal_board_features(self):
        legal_board_features = []
        for env in self.envs:
            legal_board_features.append(env.legal_board_features.numpy())
        return np.stack(legal_board_features)

    def render(self):
        for env in self.envs:
            env.render()

    def close(self):
        for env in self.envs:
            env.close()
