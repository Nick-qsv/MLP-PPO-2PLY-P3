import torch
from src.environment.backgammon_env import BackgammonEnv
import torch.multiprocessing as mp
from multiprocessing import Pipe


class VectorizedBackgammonEnv:
    def __init__(self, num_envs=1, match_length=15, max_legal_moves=500, device=None):
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


class ParallelBackgammonEnv:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.parent_conns, self.child_conns = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = []
        for idx in range(num_envs):
            p = mp.Process(target=self.worker, args=(self.child_conns[idx], device))
            p.daemon = True
            p.start()
            self.processes.append(p)
        for conn in self.child_conns:
            conn.close()

    @staticmethod
    def worker(conn, device):
        env = BackgammonEnv(match_length=15, max_legal_moves=500, device=device)
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                observation = env.reset()
                conn.send(observation)
            elif cmd == "step":
                action = data
                observation, reward, done, info = env.step(action)
                if done:
                    observation = env.reset()
                conn.send((observation, reward, done, info))
            elif cmd == "close":
                conn.close()
                break

    def reset(self):
        for conn in self.parent_conns:
            conn.send(("reset", None))
        observations = [conn.recv() for conn in self.parent_conns]
        return torch.stack(observations).to(self.device)

    def step(self, actions):
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self.parent_conns]
        observations, rewards, dones, infos = zip(*results)
        observations = torch.stack(observations).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return observations, rewards, dones, infos

    def get_action_masks(self):
        # Implement similar multiprocessing steps if needed
        action_masks = []
        for conn in self.parent_conns:
            conn.send(("get_action_masks", None))
        action_masks = [conn.recv() for conn in self.parent_conns]
        return torch.stack(action_masks).to(self.device)

    def close(self):
        for conn in self.parent_conns:
            conn.send(("close", None))
        for p in self.processes:
            p.join()
