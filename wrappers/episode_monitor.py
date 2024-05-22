import time

import numpy as np
import gymnasium as gym
from safety_gymnasium.builder import Builder


class EpisodeMonitor(Builder):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: Builder):
        self._env = env

        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.cost_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        self._reset_stats()
        return self._env.reset(seed=seed, options=options)  # type: ignore

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        observation, reward, cost, terminal, truncal, info = self._env.step(action)

        self.reward_sum += reward
        self.cost_sum += cost
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminal:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["cost"] = self.cost_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

        return observation, reward, cost, terminal, truncal, info

    @property
    def action_space(self) -> gym.spaces.Box:
        """Helper to get action space."""
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.Box | gym.spaces.Dict:
        """Helper to get observation space."""
        return self._env.observation_space
