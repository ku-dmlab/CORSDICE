import csv
import json
import random
import string
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from common import Batch


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class SafetyGymDataset:
    def __init__(self, env: gym.Env):
        dataset = env._env.get_dataset()  # type: ignore

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            observation_gap = float(
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
            )

            if observation_gap > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        # Create timestep informations.
        t = 0
        timesteps = np.zeros_like(dataset["rewards"], dtype=np.int64)
        for i in range(len(dataset["observations"])):
            timesteps[i] = t

            if dones_float[i] == 1.0:
                t = 0
            else:
                t += 1

        (terminal_indexes,) = np.where(dones_float == 1.0)  # noqa: E712
        terminal_indexes = np.insert(terminal_indexes, 0, -1)[:-1]
        initial_observations = dataset["observations"][terminal_indexes + 1]  # type: ignore

        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)
        self.dones = dataset["terminals"].astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)
        self.timesteps = timesteps
        self.initial_observations = initial_observations
        self.costs = dataset["costs"]

        self.initial_size = len(initial_observations)
        self.size = len(dataset["observations"])

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size, dtype=np.int64)
        initial_indexes = np.random.randint(self.initial_size, size=batch_size, dtype=np.int64)

        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            costs=self.costs[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            timesteps=self.timesteps[indx],
            initial_observations=self.initial_observations[initial_indexes],
        )


def _gen_dir_name():
    now_str = datetime.now().strftime("%m-%d-%y_%H.%M.%S")
    rand_str = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{now_str}_{rand_str}"


class Log:
    def __init__(
        self,
        root_log_dir,
        cfg_dict,
        txt_filename="log.txt",
        csv_filename="progress.csv",
        cfg_filename="config.json",
        flush=True,
    ):
        self.dir = Path(root_log_dir) / _gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir / txt_filename, "w")
        self.csv_file = None
        (self.dir / cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end="\n"):
        now_str = datetime.now().strftime("%H:%M:%S")
        message = f"[{now_str}] " + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir / self.csv_filename, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
