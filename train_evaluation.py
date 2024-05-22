import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import dsrl  # noqa: F401
import gymnasium as gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm

import wandb
from environment import Environment, ENVIRONMENT_MAX_TIMESTEP
from common import Batch
from dataset_utils import Log, SafetyGymDataset, split_into_trajectories
from divergence import FDivergence
from evaluation import evaluate
from learner import Learner
from wrappers.episode_monitor import EpisodeMonitor
from safety_gymnasium.builder import Builder

FLAGS = flags.FLAGS
flags.DEFINE_string("proj_name", "debug", "Project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./results/", "Tensorboard logging dir.")
flags.DEFINE_enum("divergence", "KL", FDivergence, None)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_string("mix_dataset", "None", "mix the dataset")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("alpha", 1.0, "temperature")
flags.DEFINE_float("lr_ratio", 0.01, None)
flags.DEFINE_float("gradient_penalty_coeff", 1e-5, None)
flags.DEFINE_bool("train_cost", None, None)
flags.DEFINE_float("cost_ub", 0.01, None)
flags.DEFINE_float("initial_lambda", 1.0, None)
flags.DEFINE_string("ckpt_dir", None, None, required=False)
flags.DEFINE_string("eval_ckpt_dir", None, None, required=False)
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: Environment) -> Tuple[Builder, SafetyGymDataset]:
    env: Builder = gym.make(env_name).unwrapped  # type: ignore
    env = EpisodeMonitor(env)
    dataset = SafetyGymDataset(env)
    return env, dataset


def main(_):
    env_name = Environment(FLAGS.env_name)
    divergence = FDivergence(FLAGS.divergence)

    env, dataset = make_env_and_dataset(env_name)

    kwargs = dict(FLAGS.config)
    kwargs["alpha"] = FLAGS.alpha
    kwargs["lr_ratio"] = FLAGS.lr_ratio
    kwargs["divergence"] = divergence
    kwargs["initial_lambda"] = FLAGS.initial_lambda
    kwargs["cost_ub"] = FLAGS.cost_ub
    kwargs["gradient_penalty_coeff"] = FLAGS.gradient_penalty_coeff

    timestamp = datetime.fromtimestamp(time.time()).strftime("%m_%d_%H_%M_%S")
    ckpt_dir = Path(
        f"checkpoints/{env_name}_{divergence}_{FLAGS.alpha}_{FLAGS.seed}_{timestamp}"
    )
    ckpt_eval_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],  # type: ignore
        env.action_space.sample()[np.newaxis],  # type: ignore
        max_timesteps=ENVIRONMENT_MAX_TIMESTEP[env_name],
        max_steps=FLAGS.max_steps,
        ckpt_dir=ckpt_dir,
        ckpt_eval_dir=ckpt_eval_dir,
        **kwargs,
    )

    kwargs["env"] = env_name
    kwargs["seed"] = FLAGS.seed
    wandb.init(
        project=FLAGS.proj_name,
        name=env_name,
        config=kwargs,
    )

    log = Log(Path("benchmark") / env_name, kwargs)
    log(f"Log dir: {log.dir}")

    if FLAGS.ckpt_dir is None:
        i = 0
        for i in tqdm(
            range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
        ):
            batch = dataset.sample(FLAGS.batch_size)
            update_info = agent.update(batch)
            constraint_info = agent.update_constraint(batch)
            update_info |= constraint_info

            if i % FLAGS.log_interval == 0:
                wandb.log(update_info, i)

            if i % FLAGS.eval_interval == 0:
                (
                    normalized_return,
                    estimated_return,
                    undiscounted_cost,
                    average_discounted_cost,
                ) = evaluate(agent, env, FLAGS.eval_episodes)

                tqdm.write(
                    str(
                        {
                            "normalized_return": normalized_return,
                            "undiscounted_cost": undiscounted_cost,
                            "discounted_cost": average_discounted_cost,
                            "estimated_return": estimated_return,
                        }
                    )
                )
                wandb.log(
                    {
                        "normalized_return": normalized_return,
                        "undiscounted_cost": undiscounted_cost,
                        "discounted_cost": average_discounted_cost,
                        "estimated_return": estimated_return,
                    },
                    i,
                )

        agent.save_ckpt(i)
        agent.save_evaluation_ckpt(i)
        agent.save_constraint_ckpt(i)
    else:
        agent.load_ckpt(Path(FLAGS.ckpt_dir), FLAGS.max_steps)
        log(f"Loaded checkpoints from {FLAGS.ckpt_dir}")

    # Train additional models for offline evaluation.
    _, average_discounted_return, undiscounted_cost, average_discounted_cost = evaluate(
        agent, env, FLAGS.eval_episodes
    )

    # Evaluate the policy offline.
    num_batch = dataset.size // FLAGS.batch_size
    total_estimated_return = 0.0
    total_estimated_cost = 0.0

    for i in tqdm(range(num_batch)):
        index = np.arange(
            i * FLAGS.batch_size, (i + 1) * FLAGS.batch_size, dtype=np.int64
        )
        batch = Batch(
            observations=dataset.observations[index],
            actions=dataset.actions[index],
            rewards=dataset.rewards[index],
            masks=dataset.masks[index],
            next_observations=dataset.next_observations[index],
            initial_observations=None,  # type: ignore
            timesteps=dataset.timesteps[index],
            costs=dataset.costs[index],
        )

        batch_estimated_return, batch_estimated_cost = agent.estimate_return(batch)
        total_estimated_return += batch_estimated_return
        total_estimated_cost += batch_estimated_cost

        tqdm.write(
            f"{env_name}/Return: {total_estimated_return / (i+1)} / {average_discounted_return}\n"
            f"{env_name}/Cost: {total_estimated_cost / (i+1)} / {average_discounted_cost}"
        )

    print(
        f"{env_name}/Return: {total_estimated_return / num_batch} / {average_discounted_return}",
        f"{env_name}/Cost: {total_estimated_cost / num_batch} / {average_discounted_cost}",
        sep="\n",
    )
    wandb.log(
        {
            "estimated_return": total_estimated_return / num_batch,
            "discounted_return": average_discounted_return,
            "cost/estimate": total_estimated_cost / num_batch,
            "cost/ground_truth": average_discounted_cost,
        }
    )

    log.close()
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
