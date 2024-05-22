import typing

import numpy as np
from safety_gymnasium.builder import Builder

if typing.TYPE_CHECKING:
    from learner import Learner


def evaluate(
    agent: "Learner",
    env: Builder,
    num_episodes: int,
    discount: float = 0.99,
) -> tuple[float, float, float, float]:
    # stats = {'return': [], 'length': []}
    total_cost_ = []
    total_reward_ = []
    discounted_total_cost_ = []
    discounted_total_reward_ = []
    for _ in range(num_episodes):
        observation, _ = env.reset()  # type: ignore
        terminal = False
        truncal = False

        total_reward = 0.0
        discounted_total_reward = 0.0
        total_cost = 0.0
        discounted_total_cost = 0.0
        cumulated_discount = 1
        while not (terminal or truncal):
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, cost, terminal, truncal, _ = env.step(action)
            total_reward += reward
            discounted_total_reward += cumulated_discount * reward
            total_cost += cost
            discounted_total_cost += cumulated_discount * cost
            cumulated_discount *= discount

        total_reward_.append(total_reward)
        discounted_total_reward_.append(discounted_total_reward)

        total_cost_.append(total_cost)
        discounted_total_cost_.append(discounted_total_cost)

    average_return = np.array(total_reward_).mean()
    average_discounted_return = np.array(discounted_total_reward_).mean()

    average_undiscounted_cost = np.array(total_cost_).mean()
    average_discounted_cost = np.array(discounted_total_cost_).mean()

    return (
        average_return,
        average_discounted_return,
        average_undiscounted_cost,
        average_discounted_cost,
    )
