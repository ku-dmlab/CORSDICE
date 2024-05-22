"""Implementations of algorithms for continuous control."""

import os
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import Array

import divergence
import policy
import value_net
from actor import update_actor
from common import Batch, InfoDict, Model, Params, PRNGKey
from critic import (
    update_adv,
    update_constrained_q,
    update_nu,
    update_v,
)
from divergence import FDivergence


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=["divergence"])
def _update_constrained_agent(
    divergence: FDivergence,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    nu: Model,
    advantage: Model,
    cost_lambda: Model,
    batch: Batch,
    discount: float,
    tau: float,
    alpha: float,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
):
    new_value, value_info = update_v(batch, target_critic, value, divergence, alpha)
    new_rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        batch, actor, target_critic, new_value, alpha, actor_rng
    )
    new_critic, critic_info = update_constrained_q(
        batch, critic, new_value, cost_lambda, discount
    )
    new_target_critic = target_update(new_critic, target_critic, tau)

    new_advantage, advantage_info = update_adv(
        batch=batch,
        advantage=advantage,
        value=new_value,
        nu_network=nu,
        critic=new_critic,
        f_divergence=divergence,
        alpha=alpha,
        discount=discount,
        gradient_penalty_coeff=0.0,
    )

    new_rng, nu_rng = jax.random.split(new_rng)
    new_nu, nu_info = update_nu(
        batch=batch,
        advantage=new_advantage,
        value=new_value,
        nu_net=nu,
        critic=new_critic,
        alpha=alpha,
        discount=discount,
        f_divergence=divergence,
        gradient_penalty_coeff=gradient_penalty_coeff,
        rng=nu_rng,
    )

    return (
        new_rng,
        new_actor,
        new_critic,
        new_value,
        new_target_critic,
        new_advantage,
        new_nu,
        {
            **critic_info,
            **value_info,
            **actor_info,
            **advantage_info,
            **nu_info,
        },
    )


@partial(jax.jit, static_argnames=["f_divergence"])
def update_cost_lambda(
    f_divergence: FDivergence,
    batch: Batch,
    value: Model,
    critic: Model,
    advantage: Model,
    nu_network: Model,
    cost_lambda: Model,
    alpha: float,
    discount: float,
    cost_ub: float,
):
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)
    adv = advantage(batch.observations)
    nu = nu_network(batch.observations)
    next_nu = nu_network(batch.next_observations)

    policy_ratio = divergence.policy_ratio(q, v, alpha, f_divergence)
    state_ratio = divergence.state_ratio(
        adv, policy_ratio, f_divergence, discount, nu, next_nu
    )
    cost_estimate = (state_ratio * policy_ratio * batch.costs).mean()

    def cost_lambda_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        cost_lambda_value = cost_lambda.apply({"params": params})
        cost_lambda_loss = cost_lambda_value * (cost_ub - cost_estimate)
        return cost_lambda_loss, {
            "loss/cost_lambda": cost_lambda_loss,
            "cost/estimate": cost_estimate,
            "cost/lambda": cost_lambda_value,
            "cost/dc": cost_estimate - cost_ub,
        }

    new_cost_lambda, info = cost_lambda.apply_gradient(cost_lambda_loss_fn)
    info["cost/after_update"] = new_cost_lambda()

    return new_cost_lambda, info


@partial(jax.jit, static_argnames=["divergence"])
def _update_jit_evaluation(
    divergence: FDivergence,
    critic: Model,
    value: Model,
    value2: Model,
    advantage: Model,
    batch: Batch,
    alpha: float,
    discount: float,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
) -> Tuple[Model, Model, PRNGKey, InfoDict]:
    new_advantage, advantage_info = update_adv(
        batch=batch,
        advantage=advantage,
        value=value,
        nu_network=value2,
        critic=critic,
        f_divergence=divergence,
        alpha=alpha,
        discount=discount,
        gradient_penalty_coeff=0.0,
    )

    new_rng, v2_rng = jax.random.split(rng)
    new_value2, value_info = update_nu(
        batch=batch,
        advantage=new_advantage,
        value=value,
        nu_net=value2,
        critic=critic,
        alpha=alpha,
        discount=discount,
        f_divergence=divergence,
        gradient_penalty_coeff=gradient_penalty_coeff,
        rng=v2_rng,
    )

    return (
        new_advantage,
        new_value2,
        new_rng,
        {**advantage_info, **value_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: Array,
        actions: Array,
        max_timesteps: int,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        lr_ratio: float = 1.0,
        cost_lr: float = 3e-4,
        cost_ub: float = 0.5,
        gradient_penalty_coeff: float = 1e-5,
        initial_lambda: float = 1.0,
        divergence: FDivergence = FDivergence.CHI,
        dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        layernorm: bool = False,
        max_steps: Optional[int] = None,
        max_clip: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
        ckpt_dir: Optional[Path] = None,
        ckpt_eval_dir: Optional[Path] = None,
    ):
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.max_clip = max_clip
        self.gradient_penalty_coeff = gradient_penalty_coeff
        self.divergence = divergence
        self.ckpt_dir = ckpt_dir
        self.ckpt_eval_dir = ckpt_eval_dir

        self.cost_limit = cost_ub
        self.cost_threshold = (
            cost_ub * (1 - discount**max_timesteps) / (1 - discount) / max_timesteps
        )

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )

        if opt_decay_schedule == "cosine":
            if max_steps is None:
                raise ValueError(f"{opt_decay_schedule} scheduler require max_steps.")

            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            optimiser = optax.adam(learning_rate=actor_lr)
        self.actor = Model.create(
            actor_def, inputs=[actor_key, observations], tx=optimiser
        )

        critic_def = value_net.DoubleCritic(hidden_dims)
        self.critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(
            hidden_dims, layer_norm=layernorm, dropout_rate=value_dropout_rate
        )
        self.value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        self.target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )

        # Define a model for offline evaluation.
        rng, value2_key, adv_key = jax.random.split(rng, 3)

        self.value2 = Model.create(
            value_def,
            inputs=[value2_key, observations],
            tx=optax.adam(learning_rate=value_lr * lr_ratio),
        )
        self.advantage = Model.create(
            value_def,
            inputs=[adv_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )
        rng, lambda_key = jax.random.split(rng, 2)

        lambda_def = value_net.CostLambda(initial_lambda)
        self.cost_lambda = Model.create(
            lambda_def,
            inputs=[lambda_key],
            tx=optax.adam(learning_rate=cost_lr),
        )

        self.rng = rng

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        (
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            self.advantage,
            self.value2,
            info,
        ) = _update_constrained_agent(
            divergence=self.divergence,
            actor=self.actor,
            critic=self.critic,
            value=self.value,
            target_critic=self.target_critic,
            nu=self.value2,
            advantage=self.advantage,
            cost_lambda=self.cost_lambda,
            batch=batch,
            discount=self.discount,
            tau=self.tau,
            alpha=self.alpha,
            gradient_penalty_coeff=self.gradient_penalty_coeff,
            rng=self.rng,
        )
        return info

    def update_constraint(self, batch: Batch) -> InfoDict:
        self.cost_lambda, info = update_cost_lambda(
            f_divergence=self.divergence,
            batch=batch,
            value=self.value,
            critic=self.critic,
            advantage=self.advantage,
            nu_network=self.value2,
            cost_lambda=self.cost_lambda,
            alpha=self.alpha,
            discount=self.discount,
            cost_ub=self.cost_threshold,
        )

        return info

    def update_evaluation(self, batch: Batch) -> InfoDict:
        (self.advantage, self.value2, self.rng, info) = _update_jit_evaluation(
            critic=self.critic,
            value=self.value,
            value2=self.value2,
            advantage=self.advantage,
            batch=batch,
            alpha=self.alpha,
            discount=self.discount,
            gradient_penalty_coeff=self.gradient_penalty_coeff,
            divergence=self.divergence,
            rng=self.rng,
        )

        return info

    def estimate_return(self, sample: Batch) -> tuple[float, float]:
        adv = self.advantage(sample.observations)
        q1, q2 = self.critic(sample.observations, sample.actions)
        q = jnp.minimum(q1, q2)
        v = self.value(sample.observations)

        policy_ratio = divergence.policy_ratio(q, v, self.alpha, self.divergence)
        state_ratio = divergence.state_ratio(
            adv,
            policy_ratio,
            self.divergence,
            self.discount,
            self.value2(sample.observations),
            self.value2(sample.next_observations),
        )

        estimated_return = (state_ratio * policy_ratio * sample.rewards).mean() / (
            1.0 - self.discount
        )
        estimated_cost = (state_ratio * policy_ratio * sample.costs).mean() / (
            1.0 - self.discount
        )

        return estimated_return, estimated_cost

    def save_ckpt(self, step: int):
        # Silently fail if save directory is not provided.
        if self.ckpt_dir is None:
            pass

        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.actor.train_state,
            step=step,
            prefix="actor_ckpt_",
        )
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.critic.train_state,
            step=step,
            prefix="critic_ckpt_",
        )
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.value.train_state,
            step=step,
            prefix="value_ckpt_",
        )

    def load_ckpt(self, ckpt_dir: Path, step: int):
        actor_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.actor.train_state,
            step=step,
            prefix="actor_ckpt_",
        )
        self.actor = self.actor.replace(params=actor_state.params)
        self.actor = self.actor.replace(tx=actor_state.tx)

        critic_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.critic.train_state,
            step=step,
            prefix="critic_ckpt_",
        )
        self.critic = self.critic.replace(params=critic_state.params)
        self.critic = self.critic.replace(tx=critic_state.tx)

        value_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.value.train_state,
            step=step,
            prefix="value_ckpt_",
        )
        self.value = self.value.replace(params=value_state.params)
        self.value = self.value.replace(tx=value_state.tx)

    def save_evaluation_ckpt(self, step: int):
        # Silently fail if save directory is not provided.
        if self.ckpt_dir is None:
            pass

        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(str(self.ckpt_eval_dir)),
            target=self.value2,
            step=step,
            prefix="value2_checkpoint_",
        )
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(str(self.ckpt_eval_dir)),
            target=self.advantage,
            step=step,
            prefix="advantage_checkpoint_",
        )

    def load_evaluation_ckpt(self, ckpt_dir: Path, step: int):
        value2_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.value2.train_state,
            step=step,
            prefix="value2_checkpoint_",
        )
        self.value2 = self.value2.replace(params=value2_state.params)
        self.value2 = self.value2.replace(tx=value2_state.tx)

        advantage_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.advantage.train_state,
            step=step,
            prefix="advantage_checkpoint_",
        )
        self.advantage = self.advantage.replace(params=advantage_state.params)
        self.advantage = self.advantage.replace(tx=advantage_state.tx)

    def save_constraint_ckpt(self, step: int):
        # Silently fail if save directory is not provided.
        if self.ckpt_dir is None:
            pass

        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(str(self.ckpt_eval_dir)),
            target=self.cost_lambda,
            step=step,
            prefix="cost_lambda_checkpoint_",
        )

    def load_constraint_ckpt(self, ckpt_dir: Path, step: int):
        lambda_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.cost_lambda.train_state,
            step=step,
            prefix="cost_lambda_checkpoint_",
        )
        self.cost_lambda = self.value2.replace(params=lambda_state.params)
        self.cost_lambda = self.value2.replace(tx=lambda_state.tx)
