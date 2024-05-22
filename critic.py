from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

import divergence
from common import Batch, InfoDict, Model, Params, PRNGKey
from divergence import FDivergence


def update_v(
    batch: Batch,
    critic: Model,
    value: Model,
    f_divergence: FDivergence,
    alpha: float,
) -> Tuple[Model, InfoDict]:
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[Array, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        sp_term = (q - v) / alpha
        value_loss = (v + alpha * divergence.f_conjugate(sp_term, f_divergence)).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
            "q-v": (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_constrained_q(
    batch: Batch,
    critic: Model,
    value: Model,
    cost_lambda: Model,
    discount: float,
) -> Tuple[Model, InfoDict]:
    next_v = value(batch.next_observations)
    target_q = (
        batch.rewards - cost_lambda() * batch.costs + discount * batch.masks * next_v
    )

    def critic_loss_fn(critic_params: Params) -> Tuple[Array, InfoDict]:
        q1: Array
        q2: Array
        q1, q2 = critic.apply(
            {"params": critic_params}, batch.observations, batch.actions
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_nu(
    batch: Batch,
    advantage: Model,
    value: Model,
    nu_net: Model,
    critic: Model,
    f_divergence: FDivergence,
    alpha: float,
    discount: float,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
) -> Tuple[Model, InfoDict]:
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)

    policy_ratio = divergence.policy_ratio(q, v, alpha, f_divergence)
    adv = advantage(batch.observations)

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.grad, argnums=1)
    def grad_nu(value_params, obs):
        return nu_net.apply({"params": value_params}, obs)

    def value_loss_fn(value_params: Params) -> Tuple[Array, InfoDict]:
        nu = nu_net.apply({"params": value_params}, batch.observations)
        next_nu = nu_net.apply({"params": value_params}, batch.next_observations)
        initial_nu = nu_net.apply({"params": value_params}, batch.initial_observations)
        state_ratio = divergence.state_ratio(
            adv, policy_ratio, f_divergence, discount, nu, next_nu
        )

        initial_value_loss = (1 - discount) * initial_nu
        non_initial_value_loss = (
            state_ratio * adv
            - divergence.f(state_ratio, f_divergence)
            + jax.lax.stop_gradient(state_ratio)
            * policy_ratio
            * (discount * next_nu - nu)
        )

        value_loss = (non_initial_value_loss + initial_value_loss).mean()

        # Interpolate observations for gradient penalty.
        interpolation_epsilon = jax.random.uniform(rng)
        interpolated_observations = (
            batch.initial_observations * interpolation_epsilon
            + batch.next_observations * (1 - interpolation_epsilon)
        )

        value2_grad = grad_nu(value_params, interpolated_observations)
        value2_grad_norm = jnp.linalg.norm(value2_grad, axis=1)
        value2_grad_penalty = gradient_penalty_coeff * jnp.mean(
            jax.nn.relu(value2_grad_norm - 5) ** 2
        )

        return value_loss + value2_grad_penalty, {
            "loss/nu(s0)": initial_value_loss.mean(),
            "loss/nu(s)": non_initial_value_loss.mean(),
            "loss/nu(s)_grad_penalty": value2_grad_penalty,
            "nu(s0)/mean": initial_nu.mean(),
            "nu(s0)/max": initial_nu.max(),
            "nu(s0)/min": initial_nu.min(),
            "nu(s)/mean": nu.mean(),
            "nu(s)/max": nu.max(),
            "nu(s)/min": nu.min(),
            "nu(s')/mean": next_nu.mean(),
            "nu(s')/max": next_nu.max(),
            "nu(s')/min": next_nu.min(),
            "w*(s)/mean": state_ratio.mean(),
            "w*(s)/max": state_ratio.max(),
            "w*(s)/min": state_ratio.min(),
            "w(a|s)/mean": policy_ratio.mean(),
            "w(a|s)/max": policy_ratio.max(),
            "w(a|s)/min": policy_ratio.min(),
        }

    new_nu_net, info = nu_net.apply_gradient(value_loss_fn)
    return new_nu_net, info


def update_adv(
    batch: Batch,
    advantage: Model,
    value: Model,
    nu_network: Model,
    critic: Model,
    f_divergence: FDivergence,
    alpha: float,
    discount: float,
    gradient_penalty_coeff: float,
) -> Tuple[Model, InfoDict]:
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)

    policy_ratio = divergence.policy_ratio(q, v, alpha, f_divergence)

    nu = nu_network(batch.observations)
    next_nu = nu_network(batch.next_observations)

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.grad, argnums=1)
    def grad_advantage(advantage_params, obs):
        return advantage.apply({"params": advantage_params}, obs)

    def advantage_loss_func(advantage_params: Params) -> Tuple[Array, InfoDict]:
        adv = advantage.apply({"params": advantage_params}, batch.observations)

        dnu = discount * next_nu - nu
        wdnu = policy_ratio * dnu

        adv_loss = ((wdnu - adv) ** 2).mean()

        adv_grad = grad_advantage(advantage_params, batch.observations)
        adv_grad_norm = jnp.linalg.norm(adv_grad, axis=1)
        adv_grad_penalty = gradient_penalty_coeff * jnp.mean(
            jax.nn.relu(adv_grad_norm - 5) ** 2
        )

        return adv_loss + adv_grad_penalty, {
            "loss/adv": adv_loss,
            "loss/adv_grad_penalty": adv_grad_penalty,
            "adv(s)/mean": adv.mean(),
            "adv(s)/max": adv.max(),
            "adv(s)/min": adv.min(),
        }

    new_avantage, info = advantage.apply_gradient(advantage_loss_func)
    return new_avantage, info


def update_cost_lambda(
    batch: Batch,
    cost_lambda: Model,
    nu_state: Model,
    alpha: float,
    discount: float,
    cost_limit: float,
    f_divergence: FDivergence,
):
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    def lambda_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        cost_coeff = cost_lambda.apply({"params": params})

        e = batch.rewards - cost_coeff * batch.costs + discount * next_nu - nu
        f_temp = divergence.f_derivative_inverse(e / alpha, f_divergence)
        state_action_ratio = jax.nn.relu(f_temp)
        cost_estimate = (state_action_ratio * batch.costs).mean()

        loss = cost_coeff * (cost_limit - cost_estimate)
        return loss, {
            "loss/lambda": loss,
            "cost/lambda": cost_coeff,
            "cost/estimate": cost_estimate,
        }

    new_cost_lambda, info = cost_lambda.apply_gradient(lambda_loss_fn)
    return new_cost_lambda, info
