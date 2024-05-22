import typing
from typing import Tuple

import jax.numpy as jnp
from jax import Array

from common import Batch, InfoDict, Model, Params, PRNGKey

if typing.TYPE_CHECKING:
    from tensorflow_probability.substrates.jax.distributions import Distribution


def update_actor(
    batch: Batch,
    actor: Model,
    critic: Model,
    value: Model,
    alpha: float,
    rng: PRNGKey,
) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    weight = 1 + (q - v) / alpha
    weight = jnp.maximum(weight, 0)
    weight = jnp.clip(weight, 0, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[Array, InfoDict]:
        dist: Distribution = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": rng},
        )  # type: ignore
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
