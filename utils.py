import jax
import jax.numpy as jnp

def has_converged(beta, beta_new, tolerance):
    return jnp.abs(beta_new - beta).sum() < tolerance