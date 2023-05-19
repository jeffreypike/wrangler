import jax
import jax.numpy as jnp

def lasso(beta, lam):
    return lam * jnp.abs(beta)

def lasso_grad(beta, lam):
    return lam * jnp.sign(beta)

def get_penalty_fn(penalty):
    if penalty == "lasso":
        return lasso
    return penalty

def get_penalty_grad(penalty):
    if penalty == "lasso":
        return lasso_grad
    return jnp.grad(penalty)

def get_penalty_weights(penalty, n):
    if penalty == "lasso":
        return jnp.ones(n)
    