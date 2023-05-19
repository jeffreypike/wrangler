import jax
import jax.numpy as jnp

def mls(y, yhat):
    return jnp.mean(jnp.square(y - yhat))

def mls_grad(beta, X, y):
    return X.T @ (X @ beta - y) / len(y)

def mls_hess(beta, X, y):
    return X.T @ X / len(y)

def bce(y, yhat):
    return (y * jnp.log(yhat)).sum() + ((1-y) * (1 - jnp.log(yhat)).sum( )) / -len(y)

def bce_grad(beta, X, y):
    return 1

def bce_hess(beta, X, y):
    return 1

def get_loss_fn(loss):
    if loss in ['linear', 'mls']:
        return mls
    if loss in ['logistic', 'bce']:
        return bce
    return loss

def get_loss_gradient(loss):
    if loss in ['linear', 'mls']:
        return mls_grad
    if loss in ['logistic', 'bce']:
        return bce_grad
    return jnp.grad(loss) # probably more efficient to return pointwise computation

def get_loss_hessian(loss):
    if loss in ['linear', 'mls']:
        return mls_hess
    if loss in ['logistic', 'bce']:
        return bce_hess
    return jnp.grad(jnp.grad(loss))