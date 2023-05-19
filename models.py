import jax
import jax.numpy as jnp

# def linear_regression(params, X):
#     return X @ params['weights'] + params['bias']

def linear_regression(beta, X):
    return X @ beta

def sigmoid(X):
    return 1 / (1 + jnp.exp(-x))

def logistic_regression(params, X):
    return sigmoid(linear_regression(params, X))

def get_model_fn(loss):
    if loss in ['linear', 'mls']:
        return linear_regression
    if loss in ['logistic', 'bce']:
        return logistic_regression
    raise ValueError('If the loss function is not linear or logistic, you must specify the model!')