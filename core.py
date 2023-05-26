import jax
import jax.numpy as jnp
from math import floor

from models import sigmoid, linear_regression, logistic_regression, get_model_fn
from losses import mls, mls_grad, mls_hess, bce, bce_grad, bce_hess, get_loss_fn, get_loss_gradient, get_loss_hessian
from penalties import lasso, lasso_grad, get_penalty_fn, get_penalty_grad, get_penalty_weights
from utils import has_converged

#######################################################
#                                                     #
#   Section 1: Solving the Quadratic Lasso Problem    #
#                                                     #
#######################################################

# S1.0 General Functions

def fails_quadratic_kkt0(beta, Q, l, w, lam, tolerance):
    return jnp.where((jnp.abs(2 * Q @ beta + l + lam * w * jnp.sign(beta))) < tolerance, 0, 1)

def fails_quadratic_kkt1(beta, Q, l, w, lam, tolerance):
    return jnp.where((jnp.abs(2 * Q @ beta + l) - lam * w) < tolerance, 0, 1)

# S1.1 Coordinate Descent

def soft_threshold(beta, gamma):
    return jnp.sign(beta) * (jnp.abs(beta) - gamma) * (jnp.abs(beta) > gamma)

def quadratic_minimizer_1d(j, beta, Q, l):
    return jnp.where(Q[j,j] != 0,
                     -(2 * (Q[:, j] @ beta - Q[j,j] * beta[j]) + l[j]) / (2 * Q[j,j]),
                     0)

def quadratic_threshold_1d(j, Q, w, lam):
    return jnp.where(Q[j,j] != 0,
                    (lam * w[j]) / (2 * Q[j,j]),
                    0)

@jax.jit
def step_quadratic_lasso_cd(beta, Q, l, w, lam, alternate_conditions = None):
    beta_new = beta.copy()
    cond = beta if alternate_conditions is None else alternate_conditions
    for j in range(len(beta_new)):
        update = jnp.where(cond[j] != 0,
                            jnp.where(w[j] != 0,
                                      soft_threshold(quadratic_minimizer_1d(j, beta, Q, l),
                                                     quadratic_threshold_1d(j, Q, w, lam)),
                                      quadratic_minimizer_1d(j, beta, Q, l)),
                            beta[j])
        beta_new = beta_new.at[j].set(update)
    return beta_new

# TODO: Currently oesn't work if there are any zeros on the diagonal
def fit_quadratic_lasso(beta_0, Q, l, w, lam, max_iter_outer, max_iter_inner, tolerance_cd, tolerance_kkt):
    beta = beta_0.copy()

    for _ in range(max_iter_outer):
        # run coordinate descent on the active set until convergence
        for _ in range(max_iter_inner):
            beta_prev = beta.copy()
            beta = step_quadratic_lasso_cd(beta_prev, Q, l, w, lam)
            if has_converged(beta, beta_prev, tolerance_cd):
                break
        
        # cycle through the (nontrivial) null set once to perturb the stationary solution
        cond = jnp.where((beta == 0) & (w != 0), 1, 0)
        beta = step_quadratic_lasso_cd(beta, Q, l, w, lam, cond)

        # now check KKT conditions
        kkt0 = fails_quadratic_kkt0(beta, Q, l, w, lam, tolerance_kkt)
        kkt1 = fails_quadratic_kkt1(beta, Q, l, w, lam, tolerance_kkt)
        kkt = jnp.where(beta != 0, kkt0, kkt1)
        if kkt.sum() == 0:
            break

    return beta

#######################################################
#                                                     #
#      Section 2: Pointwise Penalized Regression      #
#                                                     #
#######################################################

# S2.0 General Functions

def fails_kkt0(grad, penalty, tolerance):
    return jnp.where((grad + penalty) < tolerance, 0, 1)

def fails_kkt1(grad, w, lam, alpha, tolerance):
    return jnp.where((grad - (1 - alpha) * lam * w) < tolerance, 0, 1)


# S2.1 Local Quadractic Approximation

def fit_penalized(beta_0, 
                  X, 
                  y, 
                  loss, 
                  lam, 
                  alpha, 
                  max_iter,
                  max_iter_quadratic_outer, 
                  max_iter_quadratic_inner, 
                  tolerance_cd, 
                  tolerance_kkt, 
                  tolerance_loss,
                  penalty = 'lasso',
                  model = None):
    if not model:
        model = get_model_fn(loss)
    beta = beta_0.copy()
    loss_fn = get_loss_fn(loss)
    w = get_penalty_weights(penalty, len(beta))

    for _ in range(max_iter):
        # apply local quadratic approximation
        g = get_loss_gradient(loss)(beta, X, y)
        H = get_loss_hessian(loss)(beta, X, y)
        Q = H/2 + alpha * lam * jnp.diag(w) # for adding ridge regression
        l = g - H @ beta + (1 - alpha) * w * get_penalty_grad(penalty)(beta, lam) - (1 - alpha) * lam * w * jnp.sign(beta)

        # solve the local quadratic approximation
        beta_new = fit_quadratic_lasso(beta, Q, l, w, lam * (1 - alpha), max_iter_outer = max_iter_quadratic_outer, max_iter_inner = max_iter_quadratic_inner, tolerance_cd = tolerance_cd, tolerance_kkt = tolerance_kkt)

        def penalized_loss(yhat, y, w, lam, alpha):
            return loss_fn(y, yhat) + (1 - alpha) * lam * w @ jnp.abs(beta) + alpha * lam * w @ jnp.square(beta)

        # if the solution didn't lower the loss, find one that does with a golden-section search
        if penalized_loss(model(beta_new, X), y, w, lam, alpha) > penalized_loss(model(beta, X), y, w, lam, alpha) + tolerance_loss:
            golden_ratio = (jnp.sqrt(5) - 1) / 2
            left = -1
            right = 1
            for _ in range(max_iter):
                update_left = golden_ratio * left + (1 - golden_ratio) * right
                update_right = (1 - golden_ratio) * left + golden_ratio * right

                beta_left = update_left * beta_new + (1 - update_left) * beta
                beta_right = update_right * beta_new + (1 - update_right) * beta

                if penalized_loss(model(beta_left, X), y, w, lam, alpha) > penalized_loss(model(beta_right, X), y, w, lam, alpha):
                    left = update_left
                    beta_ls = beta_left
                else:
                    right = update_right
                    beta_ls = beta_right

                if has_converged(beta_ls, beta_new, tolerance_loss):
                    break

                beta_new = beta_ls

        beta = beta_new

        # now check KKT conditions
        grad = get_loss_gradient(loss)(beta, X, y) + (1 - alpha) * w * get_penalty_grad(penalty)(beta, lam) + lam * (1 - alpha) * w * jnp.sign(beta) + 2 * alpha * lam * beta
        kkt0 = fails_kkt0(grad, alpha * lam * w * jnp.sign(beta), tolerance_kkt)
        kkt1 = fails_kkt1(grad, w, lam, alpha, tolerance_kkt)
        kkt = jnp.where(beta != 0, kkt0, kkt1)
        if kkt.sum() == 0:
            break
    
    return beta

######################################################
#                                                    #
#      Section 3: Pathwise Penalized Regression      #
#                                                    #
######################################################

# S3.1 Input control

def get_lambda_path(X, y, loss, lambda_count, lambda_minmax_ratio, lambda_weights, alpha):
    grad = jnp.abs(get_loss_gradient(loss)(jnp.zeros(X.shape[1]), X, y))
    lambda_max = 1.1 * jnp.max(jnp.where(lambda_weights != 0, grad, 0) / jnp.where(lambda_weights != 0, lambda_weights, 1)) / alpha # why 1.1?
    lambda_min = lambda_minmax_ratio * lambda_max
    return jnp.exp(jnp.linspace(jnp.log(lambda_max), jnp.log(lambda_min), lambda_count))

def validate_inputs(X, 
                  y, 
                  loss,
                  penalty,
                  model,
                  lambdas,
                  lambda_count,
                  lambda_minmax_ratio,
                  lambda_weights,
                  alpha):

    supported_penalties = ['lasso', 'ridge']
    if penalty not in supported_penalties:
        raise ValueError(f"Only the following penalties are currently supported: {supported_penalties}")

    if model is None:
        model = get_model_fn(loss)

    if lambda_minmax_ratio is None:
        lambda_minmax_ratio = 0.01 if X.shape[0] < X.shape[1] else 0.001
    
    if lambda_count is None:
        lambda_count = 100
    lambda_count = floor(lambda_count)
    if lambda_count < 10 or lambda_count > 100:
        raise ValueError("Lambda count must be an integer between 10 and 100")

    if lambda_weights is None:
        lambda_weights = get_penalty_weights(penalty, X.shape[1])
    if len(lambda_weights) != X.shape[1]:
        raise ValueError(f"Number of penalty weights must match number of features.  Expected {X.shape[1]} weights but got {len(lambda_weights)}")
    
    if alpha is None:
        alpha = 1 if penalty == "ridge" else 0

    if lambdas is None:
        lambdas = get_lambda_path(X, y, loss, lambda_count, lambda_minmax_ratio, lambda_weights, alpha)

    return X, y, loss, penalty, model, lambdas, lambda_count, lambda_minmax_ratio, lambda_weights, alpha

# S3.2 Fitting the path
def fit_penalized_path(X, 
                  y, 
                  loss = "linear", 
                  penalty = 'lasso',
                  model = None,
                  lambdas = None,
                  lambda_count = None,
                  lambda_minmax_ratio = None,
                  lambda_weights = None,
                  alpha = None,
                  max_iter = 30,
                  max_iter_quadratic_outer = 10, 
                  max_iter_quadratic_inner = 100, 
                  tolerance_cd = 1e-7, 
                  tolerance_kkt = 1e-4, 
                  tolerance_loss = 1e-7):

    # validate inputs and set default values
    X, y, loss, penalty, model, lambdas, lambda_count, lambda_minmax_ratio, lambda_weights, alpha = validate_inputs(X, y, loss, penalty, model, lambdas, lambda_count, lambda_minmax_ratio, lambda_weights, alpha)
       
    output = {}

    # initialize beta as the "solution for large lambda" (all zeros!)
    beta = jnp.zeros(X.shape[1])
    w = get_penalty_weights(penalty, X.shape[1])

    for l in range(len(lambdas)):
        lam = lambdas[l]
        active_set = jnp.where(beta != 0, True, False)

        for j in range(max_iter):
            # first check KKT conditions
            grad = get_loss_gradient(loss)(beta, X, y) + (1 - alpha) * w * get_penalty_grad(penalty)(beta, lam) + lam * (1 - alpha) * w * jnp.sign(beta) + 2 * alpha * lam * beta
            kkt0 = fails_kkt0(grad, alpha * lam * w * jnp.sign(beta), tolerance_kkt)
            kkt1 = fails_kkt1(grad, w, lam, alpha, tolerance_kkt)
            kkt = jnp.where(active_set == True, kkt0, kkt1)
            if kkt.sum() == 0:
                break

            # if they are not satisfied, identify the worst offender
            grad_null = jnp.where(active_set == False, jnp.abs(grad), 0)
            jmax = jnp.argmax(grad_null)
            active_set = active_set.at[jmax].set(True)

            # now solve the penalized regression on the updated active set
            beta_reduced = beta[active_set]
            X_reduced = X[jnp.repeat(active_set.reshape(1, len(active_set)), repeats = X.shape[0], axis = 0)].reshape(-1, beta_reduced.shape[0])
            beta_new = fit_penalized(beta_reduced, X_reduced, y, loss, lam, alpha, max_iter, max_iter_quadratic_outer, max_iter_quadratic_inner, tolerance_cd, tolerance_kkt, tolerance_loss, penalty, model)
            print(beta_new)
            # update the active parameters
            beta = beta.at[active_set].set(beta_new)

        # record the result
        output[lam] = beta
    
    return output