import jax
import jax.numpy as jnp

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
    return jnp.where((grad - alpha * lam * w) < tolerance, 0, 1)


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

    for _ in range(max_iter):
        # apply local quadratic approximation
        g = get_loss_gradient(loss)(beta, X, y)
        H = get_loss_hessian(loss)(beta, X, y)
        w = get_penalty_weights(penalty, len(beta))

        Q = H/2 + (1 - alpha) * lam * jnp.diag(w) # for adding ridge regression
        l = g - H @ beta + alpha * w * get_penalty_grad(penalty)(beta, lam) - alpha * lam * w * jnp.sign(beta)

        # solve the local quadratic approximation
        beta_new = fit_quadratic_lasso(beta, Q, l, w, lam * alpha, max_iter_outer = max_iter_quadratic_outer, max_iter_inner = max_iter_quadratic_inner, tolerance_cd = tolerance_cd, tolerance_kkt = tolerance_kkt)

        def penalized_loss(yhat, y, w, lam, alpha):
            return loss_fn(y, yhat) + alpha * lam * w @ jnp.abs(beta) + (1 - alpha) * lam * w @ jnp.square(beta)

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
        grad = get_loss_gradient(loss)(beta, X, y) + alpha * w * get_penalty_grad(penalty)(beta, lam) + lam * alpha * w * jnp.sign(beta) + 2 * (1 - alpha) * lam * beta
        kkt0 = fails_kkt0(grad, alpha * lam * w * jnp.sign(beta), tolerance_kkt)
        kkt1 = fails_kkt1(grad, w, lam, alpha, tolerance_kkt)
        kkt = jnp.where(beta != 0, kkt0, kkt1)
        if kkt.sum() == 0:
            break
    
    return beta
