import itertools
import warnings

import numpy as np
import scipy.optimize
import scipy.linalg

from lib.linear_sem import intsqrt


def h(W):
    # tr exp(W ◦ W) − d
    d_nodes = W.shape[0]
    return np.trace(scipy.linalg.expm(np.multiply(W, W))) - d_nodes


def grad_h(W):
    # 2 W ◦ exp(W ◦ W).T
    return np.multiply(2 * W, scipy.linalg.expm(np.multiply(W, W)).transpose())


def W_from_theta(theta, L_parametrization):
    Wvec = L_parametrization @ theta
    d2 = L_parametrization.shape[0]
    d = intsqrt(d2)
    W = Wvec.reshape(d, d).T
    return W


def make_h_paramterized(M):
    def h_from_v(v):
        W = W_from_theta(v, M)
        return h(W)

    def grad_h_from_v(v):
        W = W_from_theta(v, M)
        grad_w = grad_h(W)  # gradient with respect to the matrix W
        grad_v = grad_w.T.flatten() @ M  # vectorize and apply chain rule
        return grad_v

    return h_from_v, grad_h_from_v


def make_notears_loss(sigma_hat: np.ndarray, L: np.ndarray):
    """Paramterized versions. From Remark 3.7 i mest.pdf note but with Wref=0

    a = L@v - vec(I)

    loss = 0.5 ( a.T @ (I kron SigmaHat) @ a)
    grad = L.T @ @ (I kron SigmaHat) @ a

    sigma_hat is the covariance of the data

    """
    id = np.eye(sigma_hat.shape[0])
    i = id.T.flatten()
    Q = scipy.linalg.kron(id, sigma_hat)

    def notears_loss(v: np.ndarray):
        # logging.debug(f"{L},{v},{vref},{i}")
        a = L @ v - i
        return a @ Q @ a / 2.0

    def notears_gradient(v: np.ndarray):
        a = L @ v - i
        return L.T @ Q @ a

    return notears_loss, notears_gradient


def relaxed_notears(
    data_cov: np.ndarray,
    L,
    W_initial,
    dag_tolerance: float,
    optim_opts=None,
    verbose=False,
) -> dict:
    """Get Notears solution with a guarantee of zero on diagonal, to accepted tolerance
    """
    opts = optim_opts.copy() if optim_opts else {}

    f_fun, f_grad = make_notears_loss(data_cov, L)
    h_fun, h_grad = make_h_paramterized(L)
    theta_initial = np.linalg.pinv(L) @ W_initial.T.flatten()

    alpha_initial = 0.0  # lagrangian multiplier
    nitermax = opts.pop("nitermax", 100)
    s_start = opts.pop(
        "slack_start", 10.0
    )  # slack in inequality constraint. gives wierd results if the start is feasible.
    log_every = opts.pop("log_every", 10)
    rho_start = opts.pop("penalty_start", 1.0)  # quadratic penalty multiplier
    rho_max = opts.pop("penalty_max", 1e20)
    mu = opts.pop("penalty_growth_rate", 2.0)  # the rho increase factor
    constraint_violation_decrease_factor = opts.pop(
        "minimum_progress_rate", 0.25
    )  # the accepted least decrease in infeasibility
    tolerated_constraint_violation = opts.pop("tolerated_constraint_violation", 1e-12)
    if len(opts.keys()) != 0:
        warnings.warn(f"Unknown options keys: {opts.keys()}")

    def lagrangian(theta_and_s, rho, alpha):
        theta = theta_and_s[:-1]
        s = theta_and_s[-1]
        h_fun_theta = h_fun(theta)
        c = h_fun_theta + s ** 2 - dag_tolerance
        t1 = (rho / 2) * c ** 2
        t2 = alpha * c + t1
        return f_fun(theta) + t2

    def lagrangian_grad(theta_and_s, rho, alpha):
        theta = theta_and_s[:-1]
        s = theta_and_s[-1]
        h_fun_theta = h_fun(theta)
        h_grad_theta = h_grad(theta)
        f_grad_t = f_grad(theta)
        c = h_fun_theta + s ** 2 - dag_tolerance
        t1 = alpha + rho * c
        t2 = h_grad_theta * t1
        grad_wrt_theta = f_grad_t + t2
        grad_wrt_s = 2.0 * s * t1
        return np.hstack([grad_wrt_theta, grad_wrt_s])

    def solve_inner(rho, theta_and_s, alpha):
        res = scipy.optimize.minimize(
            fun=lagrangian,
            jac=lagrangian_grad,
            x0=theta_and_s,
            args=(rho, alpha),
            method="L-BFGS-B",
            options={
                "disp": None,  # None means that the iprint argument is used
                # 'ftol':1e-16, # default is 2.220446049250313e-09
                # 'gtol':1e-6, # default is 1e-5
                # 'iprint':0 # 0 = one output, at last iteration
            },
        )
        return res["x"], res["message"]

    theta = theta_initial
    alpha = alpha_initial
    s = s_start
    rho = rho_start
    nit = 0
    solved_complete = False
    running = True
    theta_inner = None
    message = ""
    while running:

        theta_and_s_inner, _ = solve_inner(rho, np.hstack([theta, s]), alpha)
        theta_inner = theta_and_s_inner[:-1]
        s_inner = theta_and_s_inner[-1]
        h_fun_theta_inner = h_fun(theta_inner)
        h_fun_theta = h_fun(theta)
        current_inner_constraint_violation = (
            h_fun_theta_inner + s_inner ** 2 - dag_tolerance
        )
        current_constraint_violation = h_fun_theta + s ** 2 - dag_tolerance

        if current_constraint_violation != 0.0:
            while current_inner_constraint_violation >= max(
                constraint_violation_decrease_factor * current_constraint_violation,
                tolerated_constraint_violation,
            ):
                rho = mu * rho
                theta_and_s_inner, message_inner = solve_inner(
                    rho, np.hstack([theta, s]), alpha
                )
                theta_inner = theta_and_s_inner[:-1]
                s_inner = theta_and_s_inner[-1]
                h_fun_theta_inner = h_fun(theta_inner)
                current_inner_constraint_violation = (
                    h_fun_theta_inner + s_inner ** 2 - dag_tolerance
                )
                nit = nit + 1
                if rho > rho_max:
                    break
                elif nit == nitermax:
                    break
                if nit % log_every == 0 and verbose:
                    print(
                        f"nit={nit}\t|theta_and_s|={np.linalg.norm(theta_and_s_inner)}"
                        f"\trho={rho:.3g}\t|c|={current_constraint_violation}"
                        f"\tmessage_inner={message_inner}"
                    )

        if current_inner_constraint_violation < tolerated_constraint_violation:
            if verbose:
                message = "Found a feasible solution!"
                print(message)
            solved_complete = True
            running = False
        elif rho > rho_max:
            message = "Rho > rho_max. Stopping"
            warnings.warn(message)
            running = False
        elif nit >= nitermax:
            running = False
            message = "Maximum number of iterations reached. Stopping."
            warnings.warn(message)

        # Outer loop of augmented lagrangian
        alpha = alpha + rho * h_fun_theta_inner
        theta = theta_inner
        s = s_inner
        # log.info("Stepping outer")

    theta_final = theta_inner
    W_final = W_from_theta(theta_final, L)
    return dict(
        theta=theta_final,
        s=s,
        w=W_final,
        success=solved_complete,
        rho=rho,
        alpha=alpha,
        nit=nit,
        message=message,
    )


def mest_covarance(W_hat, data_covariance, L_parametrization_matrix, verbose=False):
    """Compute the Least-squares precision matrix assuming:
    - Equality-constrained by a epsilon-relaxed h-function
    - Data generator has normally distributed noise (needed to skip estimating
        higher moments that covariance)
    - the W matrix is parametrized by vec(W) = L*theta

    the covariance matrix returned is the one for theta, not for W
    """

    # aliases and simple variabels
    d = W_hat.shape[0]
    d2 = d ** 2
    id = np.eye(d)
    sigma = data_covariance
    W_I = W_hat - id
    L = L_parametrization_matrix

    #
    # Make the computations
    #

    # The permutation matrix that works like P@vecop(A) = vecop(A.T)
    P = np.zeros((d2, d2))
    for i, j in itertools.product(range(d), repeat=2):
        P[d * i + j, d * j + i] = 1

    K_expected_loss_hessian = L.T @ scipy.linalg.kron(id, sigma) @ L

    # simpler formula for J, valid when using Isserlis' theorem
    J_score_covariance = (
        L.T
        @ (
            scipy.linalg.kron(W_I.T @ sigma @ W_I, sigma)
            + scipy.linalg.kron(W_I.T @ sigma, sigma @ W_I) @ P
        )
        @ L
    )

    grad_h_theta = grad_h(W_hat).T.flatten() @ L  # assume vec(W) = L@theta
    plane_normal_vec = grad_h_theta / np.linalg.norm(grad_h_theta, ord=2)
    Pi_projector = np.eye(plane_normal_vec.size, plane_normal_vec.size) - np.einsum(
        "i,j->ij", plane_normal_vec, plane_normal_vec
    )

    Pi = Pi_projector
    K = K_expected_loss_hessian
    Kinv = np.linalg.inv(K)
    J = J_score_covariance
    estimator_covariance_mat = Kinv @ Pi @ J @ Pi @ Kinv.T

    # estimator_precision_mat = np.linalg.pinv(estimator_covariance_mat)
    if verbose:

        def log_matrix_stats(matrix_variable_name, matrix):
            print(f"{matrix_variable_name} : {matrix}")
            print(f"{matrix_variable_name} condition number: {np.linalg.cond(matrix)}")
            eigvals = [
                a for a in sorted(np.linalg.eigvals(matrix), key=lambda l: abs(l))
            ]
            print(f"{matrix_variable_name} eigenvalues: {eigvals}")
            print(f"{matrix_variable_name} rank: {np.linalg.matrix_rank(matrix)}")

        log_matrix_stats("Pi", Pi)
        log_matrix_stats("J", J)
        log_matrix_stats("K", K)
        log_matrix_stats("estimator_covariance_mat", estimator_covariance_mat)
        # log_matrix_stats('estimator_precision_mat', estimator_precision_mat)

    return estimator_covariance_mat
