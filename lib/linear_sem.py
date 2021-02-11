import numpy as np
import scipy.linalg


def intsqrt(d2):
    """Takes the square root, and assert it is an int"""
    d = np.sqrt(d2)
    if d != int(d):
        raise ValueError("Supplied value is not a perfect square")
    return int(d)


def ace(theta, L):
    """Compute the Average Causal Effect between node 0 and node 1 in the SEM
    specified by the parameter vector theta and parametrization matrix L"""
    d2 = L.shape[0]
    d = intsqrt(d2)
    Z = np.eye(d)
    Z[0, 0] = 0
    vecW = L @ theta
    W = vecW.reshape(d, d).T
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    return M[1, 0]


def generate_data_from_dag(m_obs: int, W: np.ndarray, seed: int = None) -> np.ndarray:
    """Take n samples from a linear SEM parametrized by w

    if seed=None, then no seed is set when generating the data
    rows in output = observations
    """
    d = W.shape[0]
    assert d >= 2
    assert W.shape[1] == d
    sigma = np.eye(d)
    mu = np.zeros(d)
    mean_true = np.linalg.inv(np.eye(d) - W.transpose()) @ mu
    cov_true = (
        np.linalg.inv(np.eye(d) - W.transpose()) @ sigma @ np.linalg.inv(np.eye(d) - W)
    )  # the true variance of the data
    if seed:
        np.random.seed(seed)  # setting the seed means we get the same rows every time
    data = np.random.multivariate_normal(mean=mean_true, cov=cov_true, size=m_obs)
    return data


def ace_grad(theta, L):
    """Compute the gradient of the causal effect
    under linearity assumptions and vec(W)=L@theta"""
    d2 = L.shape[0]
    d = intsqrt(d2)
    Z = np.eye(d)
    Z[0, 0] = 0
    vecW = L @ theta
    W = vecW.reshape(d, d).T
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    MZ = M @ Z
    prod = scipy.linalg.kron(MZ, M.T)
    myMat = prod @ L
    return myMat[d, :]


selected_graphs = {
    "2forward": np.array([[0, 0.4], [0, 0]]),
    "2backwards": np.array([[0, 0], [0.4, 0]]),
    "3fork": np.array([[0, 0.4, 0], [0, 0, 0], [0.7, 0.2, 0]]),  # dense v model - fork!
    "3path": np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0.2, 0]]),  # dense v model - path!
    "3collider": np.array(
        [[0, 0, 0.7], [0, 0, 0.2], [0, 0, 0]]
    ),  # dense v model - path!
    "3path stronger": 10
    * np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0.2, 0]]),  # dense v model - path!
    "3path possible": np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0, 0]]),
    "3path possible backwards": np.array([[0, 0, 0.7], [0.4, 0, 0], [0, 0, 0]]),
    "4collider": np.array(
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    ),  # one fork, one collider
    "calibration": np.array(
        [
            [0.0, -1.0, 1.6, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),  # for the experiment on calibration
}
