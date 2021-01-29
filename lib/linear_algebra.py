import itertools
import numpy as np


def make_L_no_diag(d_nodes_in_graph):
    """
    The parametrization matrix L so that
        L@v = vecop(W)
    with W having zeros on the diagonal
    """
    d2 = d_nodes_in_graph ** 2
    n_vars = d_nodes_in_graph * (d_nodes_in_graph - 1)
    L = np.zeros(shape=(d2, n_vars))
    L_col = 0
    for L_row, (i, j) in enumerate(
        itertools.product(range(d_nodes_in_graph), repeat=2)
    ):
        if i == j:
            pass
        else:
            L[L_row, L_col] = 1
            L_col += 1
    return L
