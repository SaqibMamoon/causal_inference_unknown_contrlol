"""
A simple toy to understand the difference in convergence rate when estimating higher
cross moments

This was not included in the article, but only for my own understanding.
"""
from argparse import ArgumentParser
from itertools import product
from collections import Counter
from math import prod

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from lib.linear_sem import selected_graphs
from lib.daggnn_util import simulate_sem
from lib.misc import CheckUniqueStore

noise_characteristics = {
    "gauss": [0, 1, 0, 0],
    "exp": [0, 1, 2, 6],
    "gumbel": [0, np.pi ** 2 / 6, 1.14, 12 / 5],
}

variances = {
    "linear-" + noise: stuff[1] for noise, stuff in noise_characteristics.items()
}

np.set_printoptions(precision=2)


def simulate_linear_sem(W, sem_type, n):
    """Simulates a linear SEM as in DAG-GNN-article, and centers the data.
    Forces the noise to have variance 1"""
    data = simulate_sem(
        nx.DiGraph(W),
        n=n,
        x_dims=1,
        sem_type=sem_type,
        linear_type="linear",
        noise_scale=1,
    ).squeeze()
    data = data - data.mean(axis=0)
    return data


def cm(d, o, moments):
    """Compute cross moments under independence assumption

    Takes a random vector v with independent components and computes
     E[v⊗v⊗v....⊗v]

    Args:
        d: the dimensionality of the random vector v
        o: the order of the cross moments, i.e. how many copies of v
        moments: the raw (non-central/non-standardized) moments each components of v.

    Returns:
        The tensor of all cross moments
    """
    assert o <= len(moments), "You need to supply higher moments!"
    std_normal_cm4 = np.zeros(o * (d,))
    for idxs in product(range(d), repeat=o):
        std_normal_cm4[idxs] = prod(moments[v] for c, v in Counter(idxs).items())
    return std_normal_cm4


def raw_moments(mean, var, skew, ex_kurtosis):
    mu_tilde = {}
    mu_tilde[1] = 0
    mu_tilde[2] = 1
    mu_tilde[3] = skew
    mu_tilde[4] = ex_kurtosis + 3

    std = np.sqrt(var)
    mu = {}
    mu[1] = mu_tilde[1] * std ** 1
    mu[2] = mu_tilde[2] * std ** 2
    mu[3] = mu_tilde[3] * std ** 3
    mu[4] = mu_tilde[4] * std ** 4

    mu_prime = {}
    mu_prime[1] = mean + mu[1]
    mu_prime[2] = mu[2] + mu_prime[1] ** 2
    mu_prime[3] = mu[3] + 3 * mu_prime[1] * mu_prime[2] - 2 * mu_prime[1] ** 3
    mu_prime[4] = (
        mu[4]
        + 4 * mu_prime[1] ** 1 * mu_prime[3]
        - 6 * mu_prime[1] ** 2 * mu_prime[2]
        + 3 * mu_prime[1] ** 3 * mu_prime[1]
    )
    return mu_prime


def parse_args():
    p = ArgumentParser()
    p.add_argument("--named_graph", default="2forward")
    p.add_argument(
        "--noises",
        default=["gauss", "exp", "gumbel"],
        type=str,
        choices=noise_characteristics.keys(),
        nargs="+",
        action=CheckUniqueStore,
    )
    opts = p.parse_args()
    return opts


def run_experiment(opts):
    W = selected_graphs[opts.named_graph]
    ress = []
    for noise_type in opts.noises:
        print("does work on " + noise_type)

        d = W.shape[0]
        moments = raw_moments(*(noise_characteristics[noise_type]))
        noise_cov = cm(d, 2, moments)
        noise_cm4 = cm(d, 4, moments)

        M = np.linalg.inv(np.eye(d) - W.T)
        data_cov = np.einsum("ij,ki,lj->kl", noise_cov, M, M)
        data_cm4 = np.einsum(
            "ijkl,mi,nj,ok,pl->mnop", noise_cm4, M, M, M, M
        )  # by direct computation

        for n in np.logspace(1, 5, 10, dtype=int):
            for _ in range(50):
                # Generate data
                sample = simulate_linear_sem(W, "linear-" + noise_type, n=n)

                # Estimate with and without isserlis
                est_var = np.cov(sample, rowvar=False)
                est_cm4 = (
                    np.einsum("ij,ik,il,im->jklm", sample, sample, sample, sample) / n
                )
                varvar = np.tensordot(est_var, est_var, 0)
                est_cm4_iss = varvar + varvar.swapaxes(0, 2) + varvar.swapaxes(0, 3)
                ress.append(
                    dict(
                        est_cm4=est_cm4,
                        method="Normal",
                        data_cm4=data_cm4,
                        data_cov=data_cov,
                        n=n,
                        noise_type=noise_type,
                    )
                )
                ress.append(
                    dict(
                        est_cm4=est_cm4_iss,
                        method="Isserlis",
                        data_cm4=data_cm4,
                        data_cov=data_cov,
                        n=n,
                        noise_type=noise_type,
                    )
                )
    df = pd.DataFrame(ress)
    return df


def main():
    opts = parse_args()
    df = run_experiment(opts)
    post_process(df)


def post_process(df):
    # Post process
    df["maxerr"] = df.apply(
        lambda r: np.max(np.abs(r["est_cm4"] - r["data_cm4"])), axis=1
    )
    df["maxerr_rel"] = df.apply(
        lambda r: r["maxerr"] / np.max(np.abs(r["data_cm4"])), axis=1
    )
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df, x="n", y="maxerr_rel", hue="method", style="noise_type", ax=ax
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
