import argparse
import datetime
import pathlib
import os

import numpy as np
import pandas as pd
import lingam
import scipy.stats
import networkx as nx
from tqdm import trange

from lib.relaxed_notears import relaxed_notears, mest_covarance
from lib.linear_sem import ace, ace_grad
from lib.linear_algebra import make_L_no_diag
from lib.daggnn_util import simulate_random_dag, simulate_sem

raw_fname = "raw.csv"
summary_fname = "summary.tex"


def printt(*args):
    print(datetime.datetime.now().isoformat(), *args)


def main():
    pass


def lingam_stuff(data):
    model = lingam.DirectLiNGAM()
    model.fit(data)
    W = model.adjacency_matrix_.T
    d = W.shape[0]
    L = make_L_no_diag(d)
    Z = np.eye(d)
    Z[0, 0] = 0
    assert all(np.diag(W) == 0), "W matrix has nonzero diagonal - L not appropriate"
    theta = np.linalg.pinv(L) @ W.T.flatten()
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    ace = M[1, 0]
    return W, theta, ace


def notears_stuff(data: np.ndarray, opts: argparse.Namespace):
    data_cov = np.cov(data, rowvar=False)
    d = data.shape[1]
    L = make_L_no_diag(d)
    res = relaxed_notears(
        data_cov, L=L, W_initial=np.zeros((d, d)), dag_tolerance=opts.dag_tolerance,
    )
    assert res["success"]
    w = res["w"]
    theta = res["theta"]

    covariance_matrix = mest_covarance(w, data_cov, L)
    gradient_of_ace_with_respect_to_theta = ace_grad(theta=theta, L=L)
    ace_variance = (
        gradient_of_ace_with_respect_to_theta
        @ covariance_matrix
        @ gradient_of_ace_with_respect_to_theta
    )
    ace_standard_error = np.sqrt(ace_variance / opts.n_data)
    ace_value = ace(theta, L=L)

    return w, theta, ace_value, ace_standard_error


def true_stuff(G):
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    Z = np.eye(d)
    Z[0, 0] = 0
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    ace = M[1, 0]
    return W, ace


def run_experiment(opts):

    ress = []
    for _ in trange(opts.repetitions):
        G = simulate_random_dag(
            d=opts.d_nodes, degree=opts.node_multiplier * 2, graph_type="erdos-renyi",
        )
        for sem_type in ["linear-gaus", "linear-exp", "linear-gumbel"]:
            W_true, ace_true = true_stuff(G)
            data = simulate_sem(
                G,
                n=opts.n_data,
                x_dims=1,
                sem_type="linear-gauss",
                linear_type="linear",
            ).squeeze()
            W_lingam, theta_lingam, ace_lingam = lingam_stuff(data)
            W_n, theta_n, ace_n, ace_n_se = notears_stuff(data, opts)
            resd = dict(
                sem_type=sem_type,
                # W_true=W_true,
                ace_true=ace_true,
                # W_lingam=W_lingam,
                # theta_lingam=theta_lingam,
                ace_lingam=ace_lingam,
                # W_n=W_n,
                # theta_n=theta_n,
                ace_n=ace_n,
                ace_n_se=ace_n_se,
            )
            ress.append(resd)

    df = pd.DataFrame(ress)
    return df


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--dag_tolerance", default=1e-7, help="The epsilon-value we aim for")
    p.add_argument("--n_data", default=1000, help="The epsilon-value we aim for")
    p.add_argument(
        "--repetitions", default=10, help="How many random dag comparisons do we do?"
    )
    p.add_argument("--d_nodes", default=10, help="How many nodes in the graph?")
    p.add_argument(
        "--confidence_level", default=0.05, help="What confidence (95% coverage ==> 5%"
    )
    p.add_argument(
        "--node_multiplier", default=1, help="How many expected edges per node?"
    )
    opts = p.parse_args()
    output_folder = pathlib.Path(
        "output", f"baselines_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)

    raw_path = output_folder.joinpath(raw_fname)
    df = run_experiment(opts)
    df.to_csv(raw_path)

    df = pd.read_csv(raw_path)
    df["q"] = scipy.stats.norm.ppf(1 - np.array(opts.confidence_level) / 2)
    df["err_n"] = df.ace_n - df.ace_true
    df["err_lingam"] = df.ace_lingam - df.ace_true
    df_errs = df[["sem_type", "err_lingam", "err_n", "ace_true"]]
    df_rmse = np.sqrt(
        df_errs.groupby("sem_type").mean() ** 2 + df_errs.groupby("sem_type").var()
    )
    df_mean_err = df_errs.groupby("sem_type").mean()
    summ_path = output_folder.joinpath(summary_fname)
    summary_df = pd.concat([df_rmse, df_mean_err], keys=["rms", "avg"], axis=1)
    summary_df.set_index(
        summary_df.index.map(
            {"linear-gaus": "Normal", "linear-exp": "Exp", "linear-gumbel": "Gumbel"}
        ),
        inplace=True,
    )
    summary_df.index.name = "Noise"
    with open(summ_path, "w") as f:
        s = summary_df.transpose().to_latex(float_format="%.2f")
        f.write(s)
