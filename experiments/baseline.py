import argparse
import datetime
import pathlib
import os
import itertools
import pprint


import numpy as np
import pandas as pd
import lingam
import scipy.stats
import networkx as nx
from tqdm import trange
import matplotlib.pyplot as plt
import colorama
import seaborn as sns

from lib.daggnn_util import simulate_random_dag, simulate_sem
from lib.linear_algebra import make_L_no_diag, make_Z_clear_first
from lib.linear_sem import ace, ace_grad
from lib.misc import printt, cross_moment_4
from lib.plotters import draw_graph
from lib.relaxed_notears import relaxed_notears, mest_covarance, ace_circ

colorama.init()


raw_fname = "raw.csv"
summary_fname = "summary.txt"
config_fname = "config.txt"
graph_fname = "graph.png"
plot_fname = "summary.png"

variances = {
    "linear-gauss": 1,
    "linear-exp": 1,
    "linear-gumbel": np.pi ** 2 / 6,
}


def main():
    tstart = datetime.datetime.now()
    printt("Starting!")
    output_folder = pathlib.Path(
        "output", f"baselines_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)

    printt("Parsing options")
    opts = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath(config_fname), "w") as f:
        f.write(pp.pformat(vars(opts)) + "\n")
    printt("Config:\n" + pp.pformat(vars(opts)))

    printt("Running experiment")
    raw_path = output_folder.joinpath(raw_fname)
    df, W_true = run_experiment(opts)
    df.to_csv(raw_path)
    draw_graph(W_true, "Adjacency matrix for SEM", output_folder.joinpath(graph_fname))

    printt("Processing experiment output")
    post_process(output_folder)

    printt("Done!")
    tend = datetime.datetime.now()
    printt(f"Total runtime was {tend-tstart}")


def lingam_once(data):
    model = lingam.DirectLiNGAM()
    model.fit(data)
    W = model.adjacency_matrix_.T
    d = W.shape[0]
    # L = make_L_no_diag(d)
    Z = make_Z_clear_first(d)
    assert all(np.diag(W) == 0), "W matrix has nonzero diagonal - L not appropriate"
    # theta = np.linalg.pinv(L) @ W.T.flatten()
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    ace = M[1, 0]
    return W, ace


def lingam_stuff(data, opts):
    W, ace = lingam_once(data)
    aces = np.zeros(opts.n_bootstrap)
    for b in trange(opts.n_bootstrap, desc="LiNGAM Bootstrap", leave=False):
        bootstrap_samples = np.random.randint(opts.n_data, size=opts.n_data)
        _, ace = lingam_once(data[bootstrap_samples, :])
        aces[b] = ace
    lo, hi = np.percentile(
        aces, [100 * opts.confidence_level / 2, 100 - 100 * opts.confidence_level / 2]
    )
    return W, ace, lo, hi


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
    q = scipy.stats.norm.ppf(1 - opts.confidence_level / 2)
    lo, hi = ace_value + ace_standard_error * np.array([-q, q])
    return w, ace_value, ace_standard_error, lo, hi


def simulate_linear_sem(G, sem_type, n):
    """Simulates a linear SEM as in DAG-GNN-article, and centers the data.
    Forces the noise to have variance 1"""
    data = simulate_sem(
        G,
        n=n,
        x_dims=1,
        sem_type=sem_type,
        linear_type="linear",
        noise_scale=1 / np.sqrt(variances[sem_type]),
    ).squeeze()
    data = data - data.mean(axis=0)
    return data


def generate_random_graph(opts):
    for seed in itertools.count(0):
        np.random.seed(seed)
        G = simulate_random_dag(
            d=opts.d_nodes, degree=opts.node_multiplier * 2, graph_type="erdos-renyi",
        )
        W = nx.to_numpy_array(G)
        d = W.shape[0]
        Z = make_Z_clear_first(d)
        id = np.eye(d)
        M = np.linalg.pinv(id - Z @ W.T)
        ace = M[1, 0]
        if not np.isclose(ace, 0):
            printt(f"Seed {seed} gave a graph of causal effect {ace}")
            return G, W, ace


def run_experiment(opts):

    G, W_true, ace_true = generate_random_graph(opts)

    ress = []
    for sem_type in ["linear-gauss", "linear-exp", "linear-gumbel"]:
        W_our_lim, ace_our_lim = ace_circ(
            W_true, np.eye(opts.d_nodes), opts.dag_tolerance
        )
        printt("Computing LiNGAM large data limit")
        W_lingam_lim, ace_lingam_lim = lingam_once(
            simulate_linear_sem(G, sem_type, opts.n_data_lingam_lim)
        )
        printt(f"Starting data draws for {sem_type}")
        for data_draw in trange(opts.n_repetitions, desc=f"Running for {sem_type}"):
            data = simulate_linear_sem(G, sem_type, opts.n_data)
            (
                W_lingam,
                ace_lingam,
                ace_ci_lingam_low,
                ace_ci_lingam_high,
            ) = lingam_stuff(data, opts)
            ress.append(
                dict(
                    data_draw=data_draw,
                    sem_type=sem_type,
                    Method="LiNGAM",
                    ace_true=ace_true,
                    ace_lim=ace_lingam_lim,
                    ace=ace_lingam,
                    ace_ci_low=ace_ci_lingam_low,
                    ace_ci_high=ace_ci_lingam_high,
                    confidence_level=opts.confidence_level,
                )
            )
            (
                W_our,
                ace_our,
                ace_our_se,
                ace_ci_our_low,
                ace_ci_our_high,
            ) = notears_stuff(data, opts)
            ress.append(
                dict(
                    data_draw=data_draw,
                    sem_type=sem_type,
                    Method="our",
                    ace_true=ace_true,
                    ace_lim=ace_our_lim,
                    ace=ace_our,
                    ace_ci_low=ace_ci_our_low,
                    ace_ci_high=ace_ci_our_high,
                    confidence_level=opts.confidence_level,
                )
            )

    df = pd.DataFrame(ress)
    return df, W_true


def post_process(output_folder):
    df = pd.read_csv(output_folder.joinpath(raw_fname))
    df[f"ci_cover"] = (
        (df[f"ace_ci_low"] < df[f"ace_lim"]) & (df[f"ace_ci_high"] > df[f"ace_lim"])
    ).astype("float")
    df[f"ci_width"] = df[f"ace_ci_high"] - df[f"ace_ci_low"]
    summary = (
        df[["sem_type", "Method", "ci_cover", "ci_width"]]
        .groupby(["sem_type", "Method"])
        .mean()
    )
    summ_path = output_folder.joinpath(summary_fname)
    with open(summ_path, "w") as f:
        f.write("Latex output\n")
        s = summary.to_latex(float_format="%.2f")
        f.write(s)
        f.write("\n\n")
        f.write("Easy readable output\n")
        f.write(str(summary))
    print(summary)

    sns.displot(df, row="Method", col="sem_type", x="ace")
    plt.savefig(output_folder.joinpath(plot_fname))


def parse_args():

    p = argparse.ArgumentParser(
        description="Compare our proposed method with baselines - currenctly LiNGAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dag_tolerance", default=1e-7, help="The epsilon-value we aim for")
    p.add_argument("--n_data", default=100, type=int, help="The number of data points")
    p.add_argument(
        "--n_data_lingam_lim",
        default=100_000,
        type=int,
        help="Data points for the large-sample lingam computation",
    )
    p.add_argument(
        "--n_repetitions",
        default=100,
        type=int,
        help="How many data sets to draw from the graph per noise type?",
    )
    p.add_argument(
        "--d_nodes", default=4, type=int, help="How many nodes in the graph?"
    )
    p.add_argument(
        "--confidence_level",
        default=0.1,
        type=float,
        help="What confidence (90%% coverage ==> 10%%)",
    )
    p.add_argument(
        "--n_bootstrap",
        default=100,
        type=int,
        help="Number of bootstrap repetitions for LiNGAM",
    )
    p.add_argument(
        "--node_multiplier",
        default=1.5,
        type=float,
        help="How many expected edges per node?",
    )
    opts = p.parse_args()
    return opts


if __name__ == "__main__":
    main()
