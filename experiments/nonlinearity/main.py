"""Code to see the performance of the method under nonlinear data generating process
"""
import os
import itertools
from pathlib import Path
import warnings
import argparse
import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.stats
import datetime
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from lib.daggnn_util import simulate_random_dag, simulate_sem
from lib.linear_algebra import make_L_no_diag
from lib.relaxed_notears import relaxed_notears, mest_covarance
from lib.linear_sem import ace, ace_grad
from lib.ols import myOLS


def summarize_results(folder_path):
    df = pd.read_csv(folder_path.joinpath("result.csv"))
    with open(folder_path.joinpath("summary.txt"), "w") as f:
        f.write(
            str(
                df.groupby("linear_type")["is_covered"]
                .mean()
                .apply(lambda v: f"{v:.2%}")
            )
        )
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="z_score", hue="linear_type", ax=ax, kde=True)
    fig.savefig(folder_path.joinpath("scores.png"))


def run_experiment(general_options, notears_options):
    output_folder = general_options["output_folder"]
    resdicts = []
    kmax = general_options["repetitions"]
    for linear_type, _ in tqdm(
        itertools.product(["linear", "nonlinear_1", "nonlinear_2"], range(kmax)),
        total=3 * kmax,
    ):
        general_options["linear_type"] = linear_type
        dic = random_G_and_ace(general_options, notears_options)
        dic["linear_type"] = linear_type
        resdicts.append(dic)

    df = pd.DataFrame(resdicts)
    q = scipy.stats.norm.ppf(1 - np.array(general_options["confidence_level"]) / 2)
    df["q"] = q
    df["z_score"] = (df["ace_circ"] - df["ace_n"]) / df["ace_n_se"]
    df["is_covered"] = np.abs(df["z_score"]) < df["q"]

    csv_path = output_folder.joinpath("result.csv")
    df.to_csv(csv_path)

    summarize_results(output_folder)


def ace_mc_naive(G, sim_args, absprec=0.01):
    """Make a naive MC computation until the SE is less than absprec"""
    G_int = G.copy()
    G_int.remove_edges_from([e for e in G.edges if e[1] == 0])
    res = ace_mc = ace_mc_se = None
    for k in 10 ** np.arange(4, 10):
        kwds = {**sim_args, "n": k}
        intervention_data = simulate_sem(G_int, **kwds).squeeze()
        res = myOLS(intervention_data[:, [0]], intervention_data[:, 1])
        ace_mc = res["params"][0]
        ace_mc_se = res["HC0_se"][0]
        if ace_mc_se < absprec:
            break
    if ace_mc_se >= absprec:
        warnings.warn(f"MC computation lacks precision, HC0_se={res['HC0_se']}")
    return ace_mc, ace_mc_se


def ace_notears(G, sim_args, m_obs, dag_tolerance, notears_options):
    data = simulate_sem(G, **sim_args, n=m_obs).squeeze()
    d_nodes = G.number_of_nodes()
    L_no_diag = make_L_no_diag(d_nodes)
    w_initial = np.zeros((d_nodes, d_nodes))
    data_cov = np.cov(data, rowvar=False)
    result_circ = relaxed_notears(
        data_cov, L_no_diag, w_initial, dag_tolerance, notears_options,
    )
    theta_n, w_n, success = (
        result_circ["theta"],
        result_circ["w"],
        result_circ["success"],
    )
    assert success
    ace_n = (ace(theta_n, L_no_diag)).item()
    covariance_matrix = mest_covarance(w_n, data_cov, L_no_diag)
    gradient_of_ace_with_respect_to_theta = ace_grad(theta=theta_n, L=L_no_diag)
    ace_var = (
        gradient_of_ace_with_respect_to_theta
        @ covariance_matrix
        @ gradient_of_ace_with_respect_to_theta
    )
    ace_se = np.sqrt(ace_var / m_obs)
    return ace_n, ace_se


def random_G_and_ace(general_options, notears_options):
    """Generate a random graph and see if the CI covers the true value"""
    d_nodes = general_options["d_nodes"]  # number of vertices
    m = general_options["k_edge_multiplier"] * d_nodes  # expected no of edges.
    G = simulate_random_dag(d=d_nodes, degree=m * 2 / d_nodes, graph_type="erdos-renyi")
    sim_args = dict(
        x_dims=1, sem_type="linear-gauss", linear_type=general_options["linear_type"],
    )
    ace_mc, ace_mc_se = ace_mc_naive(G, sim_args)
    ace_circ, ace_circ_se = ace_notears(
        G,
        sim_args,
        general_options["n_obs_circ"],
        general_options["dag_tolerance_epsilon"],
        notears_options,
    )
    ace_n, ace_n_se = ace_notears(
        G,
        sim_args,
        general_options["n_obs"],
        general_options["dag_tolerance_epsilon"],
        notears_options,
    )

    return dict(
        ace_mc=ace_mc,
        ace_circ=ace_circ,
        ace_n=ace_n,
        ace_mc_se=ace_mc_se,
        ace_n_se=ace_n_se,
        ace_circ_se=ace_circ_se,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repetitions", default=10, type=int)
    p.add_argument("--d_nodes", default=5, type=int)
    p.add_argument("--n_obs", default=1_000, type=int)
    p.add_argument("--n_obs_circ", default=1_000_000)
    p.add_argument("--dag_tolerance_epsilon", default=1e-5)
    p.add_argument(
        "--k_edge_multiplier",
        default=1,
        type=int,
        help=(
            "The number in a ER1, ER2, ER4 graph. It is the number of expected edges "
            "divided by the number of nodes. It is equal to the expected out-degree or "
            "the expected in-degree in the graph."
        ),
    )
    p.add_argument(
        "--confidence_level",
        default=0.05,
        help="95percent confidence ==> alpha=confidence_level=.05",
    )
    general_options = vars(p.parse_args())

    output_folder = Path(
        "output", f"nonlinearity_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)
    general_options["output_folder"] = output_folder
    notears_options = {}
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath("config.txt"), "w") as f:
        f.write("general_options\n")
        f.write(pp.pformat(general_options) + "\n")
        f.write("notears_options\n")
        f.write(pp.pformat(notears_options) + "\n")
    run_experiment(general_options, notears_options)


if __name__ == "__main__":
    main()
