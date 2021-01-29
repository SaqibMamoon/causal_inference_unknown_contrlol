"""
Make sure that the confidence interval is indeed calibrated, and that
the point estimate is approximately normal.
"""

import datetime
import os
import itertools

import numpy as np
import scipy.stats
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt

from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace, generate_data_from_dag, ace_grad
from lib.relaxed_notears import relaxed_notears, mest_covarance, h


def run_experiment(general_options, notears_options):
    # Initialize
    #
    seed = 123456
    output_folder = (
        f"output/calibration_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)
    t_start = datetime.datetime.now()
    d_nodes = general_options["w_true"].shape[0]
    L_parametrization = make_L_no_diag(d_nodes)
    w_initial = np.zeros((d_nodes, d_nodes))
    q = scipy.stats.norm.ppf(1 - np.array(general_options["confidence_level"]) / 2)

    # Compute unconstrained minimum
    w_true = general_options["w_true"]
    d_nodes = w_true.shape[0]
    id = np.eye(d_nodes)
    M = np.linalg.pinv(id - w_true.T)
    noise_cov = np.eye(d_nodes)
    data_cov = M @ noise_cov @ M.T
    noise_prec = np.linalg.pinv(noise_cov)
    Q = scipy.linalg.kron(noise_prec, data_cov)
    sQrt = scipy.linalg.sqrtm(Q)
    theta_unconstrained = (
        np.linalg.pinv(sQrt @ L_parametrization) @ sQrt @ id.T.flatten()
    )
    w_unconstrained = (
        (L_parametrization @ theta_unconstrained).reshape(d_nodes, d_nodes).T
    )
    h_unconstrained = h(w_unconstrained)
    print(f"The unconstrained solution has a dag-tolerance of {h_unconstrained}")
    if h_unconstrained < general_options["dag_tolerance_epsilon"]:
        raise ValueError(
            "The chosen tolerance is too rough for this graph "
            "- the confidence interval is invalid!"
        )

    print(
        "### START ###################################################################"
    )

    theta_true = L_parametrization.T @ general_options["w_true"].T.flatten()
    ace_true = ace(theta_true, L_parametrization)
    print(f"True ACE is {ace_true}")

    data = generate_data_from_dag(
        m_obs=general_options["m_obs_for_gamma0_computation"],
        W=general_options["w_true"],
        seed=seed,
    )
    data_cov = np.cov(data, rowvar=False)
    result = relaxed_notears(
        data_cov,
        L_parametrization,
        w_initial,
        general_options["dag_tolerance_epsilon"],
        notears_options,
    )
    theta_circ, w_circ, success = result["theta"], result["w"], result["success"]
    if not success:
        raise ValueError
    ace_circ = ace(theta_circ, L_parametrization)
    print(f"The approximation of 'true' ACE is ACE_circ, which is {ace_circ}")
    print(
        f"The L1 error between 'w_true' and 'w_circ' is {np.abs(w_circ-w_true).sum()}"
    )

    result_dicts = []
    for n_data, rep in itertools.product(
        general_options["m_obss"], range(general_options["no_of_repetitions"])
    ):
        print(f"Running repetition {n_data, rep}")

        data = generate_data_from_dag(
            m_obs=n_data, W=general_options["w_true"], seed=rep
        )
        data_cov = np.cov(data, rowvar=False)
        result = relaxed_notears(
            data_cov,
            L_parametrization,
            w_initial,
            general_options["dag_tolerance_epsilon"],
            notears_options,
        )
        theta_n, w_n, success = result["theta"], result["w"], result["success"]
        ace_n = ace(theta_n, L_parametrization)
        if not success:
            raise ValueError

        covariance_matrix = mest_covarance(w_n, data_cov, L_parametrization)
        gradient_of_ace_with_respect_to_theta = ace_grad(
            theta=theta_n, L=L_parametrization
        )
        ace_variance = (
            gradient_of_ace_with_respect_to_theta
            @ covariance_matrix
            @ gradient_of_ace_with_respect_to_theta
            / n_data
        )
        ace_standard_error = np.sqrt(ace_variance)

        conf_interval_low = ace_n - q * ace_standard_error
        conf_interval_high = ace_n + q * ace_standard_error
        covered_ace_circ = float(conf_interval_low <= ace_circ <= conf_interval_high)

        d = dict(
            ace_n=ace_n,
            n=n_data,
            conf_interval_low=conf_interval_low,
            conf_interval_high=conf_interval_high,
            covered_ace_circ=covered_ace_circ,
            ace_standard_error=ace_standard_error,
            ace_true=ace_true,
            ace_circ=ace_circ,
            l1_err_vs_true=np.abs(w_n - w_true).sum(),
            l1_err_vs_circ=np.abs(w_n - w_circ).sum(),
        )
        result_dicts.append(d)

    print(
        "### POST PROCESS #############################################################"
    )
    df = pd.DataFrame(result_dicts)

    emp_capture = df[["n", "covered_ace_circ"]].groupby("n").covered_ace_circ.mean()
    print(f"The empirical capture rate per amount of data was {emp_capture}")

    fig = plt.figure(num=99)
    ax = fig.subplots()
    df["ace_n_z"] = 0
    df["ace_n_theoretical_z"] = 0
    for n in df.n.unique():
        dfn = df[df.n == n]
        mu = dfn.ace_n.mean()
        sigma = dfn.ace_n.std()
        no_reps = dfn.shape[0]
        a = 0.5 if no_reps > 10 else 3 / 8
        i = dfn["ace_n"].argsort().argsort() + 1  # double argsort = rank
        q = (i - a) / (no_reps + 1 - 2 * a)
        df.loc[df.n == n, "ace_n_theoretical_z"] = scipy.stats.norm.ppf(
            q=q, loc=0, scale=1
        )
        df.loc[df.n == n, "ace_n_z"] = (dfn["ace_n"] - mu) / sigma
        ax.plot(
            df.loc[df.n == n, "ace_n_theoretical_z"],
            df.loc[df.n == n, "ace_n_z"],
            label=f"Actual (n={n}",
            marker="o",
            linestyle="none",
        )
    idx = df["ace_n_theoretical_z"].argsort()
    ax.plot(
        df["ace_n_theoretical_z"][idx],
        df["ace_n_theoretical_z"][idx],
        label="If Normal",
        linestyle="solid",
        marker="None",
    )
    ax.set_title("Normality Plot")
    ax.legend()
    ax.set_xlabel("Theoretical Quantile")
    ax.set_ylabel("Actual Quantile")
    fig.savefig(os.path.join(output_folder, "ace_normal_plot.png"))
    plt.show()

    for n in df.n.unique():
        shapiro_wilk_stat, shapiro_wilk_p = scipy.stats.shapiro(df["ace_n"][df.n == n])
        print(
            "The Shapiro-wilk statistic for normality test of the data of ace_n"
            f" and n={n} is {shapiro_wilk_stat}, corresponding to a p-value of"
            f" {shapiro_wilk_p}"
        )

    df["k"] = df.n.map({v: k for k, v in enumerate(df.n.unique())})
    df.pivot(index="ace_n_theoretical_z", columns="k", values="ace_n_z").to_csv(
        os.path.join(output_folder, "calibration_qq.csv")
    )
    df.to_csv(os.path.join(output_folder, "results.csv"))

    #   Wrap up
    #
    #
    t_end = datetime.datetime.now()
    print(f"### END RUN ### Run time: {str(t_end - t_start).split('.')[0]}")
    return df


if __name__ == "__main__":
    d = 4
    general_options = dict()
    general_options["confidence_level"] = 0.05  # alpha. 95% confidence ===> alpha=.05
    general_options["m_obss"] = [10 ** 2, 10 ** 4]
    general_options["m_obs_for_gamma0_computation"] = 10 ** 7
    general_options["dag_tolerance_epsilon"] = 1e-7
    # general_options['w_true'] = make_random_w(d, density=.5, min_val=.4,
    # max_val=2, seed=7)
    general_options["w_true"] = np.array(
        [
            [0.0, -1.0, 1.6, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    general_options["no_of_repetitions"] = 3

    notears_options = dict()
    notears_options["nitermax"] = 1000

    experiment_result = run_experiment(general_options, notears_options)
