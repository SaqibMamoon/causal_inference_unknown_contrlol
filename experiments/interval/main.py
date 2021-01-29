"""
This experiment computes the confidence interval quite directly by
the method described in the article: point estimation plus the delta method.
"""
import datetime
import pickle
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from lib.relaxed_notears import relaxed_notears, mest_covarance, h
from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace, ace_grad, make_random_w, generate_data_from_dag
import lib.ols
from lib.plotters import draw_graph


def plot_trend(df, gamma_circ, fname=None):
    ace_color = "C0"
    ols_color = "C1"
    fig, ax = plt.subplots(num="trend")
    ax.axhline(
        gamma_circ, label=r"$\approx \gamma_\circ$", color="k", linestyle="dashed"
    )
    ax.errorbar(
        x=df["m_obs"],
        y=df["ace_value"],
        yerr=df["q_ace_standard_error"],
        label=r"$\Gamma_{\alpha,n}$",
        color=ace_color,
        linestyle="None",
        capsize=3,
    )
    ax.errorbar(
        x=df["m_obs"],
        y=df["ols_value"],
        yerr=df["q_ols_standard_error"],
        label=r"$B_{\alpha,n}$",
        color=ols_color,
        linestyle="None",
        capsize=3,
    )
    ax.legend()
    ax.set_xlabel(r"Number of observations $n$")
    ax.set_xscale("log")
    if fname:
        fig.savefig(fname)
        plt.close(fig)


def run_experiment(general_options, notears_options):
    #
    #
    # Initialize
    #
    #
    output_folder = (
        f"output/interval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        f" {general_options['batch_name']}"
    )
    os.mkdir(output_folder)

    t_start = datetime.datetime.now()
    print(f"RUN START")

    #
    #
    # Set up
    #
    #
    print("Entering Set Up")
    d_nodes = general_options["w_true"].shape[0]
    L_no_diag = make_L_no_diag(d_nodes)
    w_initial = np.zeros((d_nodes, d_nodes))

    print(f"Outputting to folder {output_folder}")
    print(f"General options {general_options}")
    print(f"Notears options {notears_options}")

    #
    #
    # Do the work
    #
    #
    m_obs = general_options["m_obss"][-1] * 10
    data = generate_data_from_dag(m_obs=m_obs, W=general_options["w_true"], seed=None)
    data_cov = np.cov(data, rowvar=False)
    result_circ = relaxed_notears(
        data_cov,
        L_no_diag,
        w_initial,
        general_options["dag_tolerance_epsilon"],
        notears_options,
    )
    theta_circ, w_circ, success = (
        result_circ["theta"],
        result_circ["w"],
        result_circ["success"],
    )
    assert success
    ace_circ = (ace(theta_circ, L_no_diag)).item()
    draw_graph(
        w=w_circ,
        title=f"$\\gamma_{{{m_obs}}}={ace_circ:.2f}$",
        out_path=os.path.join(output_folder, f"w_notears_{m_obs}.png"),
    )
    with open(os.path.join(output_folder, f"ace_circ.pkl"), mode="wb") as f:
        pickle.dump(file=f, obj=ace_circ)

    result_dicts = []
    for m_obs in general_options["m_obss"]:
        print(f"Starting handling of {general_options['batch_name']}, n={m_obs}.")
        data = generate_data_from_dag(
            m_obs=m_obs, W=general_options["w_true"], seed=None
        )
        print("Computing Notears solution")
        data_cov = np.cov(data, rowvar=False)
        result = relaxed_notears(
            data_cov,
            L_no_diag,
            w_initial,
            general_options["dag_tolerance_epsilon"],
            notears_options,
        )
        theta_notears, w_notears, success = (
            result["theta"],
            result["w"],
            result["success"],
        )
        draw_graph(
            w=w_notears,
            title=f"$\\gamma_{{{m_obs}}}={(ace(theta_notears, L_no_diag)).item():.2f}$",
            out_path=os.path.join(output_folder, f"w_notears_{m_obs}.png"),
        )
        print(f"w_notears: {w_notears}")
        print(
            f"h(w_notears): {h(w_notears)}, compare with"
            f" DAG tolerance {general_options['dag_tolerance_epsilon']}"
        )
        print(f"$\\gamma_{{{m_obs}}}$={(ace(theta_notears, L_no_diag))}")

        print("Computing precision")
        covariance_matrix = mest_covarance(w_notears, data_cov, L_no_diag)
        print("Computing ace")
        gradient_of_ace_with_respect_to_theta = ace_grad(
            theta=theta_notears, L=L_no_diag
        )
        ace_variance = (
            gradient_of_ace_with_respect_to_theta
            @ covariance_matrix
            @ gradient_of_ace_with_respect_to_theta
        )
        ace_standard_error = np.sqrt(ace_variance / m_obs)
        ace_value = ace(theta_notears, L=L_no_diag)

        print("Computing the OLS solution")
        regressors = np.delete(data, 1, axis=1)
        outcomes = data[:, 1]
        ols_result = lib.ols.myOLS(X=regressors, y=outcomes)
        ols_direct_causal_effect, ols_standard_error = (
            ols_result["params"][0],
            ols_result["HC0_se"][0],
        )

        q = scipy.stats.norm.ppf(1 - np.array(general_options["confidence_level"]) / 2)

        print("Saving results")
        d = dict(
            m_obs=m_obs,
            ace_value=ace_value,
            ace_standard_error=ace_standard_error,
            q_ace_standard_error=q * ace_standard_error,
            ols_value=ols_direct_causal_effect,
            ols_standard_error=ols_standard_error,
            q_ols_standard_error=q * ols_standard_error,
            q=q,
            ace_circ=ace_circ,
            confidence_level=general_options["confidence_level"],
        )
        result_dicts.append(d)

    print("Completed the optimization. On to plotting!")
    df = pd.DataFrame(result_dicts)
    df.to_csv(os.path.join(output_folder, "summary.csv"))

    print(f"start plotting trend graph")
    fname = os.path.join(output_folder, "asymptotics.png")
    plot_trend(df, ace_circ, fname)
    print(f"Finished plotting the trend graph")

    #
    #
    #   Post process after completion of all runs (produce plots and save)
    #
    #
    gamma_true = ace(
        general_options["w_true"].T.flatten(), L=np.eye(d_nodes ** 2)
    ).item()
    print(f"\\gamma={gamma_true}")
    draw_graph(
        w=general_options["w_true"],
        title=f"$\\gamma={gamma_true:.2f}",
        out_path=os.path.join(output_folder, f"w_true.png"),
    )

    #
    #
    #   Wrap up
    #
    #
    t_end = datetime.datetime.now()
    print(f"END RUN === Run time: {str(t_end - t_start).split('.')[0]}")


models = {
    "2nodes forward": np.array([[0, 0.4], [0, 0]]),
    "2nodes backwards": np.array([[0, 0], [0.4, 0]]),
    "3nodes fork": np.array(
        [[0, 0.4, 0], [0, 0, 0], [0.7, 0.2, 0]]
    ),  # dense v model - fork!
    "3nodes path": np.array(
        [[0, 0.4, 0.7], [0, 0, 0], [0, 0.2, 0]]
    ),  # dense v model - path!
    "3nodes collider": np.array(
        [[0, 0, 0.7], [0, 0, 0.2], [0, 0, 0]]
    ),  # dense v model - path!
    "3nodes path stronger": 10
    * np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0.2, 0]]),  # dense v model - path!
    "3nodes path possible": np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0, 0]]),
    "3nodes path possible backwards": np.array([[0, 0, 0.7], [0.4, 0, 0], [0, 0, 0]]),
    "4node collider": np.array(
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    ),  # one fork, one collider
    "4node random": make_random_w(4, density=0.5, min_val=0.4, max_val=1, seed=0),
    "5node random": make_random_w(5, density=0.5, min_val=0.2, max_val=2.0, seed=0),
}

if __name__ == "__main__":
    print("Recording options")
    general_options = dict()
    general_options["confidence_level"] = 0.05  # alpha. 95% confidence ===> alpha=.05
    general_options["m_obss"] = np.logspace(start=2, stop=5, num=10, dtype="int")
    general_options["dag_tolerance_epsilon"] = 1e-8

    notears_options = dict()
    notears_options["nitermax"] = 1000
    if True:
        for name, w_true in models.items():
            general_options["w_true"] = w_true
            general_options["batch_name"] = name
            run_experiment(general_options, notears_options)
