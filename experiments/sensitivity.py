""" Analysis of how sensitive the problem is with respect to the DAG tolerance epsilon
"""
import datetime
import os
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.linalg

import lib.relaxed_notears
from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace, make_random_w

from lib.plotters import draw_graph, plot_contours_in_2d


def run_experiment(general_options, notears_options):
    #
    #
    # Initialize
    #
    #
    output_folder = (
        f"output/sensitivity_eps_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        + f" {general_options['run_name']}"
    )
    os.mkdir(output_folder)

    t_start = datetime.datetime.now()
    print(f"RUN START")

    print(f"Outputting to folder {output_folder}")
    print(f"General options {general_options}")
    print(f"Notears options {notears_options}")

    result_dfs = []
    for k, w_true in enumerate(general_options["w_trues"]):
        #
        #
        # Set up
        #
        #
        print("Entering Set Up")
        d_nodes = w_true.shape[0]
        L_no_diag = make_L_no_diag(d_nodes)
        a = -0.001
        w_initial = np.zeros((d_nodes, d_nodes))

        print(f"Handling data generator k={k}")
        print(f"w_true = {w_true}")

        # m_obs = general_options['m_obs']
        # data = lib.relaxed_notears.generate_data_from_dag(m_obs=m_obs, W=w_true,
        #                                          seed=general_options['data_seed'])
        id = np.eye(d_nodes)
        M = np.linalg.pinv(id - w_true.T)
        data_cov = M @ M.T
        noise_cov = np.eye(data_cov.shape[0])
        noise_prec = np.linalg.pinv(noise_cov)
        Q = scipy.linalg.kron(noise_prec, data_cov)
        sQrt = scipy.linalg.sqrtm(Q)
        theta_unconstrained = np.linalg.pinv(sQrt @ L_no_diag) @ sQrt @ id.T.flatten()
        w_unconstrained = (L_no_diag @ theta_unconstrained).reshape(d_nodes, d_nodes).T
        h_unconstrained = lib.relaxed_notears.h(w_unconstrained)
        print(f"theta_unconstrained={theta_unconstrained}")
        print(f"w_unconstrained={w_unconstrained}")
        print(f"h_unconstrained={h_unconstrained}")

        result_dicts = []
        for dag_tolerance in general_options["dag_tolerance_epsilons"]:
            print(
                f"Starting handling of {general_options['run_name']},"
                f" eps={dag_tolerance}."
            )
            print("Computing Notears solution")
            if dag_tolerance >= h_unconstrained:
                warnings.warn(
                    f"The current dag tolerance will allow optimum in the interior."
                    f" {dag_tolerance} >= {h_unconstrained}"
                )

            result = lib.relaxed_notears.relaxed_notears(
                data_cov, L_no_diag, w_initial, dag_tolerance, notears_options
            )
            theta_notears = result["theta"]
            w_notears = result["w"]
            theta_true = L_no_diag.T @ w_true.T.flatten()
            h, grad_h = lib.relaxed_notears.make_h_paramterized(
                L_no_diag, np.zeros(w_initial.shape)
            )
            h_notears = h(theta_notears)
            assert result["success"], "Solving failed!"
            assert (
                h_notears
                < dag_tolerance + notears_options["tolerated_constraint_violation"]
            ), (
                f"h_notears >= dag_tolerance, {h_notears} >= "
                f"{dag_tolerance + notears_options['tolerated_constraint_violation']}"
            )
            d = dict(
                theta_notears=theta_notears,
                w_notears=w_notears,
                h_notears=h_notears,
                dag_tolerance=dag_tolerance,
                k=k,
                w_true=w_true,
                theta_true=theta_true,
                h_unconstrained=h_unconstrained,
                w_unconstrained=w_unconstrained,
                theta_unconstrained=theta_unconstrained,
                d_nodes=d_nodes,
                data_cov=data_cov,
                ace_notears=ace(theta_notears, L_no_diag),
                ace_true=ace(theta_true, L_no_diag),
                ace_err=ace(theta_notears, L_no_diag) - ace(theta_true, L_no_diag),
                ace_abs_err=np.abs(
                    ace(theta_notears, L_no_diag) - ace(theta_true, L_no_diag)
                ),
                rho=result["rho"],
            )
            result_dicts.append(d)
            print(f"message: {result['message']}")
            print(f"w_notears = {w_notears}")
            print(f"h(w_notears) = {h_notears:.2e}")
            # print(f"grad_h(w_notears) = {grad_h(theta_notears)}")
            print(f"|grad_h(w_notears)| = {np.linalg.norm(grad_h(theta_notears)):.2e}")
            print(f"augmentation rho={result['rho']:.2e}")
            print(f"lagr mult={result['alpha']:.2e}")

        #   Post process after completion of all runs (produce plots and save)
        #
        #
        df_inner = pd.DataFrame(result_dicts)
        w_best = df_inner["w_notears"][
            df_inner["dag_tolerance"].idxmin()
        ]  # the one with smallest dag tolerance...
        df_inner["max-metric"] = df_inner["w_notears"].apply(
            lambda w: np.abs((w - w_true)).max()
        )
        df_inner["2-metric"] = df_inner["w_notears"].apply(
            lambda w: np.linalg.norm(w - w_true, ord="fro")
        )
        df_inner["1-metric"] = df_inner["w_notears"].apply(
            lambda w: np.abs((w - w_true)).sum()
        )
        df_inner["max-metric_unconstrained"] = np.abs((w_unconstrained - w_true)).max()
        df_inner["2-metric_unconstrained"] = np.linalg.norm(
            w_unconstrained - w_true, ord="fro"
        )
        df_inner["1-metric_unconstrained"] = np.abs((w_unconstrained - w_true)).sum()
        result_dfs.append(df_inner)

        draw_graph(
            w=w_true,
            title="$W_{true}$",
            out_path=os.path.join(output_folder, f"{k}_w_true.png"),
        )
        plt.close(plt.gcf())
        draw_graph(
            w=w_best,
            title=f"$\\hat W$ for $\\epsilon={df_inner['dag_tolerance'].min()}$",
            out_path=os.path.join(output_folder, f"{k}_w_best.png"),
        )
        plt.close(plt.gcf())
        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs.flatten()
        absmax = np.abs(w_best).max()
        opts = dict(
            cmap="PiYG",
            norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax),
        )
        ax1.matshow(w_best, **opts)
        ax1.set_title("w_notears")
        ax2.matshow(w_true, **opts)
        ax2.set_title("w_true")
        fig.savefig(os.path.join(output_folder, f"{k}_w_best_mat.png"))
        plt.close(fig)

        print(f"w_true_k: {w_true}")
        print(f"w_best_k: {w_best}")

    df: pd.DataFrame = pd.concat(result_dfs)
    df.to_pickle(path=os.path.join(output_folder, "results.csv"))
    df = pd.read_pickle(
        os.path.join(output_folder, "results.csv")
    )  # make sure that code below works with the file saved.
    df.pivot(index="dag_tolerance", columns="k", values="1-metric").to_csv(
        os.path.join(output_folder, "dagtol_pgfplots.csv")
    )

    fig = plt.figure("00")
    ax = fig.subplots()
    for k in df.k.unique():
        a = df[df.k == k]
        ax.loglog(
            a.dag_tolerance,
            a["1-metric"],
            linestyle="dotted",
            marker=".",
            color=f"C{k}",
            label=f"$w(\\varepsilon)$, k={k}",
        )
        ax.loglog(
            a.h_unconstrained.unique(),
            a["1-metric_unconstrained"].unique(),
            linestyle="none",
            label=f"$w(\\infty)$, k={k}",
            color=f"C{k}",
            marker="*",
        )
        ax.axvline(a.h_unconstrained.unique(), linestyle="dotted", color=f"C{k}")
    ax.legend()
    fig.show()
    fig.savefig(os.path.join(output_folder, f"1-metric_thresh.png"))
    plt.close()

    for m in ["max-metric", "2-metric", "1-metric"]:
        fig = plt.figure(num=m)
        ax = fig.subplots()
        sns.lineplot(data=df, x="dag_tolerance", y=m, hue="k", ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.show()
        ax.legend_.remove()
        fig.savefig(os.path.join(output_folder, f"{m}.png"))
        plt.close(fig)

    fig = plt.figure("ace")
    ax = fig.subplots()
    sns.lineplot(data=df, y="ace_err", x="dag_tolerance", hue="k", ax=ax)
    ax.set_xscale("log")
    fig.show()
    fig.savefig(os.path.join(output_folder, f"ace_by_eps.png"))
    plt.close()

    fig = plt.figure("actual_h")
    ax = fig.subplots()
    sns.lineplot(data=df, y="h_notears", x="dag_tolerance", hue="k", ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.show()
    fig.savefig(os.path.join(output_folder, f"actual_h.png"))
    plt.close()

    fig = plt.figure("ace")
    ax = fig.subplots()
    for k in df.k.unique():
        a = df[df.k == k]
        ax.plot(
            a.dag_tolerance,
            np.abs(a["ace_abs_err"]),
            linestyle="dotted",
            marker=".",
            color=f"C{k}",
            label=f"$|\\gamma(\\hat w(\\varepsilon))|$, k={k}",
        )
        ax.axvline(a.h_unconstrained.unique(), linestyle="dotted", color=f"C{k}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.show()
    fig.savefig(os.path.join(output_folder, f"ace_thresh.png"))
    plt.close()

    idxs = [8, 11]
    for k in df.k.unique():
        a = df[df.k == k]
        thetas = np.array(a.theta_notears.to_list())
        d_nodes = df.d_nodes.unique().item()
        theta_true = a.theta_true[0]
        data_cov = a.data_cov[0]
        maximum = max(thetas[:, idxs].max(), theta_true[idxs].max())
        minimum = min(thetas[:, idxs].min(), theta_true[idxs].min())
        corner0 = maximum * 1.2 + 0.1
        corner1 = minimum * 1.2 - 0.1
        bbox = (corner1, corner0, corner1, corner0)
        h, _ = lib.relaxed_notears.make_h_paramterized(
            make_L_no_diag(d_nodes), np.zeros((d_nodes, d_nodes))
        )
        loss, _ = lib.relaxed_notears.make_notears_loss(
            data_cov, make_L_no_diag(d_nodes), np.zeros((d_nodes, d_nodes))
        )

        name_snap_pairs = [
            (f"eps{p}percent", int((thetas.shape[0] - 1) * p / 100))
            for p in [0, 20, 40, 60, 80, 100]
        ]
        for name, snap_idx in name_snap_pairs:
            # n.b. 0 % means the first (smallest) value

            def h_(theta_sub):
                theta = thetas[snap_idx, :].copy()
                theta[idxs] = theta_sub
                return h(theta)

            def loss_(theta_sub):
                theta = thetas[snap_idx, :].copy()
                theta[idxs] = theta_sub
                return loss(theta)

            fig = plt.figure(f"00{k}")
            ax = fig.subplots()
            norm = matplotlib.colors.LogNorm(
                vmin=a.dag_tolerance.min(), vmax=a.dag_tolerance.max()
            )
            cmap = plt.get_cmap("viridis")
            resolution = 100
            plot_contours_in_2d(
                [h_],
                ax,
                bbox,
                resolution,
                contour_opts=dict(
                    norm=norm, levels=sorted(a.h_notears), alpha=0.7, cmap=cmap
                ),
            )
            plot_contours_in_2d(
                [loss_],
                ax,
                bbox,
                resolution,
                contour_opts=dict(levels=10, alpha=0.5, colors="black"),
            )
            ax.plot(
                thetas[:, idxs[0]],
                thetas[:, idxs[1]],
                linestyle="solid",
                color="black",
                linewidth=0.2,
            )
            ax.scatter(
                x=theta_true[idxs[0]],
                y=theta_true[idxs[1]],
                marker="*",
                c="black",
                s=300,
            )
            s = ax.scatter(
                x=thetas[:, idxs[0]],
                y=thetas[:, idxs[1]],
                c=a.h_notears,
                norm=norm,
                cmap=cmap,
                marker="x",
            )
            ax.axvline(0, linestyle="solid", color="black")
            ax.axhline(0, linestyle="solid", color="black")
            fig.colorbar(s, label=r"$\epsilon$")
            ax.set_aspect("equal")
            fig.savefig(os.path.join(output_folder, f"{k}_coords_{name}.png"))
            plt.close(fig)

    print(f"Finished outputting")

    #
    #
    #   Wrap up
    #
    #
    t_end = datetime.datetime.now()
    print(f"END RUN === Run time: {str(t_end - t_start).split('.')[0]}")


if __name__ == "__main__":
    print("Recording options")
    d = 4
    general_options = dict()
    general_options["dag_tolerance_epsilons"] = np.logspace(-11, -0, 20)
    general_options["w_trues"] = [
        make_random_w(d, density=0.5, min_val=0.1, max_val=2, seed=s)
        for s in range(0, 10)
    ]
    # general_options['w_trues'] = [make_random_w(d, density=.5, min_val=1.0,
    # max_val=10.0, seed=s) for s in range(0, 10)]
    general_options["run_name"] = f"random{d}"

    notears_options = dict()
    notears_options["nitermax"] = 1000
    notears_options["log_every"] = 10
    # notears_options['penalty_start'] = 100000
    # notears_options['slack_start'] = 1.0
    notears_options["tolerated_constraint_violation"] = 1e-12  # default is 1e-12
    run_experiment(general_options, notears_options)
