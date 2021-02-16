import argparse
import shutil
import pathlib
import subprocess

ofolder = pathlib.Path("output")


def clean():
    for folder in ofolder.iterdir():
        shutil.rmtree(folder)
    print(f"Deleted all content in folder {ofolder.relative_to('.')}")


def nonlinear():
    common = [
        "exp-nonlinear",
        "--repetitions",
        "200",
        "--n_data",
        "1000",
        "--confidence_level",
        "0.05",
    ]
    subprocess.run(common + ["--d_nodes", "5", "--k_edge_multiplier", "1"])
    subprocess.run(common + ["--d_nodes", "5", "--k_edge_multiplier", "2"])
    subprocess.run(common + ["--d_nodes", "10", "--k_edge_multiplier", "1"])
    subprocess.run(common + ["--d_nodes", "10", "--k_edge_multiplier", "2"])


def baseline():
    opts = [
        "--dag_tolerance",
        "1e-7",
        "--confidence_level",
        ".05",
        "--n_data",
        "10_000",
        "--n_data_lingam_lim",
        "1000000",
        "--rand_graph",
        "10,1",
        "--n_repetitions",
        "100",
        "--n_bootstrap",
        "100",
    ]
    subprocess.run(["exp-baseline"] + opts)


def sensitivity():
    opts = [
        "--rand_graph",
        "4,1",
        "--eps_min",
        "1e-12",
        "--eps_max",
        "1e2",
        "--n_eps",
        "30",
        "--n_graphs",
        "10",
        "--ftol",
        "1e-11",
        "--gtol",
        "1e-7",
    ]
    subprocess.run(["exp-sensitivity"] + opts)


def calibration():
    subprocess.run(["exp-calibration"])


def main():
    cmds = {
        q.__name__: q for q in [clean, nonlinear, baseline, sensitivity, calibration]
    }

    p = argparse.ArgumentParser()
    p.add_argument("cmd", type=str, choices=cmds.keys())
    args = p.parse_args()

    cmds[args.cmd]()
