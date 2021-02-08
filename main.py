import argparse
import shutil
import pathlib
import subprocess

ofolder = pathlib.Path("output")


def clean():
    for folder in ofolder.iterdir():
        shutil.rmtree(folder)
    print(f"Deleted all content in folder {ofolder.relative_to('.')}")


def all_nonlinear():
    """Build the command line arguments for the nonlinearity experiments and run them"""
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
        ".1",
        "--n_data",
        "100",
        "--d_nodes",
        "10",
        "--k_edge_multiplier",
        "1",
        "--n_data_lingam_lim",
        "1000000",
    ]
    subprocess.run(["exp-baseline"] + opts)


def main():
    cmds = {"clean": clean, "nonlinear": all_nonlinear, "baseline": baseline}

    p = argparse.ArgumentParser()
    p.add_argument("cmd", type=str, choices=cmds.keys())
    args = p.parse_args()

    cmds[args.cmd]()
