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
        "10000",
        "--confidence_level",
        "0.05",
    ]
    subprocess.run(common + ["--d_nodes", "5", "--k_edge_multiplier", "1"])
    subprocess.run(common + ["--d_nodes", "5", "--k_edge_multiplier", "2"])
    subprocess.run(common + ["--d_nodes", "10", "--k_edge_multiplier", "1"])
    subprocess.run(common + ["--d_nodes", "10", "--k_edge_multiplier", "2"])


def main():
    cmds = {"clean": clean, "nonlinears": all_nonlinear}

    p = argparse.ArgumentParser()
    p.add_argument("cmd", type=str, choices=cmds.keys())
    args = p.parse_args()

    cmds[args.cmd]()
