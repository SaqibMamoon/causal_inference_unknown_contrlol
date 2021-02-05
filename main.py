import argparse
import shutil
import pathlib

ofolder = pathlib.Path("output")


def clean():
    for folder in ofolder.iterdir():
        shutil.rmtree(folder)
    print(f"Deleted all content in folder {ofolder.relative_to('.')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", type=str, choices=["clean"])
    args = p.parse_args()

    if args.cmd == "clean":
        clean()
