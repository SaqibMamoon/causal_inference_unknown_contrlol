import datetime
import argparse

import numpy as np


def printt(*args):
    print(datetime.datetime.now().isoformat(), *args)


def cross_moment_4(data):
    """Returns all cross 4th moments of a dataset
    The computation does NOT exploit the symmetries, so there is a lot
    of cleverness to be done here..."""
    n = data.shape[0]
    return np.einsum("ij,ik,il,im->jklm", data, data, data, data) / n


class CheckUniqueStore(argparse.Action):
    """Checks that the list of arguments contains no duplicates, then stores"""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > len(set(values)):
            raise argparse.ArgumentError(
                self,
                "You cannot specify the same value multiple times. "
                + f"You provided {values}",
            )
        setattr(namespace, self.dest, values)


class RandGraphSpec:
    def __init__(self, raw_val):
        """Get the number of nodes and the edge multiplier from a string"""
        d, k = raw_val.split(",")
        self.d = int(d)
        self.k = float(k)

    def __repr__(self) -> str:
        return f"RandGraphSpec('{self.d},{self.k}')"

    def __str__(self) -> str:
        return f'"{self.d},{self.k}"'
