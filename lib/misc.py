import datetime

import numpy as np


def printt(*args):
    print(datetime.datetime.now().isoformat(), *args)


def cross_moment_4(data):
    """Returns all cross 4th moments of a dataset
    The computation does NOT exploit the symmetries, so there is a lot
    of cleverness to be done here..."""
    n = data.shape[0]
    return np.einsum("ij,ik,il,im->jklm", data, data, data, data) / n
