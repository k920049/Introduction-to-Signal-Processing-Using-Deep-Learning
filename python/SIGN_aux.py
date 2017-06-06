import numpy as np


def sample_Z(m, n):
    return np.random.uniform(low=-1.0, high=1.0, size=[m, n])
