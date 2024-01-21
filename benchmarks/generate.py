import sys
import numpy as np

SEED = 2024_01_02_17_25_20  # time of initial commit
MIN = 0
MAX = 2**45-1

rng = np.random.default_rng(SEED)
keys = rng.integers(0, MAX + 1, 20_000_000)
np.savetxt(sys.stdout, keys, fmt="%d")
