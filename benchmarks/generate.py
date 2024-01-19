import sys
import numpy as np

SEED = 2024_01_02_17_25_20  # time of initial commit

rng = np.random.default_rng(SEED)
keys = rng.integers(0, 100, 20_000_000)
np.savetxt(sys.stdout, keys, fmt="%d")
