import numpy as np

SEED = 2024_01_02_17_25_20  # time of initial commit
rng = np.random.default_rng(SEED)
keys = rng.choice(2**39, 1208_000_000, replace=False).astype(np.uint64)
keys.tofile("/data/hegemans/1208m39.bin")
