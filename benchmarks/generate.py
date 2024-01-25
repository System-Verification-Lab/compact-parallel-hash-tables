import numpy as np

SEED = 2024_01_02_17_25_20  # time of initial commit

rng = np.random.default_rng(SEED)
keys = rng.integers(0, 2**45, 20_000_000, dtype=np.uint64)
keys.tofile("benchmarks/data/1.bin")

rng = np.random.default_rng(SEED)
keys = rng.integers(0, 2**13, 20_000_000, dtype=np.uint64)
keys.tofile("benchmarks/data/dups.bin")
