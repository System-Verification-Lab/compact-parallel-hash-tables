import argparse
import numpy as np
import sys

ap = argparse.ArgumentParser(
        description="""\
Generate unique random numbers

The generated numbers are written to the output file \
in an array of unsigned 64 bit integers.
""")

ap.add_argument("-n", "--number",
                type=int,
                required=False,
                help="number of numbers to generate (default: 1208_000_000)",
                default=1208_000_000)
ap.add_argument("-w", "--width",
                type=int,
                required=False,
                help="max binary number width (default: 39)",
                default=39)
ap.add_argument("-o", "--outfile",
                type=argparse.FileType("w"),
                required=False,
                help="file to write the numbers to (default: stdout)",
                default=sys.stdout)
args = ap.parse_args()

SEED = 2024_01_02_17_25_20  # time of initial commit
rng = np.random.default_rng(SEED)
keys = rng.choice(2**args.width, args.number, replace=False).astype(np.uint64)
keys.tofile(args.outfile)
