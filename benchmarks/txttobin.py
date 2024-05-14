#!/usr/bin/env python3

import argparse
import numpy as np
import sys

ap = argparse.ArgumentParser(
        description="""\
Parse txt of line-separated unsigned integers into binary file of uint64_t[]\
""")

ap.add_argument("infile",
                type=argparse.FileType("r"),
                help="txt file to read the numbers from (- for stdin)",
                default=sys.stdin)
ap.add_argument("outfile",
                type=argparse.FileType("w"),
                help="binary file to read the numbers to (- for stdout)",
                default=sys.stdout)
args = ap.parse_args()


# We're using fromstring here because fromfile does not play well with stdin
# It's less performant (whole input in memory) but does not matter for our purposes
np.fromstring(args.infile.read(), sep="\n", dtype=np.uint64).tofile(args.outfile)
