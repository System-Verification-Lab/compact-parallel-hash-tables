# Artifact for Euro-Par 2024

The file README.md contains information about the code as a compact GPU hash
table library. This file contains the necessary information to verify the
results in the manuscript.

## Hardware requirements

- A CUDA GPU with compute capability >= 7.5
  - We verified our results on an RTX 2080 Ti, an RTX 3090, an RTX 4090, and an
    NVIDIA L40s. The figures in the manuscript were generated on the RTX 3090.
  - Some results are only achieved at high load. For this reason, benchmarks
    can be completed faster on GPUs with smaller memory. (For verification, a
    high-end GPU such as the L40s may thus not be desirable.)
- Roughly as much RAM as the GPU has memory
- Roughly as much free storage space as the GPU has memory

## Software requirements

- A relatively up-to-date Linux distribution
  - Windows with WSL may work, but we have not tested this. The instructions are for Linux
- Version 12 of the CUDA Toolkit, preferably 12.4, and matching drivers
  - The toolkit can be obtained at https://developer.nvidia.com/cuda-toolkit
- Python 3 and the ability to create virtual environments (venv / pip)
  - On some systems, the latter may require an extra package (e.g. python3-venv on Ubuntu)
  - The minimal Python version we tested is 3.7.13

## Setting up

1. Open a shell at the root directory of the artifact
2. Create and activate Python virtual environment, and install the dependencies
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r python-requirements.txt
   ```
   (this installs specific versions of meson, ninja, numpy, pandas, tomli, and matplotlib)
3. Compile the project in release mode
   ```
   meson setup release --buildtype=release -Dwerror=false
   meson compile -C release
   ```
4. Optionally, the tests can be run with
   ```
   meson test -C release
   ```
   They should all pass.

## Running the benchmarks

The project contains various benchmark suites, with various configuration
options. The interested reader may inspect the --help output of
`release/rates`, `benchmarks/generate.py` and `benchmarks/figures.py`. For
proper results, parameters need to be chosen so that the benchmark table is
around a third of the GPU memory. This allows for a large number of keys to be
inserted during the benchmark, thus measuring performance under load.

The script `benchmarks/benchmarks.sh` can be used to perform the benchmarks and
generate the figures corresponding to the ones in the article manuscript.

Continuing from the instructions of the previous section, the process can be run with

```
./benchmarks/benchmarks.sh SIZE
```

where SIZE is one of: tiny, small, normal, large. The small benchmark is
suitable for GPUs with around 12GB of memory, the normal benchmark for those
with 24GB of memory, and the large one for those with 48GB of memory. After the
script is finished, the `out` directory contains pdf files of figures
corresponding to those in the manuscript.

The tiny benchmark, though not at all representative for a system under load,
should complete in less than 20 minutes and can be used to verify that
everything works. The small benchmark can, depending on the GPU, already give
results in line with those in the manuscript for the find and find-or-put
loads. To obtain a representative insertion benchmark, a benchmark size close
to the GPU memory should be performed. (In particular, the 64-bit tables
perform much better under lower insertion loads than under high load. This will
be touched upon in the camera-ready manuscript.)

### Real-world (havi) benchmark

TODO: describe the data and how to run it

## Troubleshooting

### My GPU has much less than 12GB (or much more than 48GB) of memory

The `benchmarks.sh` script can also be supplied with an integer, in which case
this will be taken as the logarithm of the number of (primary) entries in the
benchmark tables. The `normal` 24GB size corresponds to 29, and each step
roughly halves or doubles the memory used. For GPUs with 6GB of memory one may
thus use `./benchmarks/benchmarks.sh 27`.

### Increasing the precision

Apart from increasing the benchmark size, precision may also be improved by
taking more measurements. A measurement consists of many table benchmarks, each
consisting of multiple runs. Measurements correspond with taking new random
variables (for the permutations). We have not observed much variance between
measurements, but confidence increases with the number of measurements. The
number of runs per measurement corresponds with running the same setup multiple
times (of which the first is always discarded as warmup).

The figures in the manuscript were generated on an RTX 3090 with 10
measurements, each measurement consisting of 10 runs per benchmark. A full
benchmark would take around 24 hours.

For this artifact submission, the number of runs per benchmark is set at 3 to
reduce the runtime. The `N_STEPS` constant in `benchmarks/benchmarks.cu` can be
used to control this (and the project needs to be recompiled afterwards with
`meson compile -C release`). The exact same benchmark as run for the manuscript
can then be performed with

```
./benchmarks/benchmarks.sh manuscript
```

It must be noted however, that similar results will only be obtained on a GPU
with the same 24GB memory capacity as our RTX 3090.
