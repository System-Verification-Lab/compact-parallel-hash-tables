% Compact Parallel Hash Tables on the GPU
  Artifact Overview
% Steef Hegeman; Daan Wöltgens; Anton Wijs; Alfons Laarman
% May 15, 2024

This artifact supports the article "Compact Parallel Hash Tables on the GPU",
to appear in Euro-Par 2024. It consists of a basic header-only CUDA C++ library
for compact cuckoo and compact iceberg hash tables, as well as benchmarks.

The file README.md contains information about the artifact as a compact GPU
hash table library. There is a minimal example in the `examples` folder for
host-side use, and the tests and comments included in the header files in
`include` serve as further documentation.[^tests] The library can be used from
the host and from the device side.

[^tests]: The `doctest` tests are ignored during normal compilation. They
    can also be stripped from the headers if desired.

The rest of this document details how to verify the results in the
aforementioned article, running the benchmarks and generating figures. It is
also available in plain-text markdown (`ARTIFACT.md`), which might be useful
for copying commands or referencing this document from a terminal environment.

## Getting Started

### Hardware requirements

- A CUDA GPU with compute capability >= 7.5
  - We verified our results on an RTX 2080 Ti, an RTX 3090, an RTX 4090, and an
    NVIDIA L40s. The figures in the manuscript were generated on the RTX 3090.
  - Some results are only achieved at high load. For this reason, benchmarks
    can be completed faster on GPUs with smaller memory. (For verification, a
    high-end GPU such as the L40s may thus not be desirable.)
- Roughly as much RAM as the GPU has memory
- Roughly as much free storage space as the GPU has memory

### Software requirements

- A relatively up-to-date Linux distribution
- Version 12 of the CUDA Toolkit, preferably 12.4, and matching drivers
  - The toolkit can be obtained at https://developer.nvidia.com/cuda-toolkit
- Python 3 and the ability to create virtual environments (venv / pip)
  - On some systems, the latter may require an extra package
    (`python3-venv` on Ubuntu)
  - The minimal Python version we tested is 3.7.13, but newer is recommended.
- For the real-world benchmark, `xz` is required to decompress the data.

The library itself depends only on the CUDA toolkit and can be used by simply
copying over the files in the `include` directory. The additional requirements
are for running the benchmarks and producing the figures.

### Setting up

1. Open a shell at the root directory of the artifact
2. Create and activate a Python virtual environment, and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r python-requirements.txt
   ```
   This installs specific versions of the Python dependencies for compilation
   (`meson`, `ninja`), and the benchmark data / figure generation (`numpy`,
   `pandas`, `tomli`, and `matplotlib`) in the local virtual environment. If
   the install command raises an error (which may happen for older Python
   versions like Python 3.7), a working environment can likely be created with
   ```
   pip install meson ninja numpy pandas tomli matplotlib
   ```

   When launching a new shell, the virtual environment should again be activated.

3. Compile the project in release mode
   ```
   meson setup release --buildtype=release -Dwerror=false
   meson compile -C release
   ```
   meson will automatically download specific versions of the C++ dependencies:
   `doctest` for the tests, `argparse` and `nlohmann-json` for command line
   argument and JSON parsing in the benchmark runners. (Though we will not use
   JSON here.)
4. Run the tests with
   ```
   ./release/tests
   ```
   They should all pass.

We are now ready to run the benchmarks.

## Running the benchmarks

For convenience, a benchmark suite and figure generation can be performed with

```
./benchmarks/benchmarks.sh SIZE
```

where `SIZE` is one of: `tiny`, `small`, `normal`, `large`. The benchmark
results will only be representative if the system is put under load, so it is
important to choose the right benchmark size for the GPU.[^benchmarksize] The
`small` benchmark is suitable for GPUs with around 12GB of memory, the `normal`
benchmark for those with 24GB of memory, and the `large` one for those with
48GB of memory. The `tiny` benchmark can be used to quickly verify that the
benchmark process works.

After the script is finished, the `out-SIZE` directory contains pdf files of
figures corresponding to those in the manuscript. For comparison, the
manuscript figures can be found in the `reference` directory.

[^benchmarksize]: The 64-bit tables perform much better under lower loads than
    under high load in the insertion benchmark---though still significantly
    less than the compact tables except for under very small loads. This will
    be touched upon in the camera-ready paper.

### Runtime

The `tiny` benchmark, though not at all representative for a system under load,
should complete in less than an hour (less than 30 minutes on our RTX 4090).
The `normal` benchmark takes around 6 hours on an RTX 4090.

### Real-world (havi) benchmark

The real-world benchmark that appears in the article can be performed with
```
./benchmarks/benchmarks.sh havi
```
and should complete within 5 minutes. The output can be found in `out-havi`.

The input data is included with the artifact in the file `havi-log.txt.xz`. The
benchmark script will decompress this and convert it to a binary file
`havi.bin` before passing it to the `./release/havi` benchmark runner.

### Reference output

The `reference` directory contains the benchmark results used for the
manuscript and generated figures for comparison. These results were generated
on an RTX 3090 with 24GB memory with a benchmark setup comparable to the
`normal` parameters. (See below for a more accurate description.) The host
machine had an Intel Xeon Silver 4214R processor and 240GB of RAM.

## Notes and troubleshooting

### My GPU has much less than 12GB (or much more than 48GB) of memory

The `benchmarks.sh` script can also be supplied with an integer, in which case
this will be taken as the logarithm of the number of (primary) entries in the
benchmark tables. The `normal` size for GPUs with 24GB memory corresponds to
the integer 29, and each step roughly halves or doubles the memory used. For
GPUs with 6GB of memory one may thus use `./benchmarks/benchmarks.sh 27`.

### Fine-grained benchmarks

For further research, it may be useful to run more fine-grained benchmarks.
The interested reader may inspect the `--help` output of `release/rates`,
`benchmarks/generate.py` and `benchmarks/figures.py`.

### Increasing the precision

Apart from increasing the benchmark size, precision may also be improved by
taking more measurements. A measurement consists of many table benchmarks, each
consisting of multiple runs. Measurements correspond with taking new random
variables (for the permutations). We have not observed much variance between
measurements, but confidence increases with the number of measurements. The
number of runs per measurement corresponds with running the same setup multiple
times (of which the first is always discarded as warmup).

The figures in the manuscript were generated on an RTX 3090 with 5
measurements, each measurement consisting of 10 runs per benchmark. A full
benchmark would take around a day.

For this artifact submission, the number of measurements and runs is set to 3
to reduce the runtime. The number of measurements can be controlled via the
`N_MEASUREMENTS` variable in `benchmark/benchmarks.sh` (via a command-line
argument to the `rates` benchmark runner). The `N_STEPS` constant in
`benchmarks/benchmarks.cu` can be used to control the number of runs (and the
project needs to be recompiled afterwards with `meson compile -C release`). The
exact same benchmark as run for the manuscript can then be performed with

```
./benchmarks/benchmarks.sh manuscript
```

It must be noted however, that similar results will only be obtained on a GPU
with the same 24GB memory capacity as our RTX 3090.

### The final find-or-put figure looks off

In the final find-or-put figures, there are measurements for a
before-fill-factor of 0.75 and an after-fill-factor of 0.5. In reality, the
measurement corresponds with a before-fill-factor of 0.75 and an
after-fill-factor of 0.75 (hence these final figures start with a horizontal
line, until the measurements where the after-fill-factor is greater than the
before-fill-factor). This will be fixed for the camera-ready paper (likely by
removing these confusing measurements from the figure), but for now this
artifact faithfully reproduces the figures in the accepted manuscript.
