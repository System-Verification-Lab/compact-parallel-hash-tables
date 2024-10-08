# Compact parallel hash tables for the GPU

This is a small header-only CUDA C++ library implementing compact GPU hash
tables. There is currently a bucketed cuckoo and a 2-level iceberg table. More
information can be found in ARTIFACT.md and the conference paper *Compact
Parallel Hash Tables on the GPU* to be presented at Euro-Par 2024.
Preprint: [arXiv:2406.09255](https://arxiv.org/pdf/2406.09255).

## Installation

As this is a header-only library, it suffices to copy the `include` directory.

## Usage

### Host-side API

The snippet below creates an iceberg table with 16 bit slots in both levels,
with 2^10 primary buckets of 32 slots, and 2^7 secondary buckets of 16 slots,
and find-or-puts some keys.

```cuda
#include "iceberg.cuh"

[...]

auto table = Iceberg<uint16_t, 32, uint16_t, 16>(20, 10, 7);
table.find_or_put(keys_start, keys_end, results);
```

A full example can be found in the `examples` directory.

The library contains default permutation functions. It is possible to pass a
custom permutation function as an extra template argument. See
`include/iceberg.cuh` and `include/quotient.cuh`.

### Device-side API

Some documentation (especially for device-side usage) is provided in the
comments in `include/cuckoo.cuh` and `include/iceberg.cuh`. The test cases
therein may be useful examples as well.

## Build instructions

This project was developed using GCC 10 and CUDA Toolkit 12, but should also
work on more recent versions. It uses Thrust (included with the toolkit) and
C++20 for convenience/readability. Both could be eliminated.

To compile the tests and benchmark suite, first make sure that the CUDA Toolkit
in installed and that the environment is set up [properly][cudaenv] so that, in
particular, the nvcc compiler is in the PATH.
Then [install Meson and Ninja](https://mesonbuild.com/Getting-meson.html) and
setup a build directory using
```
meson setup build -Dbuildtype=debugoptimized
```
The project can then be compiled with
```
meson compile -C build
```

For older versions of the CUDA Toolkit (12.0), the build fails because of
warnings regarding CUB.  The `werror` flag must then be disabled. This can be
done by passing `-Dwerror=true` to the setup command above, or afterwards using
```
meson configure build -Dwerror=false
```

## Tests

The tests can be run with
```
meson test -C build
```

The interface of the `tests` executable (built in the build directory) is
automatically generated by [doctest][] and has useful options. For example,
```
./build/tests -s
```
also reports passed tests (useful for debugging).

## Benchmarks

The `benchmarks` folder contains code for benchmark executables, as well as
data generators. See ARTIFACT.md for more details.

## Implementation notes

These are key-only hash tables. The code can be used as a basis for key-value
implementations.

As implemented, the number N of buckets in (each level of) the hash tables is
always a power of 2. This slightly eases the implementation, as the address of
a key is then the first log N bits of its permutation σ(k), and the remainder
the other bits. More granular variation of N can be obtained by letting the
address of k be σ(k) % N and the (unfortunately named) remainder N / σ(k).

The main algorithms are in `include/cuckoo.cuh` and `include/iceberg.cuh`.

## Acknowledgements

Non-compact parallel bucketed Cuckoo hashing on the GPU is due to [BGHT][].
Iceberg hashing is due to [IcebergHT][].

Parts of the implementation are inspired by [CompactCuckoo][] and [BGHT][].
In particular, the cooperative-group based approach from [BGHT][] is used,
and the default key permutation is (a one-round Feistel function) based on the
hash family in [BGHT][] for comparison purposes.

[cudaenv]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
[BGHT]: https://github.com/owensgroup/BGHT
[CompactCuckoo]: https://github.com/DaanWoltgens/CompactCuckoo
[doctest]: https://github.com/doctest/doctest
[IcebergHT]: https://arxiv.org/abs/2210.04068
