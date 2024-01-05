# Compact parallel hash tables for the GPU (WIP)

## Installation

As this is a header-only library, it suffices to copy the `include` directory.

## Build instructions

This project was developed using GCC 10 and CUDA Toolkit 12.
It uses Thrust and C++20 for convenience/readability. Both could be eliminated.

To compile the tests,
[install Meson and Ninja](https://mesonbuild.com/Getting-meson.html)
and setup a build directory using
```
meson setup build
```
The tests can then be run with
```
meson test -C build
```

__Warning:__ current versions of Meson do not properly trigger a rebuild for
nvcc executables if included headers change.
Run
```
meson compile -C build --clean
```
to trigger a full rebuild of the tests after modifying the headers.
This should not be necessary for Meson versions > 1.3.1.
