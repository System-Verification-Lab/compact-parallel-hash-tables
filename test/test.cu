// Tests are written with the implementation (in the "include" directory).
// This file instructs doctest to gather them and generate an entrypoint.

#define DOCTEST_CONFIG_USE_STD_HEADERS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <quotient.cuh>
#include <cuckoo.cuh>
#include <iceberg.cuh>
