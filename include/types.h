#pragma once
#include <cstdint>
#include <functional>

// Global type choices
//
// Separated so that code may use them without having to compile (CUDA)
// implementation, since other headers include implementation (the downside of
// a header-only library).

// Keys and addresses
//
// We assume keys are brought in in 64 bits and addresses are 32 bits. This is
// a safe choice, since each bucket is usually a cache line, e.g. 128 bytes.
// (And current GPU memory does not exceed 2^32 * 128 bytes.)
//
// TODO: would be nice if this could be configured on a per-table basis
using key_type = unsigned long long;
using addr_type = uint32_t;

// Random number generators
//
// Similarly, we assume permutations need random variables of at most 32 bits.
//
// WARNING: note that std::function takes function objects BY VALUE, so a
// generator object should likely be passed as std::ref(object). In the future,
// this could be an std::function_ref (C++26, hopefully).
using Rng = std::function<uint32_t()>;
