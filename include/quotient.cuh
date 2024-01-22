#pragma once
#include <cassert>
#include <exception>
#include <utility>
#include "bits.h"

// We assume keys are brought in in 64 bits (TODO: template) and addresses are
// 32 bits. This is a safe choice, since each bucket is usually a cache line,
// e.g. 128 bytes. (And current GPU memory does not exceed 2^32 * 128 bytes.)
using key_type = unsigned long long;
using addr_type = uint32_t;

// Single-round Feistel permutation based on the hash family of BGHT
// In a class so that it can easily be replaced later
class BasicPermute {
	const uint8_t hash_width; // part of key that is hashed

	// Large prime from BGHT. This is 2^32 - 5.
	static constexpr uint32_t large_prime = 4294967291ul;

	// Constants (a,b) for our hash functions
	// Uniformly generated in the range [0, 2^32)
	static constexpr uint8_t n_hash_functions = 3;
	static constexpr std::pair<uint32_t, uint32_t>  hash_constants[] = {
		{22205059, 940963638},
		{3910742802, 2487110075},
		{1028427014, 3103505973},
	};

	// Hashes x to [0, 2^hash_width) -- so long hash_width < 32
	// The hash function family from BGHT
	template <uint8_t index>
	__host__ __device__ inline uint32_t hash_base(const key_type x) const {
		constexpr auto a = hash_constants[index].first;
		constexpr auto b = hash_constants[index].second;
		return (a * x + b) % large_prime % (1ul << hash_width);
	}

	// Hashes x to [0, 2^hash_width) -- so long hash_width < 32
	__host__ __device__ inline uint32_t hash(const uint8_t index, const key_type x) const {
		// This explicit switch is necessary to allow function to be called from device
		assert(index < n_hash_functions);
		static_assert(2 == n_hash_functions - 1);
		switch (index) {
			case 0: return hash_base<0>(x);
			case 1: return hash_base<1>(x);
			case 2: return hash_base<2>(x);
			// use std::unreachable() in C++23
			default: __builtin_unreachable();
		}
	}

public:
	// One-round Feistel permutation
	// Slight inbalance because our hash function only hashes up to 32 bits
	__host__ __device__ inline key_type operator()(const uint8_t index, const key_type x) const {
		return hash(index, x >> hash_width) ^ x;
	}

	// Inverse of permute
	// (A one-round Feistel permutation is its own inverse)
	__host__ __device__ inline key_type inv(const uint8_t index, const key_type x) const {
		return operator()(index, x);
	}

	BasicPermute(const uint8_t key_width) : hash_width(std::min(key_width / 2, 31)) {
		assert(key_width < sizeof(key_type) * 8);
		assert(hash_width < 32);
	};
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("BasicPermute is as desired") {
	auto permute = BasicPermute(31);
	const key_type k = 0b110101110101; // TODO: randomize?
	const auto once = permute(2, k);
	const auto twice = permute(2, once);
	const auto inv = permute.inv(2, once);
	CHECK(k != once); // with high probability
	CHECK(twice == k);
	CHECK(inv == k);
}
#endif
