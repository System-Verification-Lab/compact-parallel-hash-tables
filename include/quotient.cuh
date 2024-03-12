#pragma once
#include <cassert>
#include <exception>
#include <random>
#include <utility>
#include "bits.h"

// We assume keys are brought in in 64 bits (TODO: template) and addresses are
// 32 bits. This is a safe choice, since each bucket is usually a cache line,
// e.g. 128 bytes. (And current GPU memory does not exceed 2^32 * 128 bytes.)
using key_type = unsigned long long;
using addr_type = uint32_t;

// WARNING: note that std::function takes BY VALUE, so a generator object
// should be passed as std::ref(object) most of the time. In the future, this
// could be an std::function_ref (in C++26 hopefully).
using Rng = std::function<uint32_t()>;

// Single-round Feistel permutation based on the hash family of BGHT
// Parameters drawn from generator provided in constructor
class RngPermute {
	static constexpr uint8_t n_hash_functions = 3;
	const uint8_t hash_width; // part of key that is hashed

	// Large prime from BGHT. This is 2^32 - 5.
	static constexpr uint32_t large_prime = 4294967291ul;

	// Constants (a,b) for our hash functions
	std::pair<uint32_t, uint32_t>  hash_constants[n_hash_functions];

	// Hashes x to [0, 2^hash_width) -- so long hash_width < 32
	// The hash function family from BGHT
	__host__ __device__ inline uint32_t hash(const uint8_t index, const key_type x) const {
		assert(index < n_hash_functions);
		auto [a, b] = hash_constants[index];
		return (a * x + b) % large_prime % (1ul << hash_width);
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

	// Instantiate permutations for keys of key_width, optionally
	// generating constants with rng().
	//
	// NOTE / WARNING: rng is taken BY VALUE, not by reference. Meaning
	// that, to call this with a random number generator object gen, you will
	// likely want to call this with std::ref(gen). In the future (C++26?),
	// std::function should be replaced with std::function_ref.
	RngPermute(const uint8_t key_width,
		std::optional<Rng> rng = std::nullopt)
		: hash_width(std::min(key_width / 2, 31)) {
		assert(key_width < sizeof(key_type) * 8);
		assert(hash_width < 32);

		auto gen = rng.value_or(std::mt19937());
		for (auto i = 0; i < n_hash_functions; i++) {
			hash_constants[i].first = gen();
			hash_constants[i].second = gen();
		}
	};
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("RngPermute is as desired") {
	std::mt19937 rng;
	auto permute = RngPermute(31, rng);
	const key_type k = 0b110101110101; // TODO: randomize?
	const auto once = permute(2, k);
	const auto twice = permute(2, once);
	const auto inv = permute.inv(2, once);
	CHECK(k != once); // with high probability
	CHECK(twice == k);
	CHECK(inv == k);
}
#endif
