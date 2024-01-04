#pragma once
#include <utility>
using std::pair;

// We assume that bucket addresses are uint32_t.
// This is a safe choice, since each bucket is usually a cache line, e.g. 128 bytes.
// (And current GPU memory does not exceed 2^32 * 128 bytes.)

// We also assume keys are brought in in 64 bits
using key_type = unsigned long long;

// Large prime taken from BGHT
constexpr uint32_t large_prime = 4294967291ul;

// Constants (a,b) for our hash functions
// Uniformly generated in the range [0, 2^32)
constexpr pair<uint32_t, uint32_t>  hash_constants[] = {
	{22205059, 940963638},
	{3910742802, 2487110075},
	{1028427014, 3103505973},
};

// The hash function from BGHT
// Hashes to the domain [0, 2^target_width - 1)
template <uint32_t a, uint32_t b, uint32_t target_width>
__host__ __device__ inline uint32_t hash_base(uint32_t x) {
	return a * x + b % large_prime % (1ul << target_width);
}

template <uint32_t target_width, uint8_t index>
__host__ __device__ inline uint32_t hash(uint32_t x) {
	constexpr auto a = hash_constants[index].first;
	constexpr auto b = hash_constants[index].second;
	return hash_base<a, b, target_width>(x);
}

template <typename row_type, uint8_t addr_width>
__host__ __device__ inline pair<uint32_t, row_type> split(const key_type k) {
	const uint32_t mask = (1ul << addr_width) - 1; // TODO: make mask func
	const uint32_t first = k & mask;
	const row_type second = k >> addr_width;
	return {first, second};
}

// One-round Feistel network
template <typename row_type, uint8_t addr_width, uint8_t index>
__host__ __device__ inline pair<uint32_t, row_type> permute(pair<uint32_t, row_type> in) {
	return {
		in.first ^ hash<addr_width, index>(in.second & ((1ull << addr_width) - 1)),
		in.second
	};
}

template <typename row_type, uint8_t addr_width, uint8_t index>
__host__ __device__ inline pair<uint32_t, row_type> to_entry(const key_type k) {
	return permute<row_type, addr_width, index>(split<row_type, addr_width>(k));
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("permutation is its own inverse") {
	const key_type k = 14235234123; // TODO: randomize?
	const auto split_k = split<uint32_t, 13>(k);
	const auto once = permute<uint32_t, 13, 1>(split_k);
	const auto twice = permute<uint32_t, 13, 1>(once);
	CHECK(split_k != once); // with high probability
	CHECK(split_k == twice);
}

// Force device code to be generated, should be expanded (preferably tests)
__global__ void do_hash(uint32_t x) {
	hash<10, 2>(x);
}
#endif
