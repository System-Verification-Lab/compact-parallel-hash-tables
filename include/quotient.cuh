#pragma once
#include <utility>

// We assume keys are brought in in 64 bits and addresses are 32 bits
// This is a safe choice, since each bucket is usually a cache line, e.g. 128 bytes.
// (And current GPU memory does not exceed 2^32 * 128 bytes.)
using key_type = unsigned long long;
using addr_type = uint32_t;

template <typename rem_type>
using AddRem = std::pair<addr_type, rem_type>;

// Large prime from BGHT. This is 2^32 - 5, so supports up to 2^32 - 5 addresses
constexpr addr_type large_prime = 4294967291ul;

// Constants (a,b) for our hash functions
// Uniformly generated in the range [0, 2^32)
constexpr uint8_t n_hash_functions = 3;
constexpr std::pair<uint32_t, uint32_t>  hash_constants[] = {
	{22205059, 940963638},
	{3910742802, 2487110075},
	{1028427014, 3103505973},
};
static_assert(sizeof(hash_constants) / sizeof(hash_constants[0]) >= n_hash_functions);

// The hash function from BGHT
// Hashes to the domain [0, 2^m), where m = min(2^addr_width, large_prime)
template <uint32_t a, uint32_t b, uint32_t target_width>
__host__ __device__ inline uint32_t hash_base(uint32_t x) {
	return (static_cast<uint64_t>(a) * x + b) % large_prime % (1ul << target_width);
}

template <uint32_t target_width, uint8_t index>
__host__ __device__ inline uint32_t hash(uint32_t x) {
	constexpr auto a = hash_constants[index].first;
	constexpr auto b = hash_constants[index].second;
	return hash_base<a, b, target_width>(x);
}

// Split key into address and remainder
template <typename rem_type, uint8_t addr_width>
__host__ __device__ inline AddRem<rem_type> split(const key_type k) {
	const uint32_t mask = (1ul << addr_width) - 1; // TODO: make mask func
	const uint32_t first = k & mask;
	const rem_type second = k >> addr_width;
	return {first, second};
}

// One-round Feistel network, modifying only the address bits
template <typename rem_type, uint8_t addr_width, uint8_t index>
__host__ __device__ inline AddRem<rem_type> permute(AddRem<rem_type> in) {
	return {
		in.first ^ hash<addr_width, index>(in.second & ((1ull << addr_width) - 1)),
		in.second
	};
}

// The permutation is its own inverse
template <typename rem_type, uint8_t addr_width, uint8_t index>
__host__ __device__ inline AddRem<rem_type> permute_inv(AddRem<rem_type> in) {
	return permute<rem_type, addr_width, index>(in);
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("permutation is its own inverse") {
	const key_type k = 14235234123; // TODO: randomize?
	const auto split_k = split<uint32_t, 13>(k);
	const auto once = permute<uint32_t, 13, 1>(split_k);
	const auto twice = permute<uint32_t, 13, 1>(once);
	const auto inv = permute_inv<uint32_t, 13, 1>(once);
	CHECK(split_k != once); // with high probability
	CHECK(split_k == twice);
	CHECK(split_k == inv);
}
#endif