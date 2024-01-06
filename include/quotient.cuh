#pragma once
#include <cassert>
#include <exception>
#include <utility>

// We assume keys are brought in in 64 bits and addresses are 32 bits
// This is a safe choice, since each bucket is usually a cache line, e.g. 128 bytes.
// (And current GPU memory does not exceed 2^32 * 128 bytes.)
using key_type = unsigned long long;
using addr_type = uint32_t;

// AddRem is a pair of an address and a remainder, with a stored address width
// Technically an std::pair<addr_type, rem_type> with a compile-time addr_width
template <typename rem_type, uint8_t addr_width>
struct AddRem : public std::pair<addr_type, rem_type> {
	using std::pair<addr_type, rem_type>::pair;
};

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

// Hashes x to [0, 2^target_width)
// The hash function family from BGHT
template <uint8_t index, uint32_t target_width>
__host__ __device__ constexpr inline uint32_t hash_base(const key_type x) {
	constexpr auto a = hash_constants[index].first;
	constexpr auto b = hash_constants[index].second;
	return (a * x + b) % large_prime % (1ul << target_width);
}

// Hashes x to [0, 2^target_width)
template <uint32_t target_width>
__host__ __device__ constexpr inline uint32_t hash(const uint8_t index, const key_type x) {
	// This explicit switch is necessary to allow function to be called from device
	assert(index < n_hash_functions);
	switch (index) {
		case 0: return hash_base<0, target_width>(x);
		case 1: return hash_base<1, target_width>(x);
		case 2: return hash_base<2, target_width>(x);
	}
	static_assert(2 == n_hash_functions - 1);
	// This cannot be reached, use std::unreachable() in C++23
	return 0;
}

// Split key into address and remainder
template <typename rem_type, uint8_t addr_width>
__host__ __device__ inline AddRem<rem_type, addr_width> split(const key_type k) {
	const uint32_t mask = (1ul << addr_width) - 1; // TODO: make mask func
	const uint32_t first = k & mask;
	const rem_type second = k >> addr_width;
	return {first, second};
}

// One-round Feistel network, modifying only the address bits
template <typename rem_type, uint8_t addr_width>
__host__ __device__ inline auto permute(uint8_t hash_index, const AddRem<rem_type, addr_width> addrem)
	-> AddRem<rem_type, addr_width> {
	return {
		addrem.first ^ hash<addr_width>(hash_index, addrem.second),
		addrem.second
	};
}

// The permutation is its own inverse
#define permute_inv permute

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("AddRem is a free abstraction over std::pair") {
	CHECK(sizeof(AddRem<uint32_t, 42>) == sizeof(std::pair<addr_type, uint32_t>));
}

TEST_CASE("permutation is as desired") {
	const key_type k = 14235234123; // TODO: randomize?
	const auto split_k = split<uint32_t, 13>(k);
	const auto once = permute(1, split_k);
	const auto twice = permute(1, once);
	const auto inv = permute_inv(1, once);
	CHECK(split_k != once); // with high probability
	CHECK(split_k.second == once.second); // only address part is modified
	CHECK(split_k == twice);
	CHECK(split_k == inv);
}
#endif
