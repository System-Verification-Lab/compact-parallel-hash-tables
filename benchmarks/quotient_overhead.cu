#include "cuda_util.cuh"
#include "quotient.cuh"
#include "timer.h"
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// This code compares the runtime overhead of permutations, both for a
// permutation whose constants are hard-coded as well as one where the
// constants are drawn at runtime (thus cannot be inlined by the compiler).
//
// The latter is used in our other benchmarks, and we simulate non-compactness
// by still using quotienting but storing the remainder in larger rows (instead
// of the full key). Thus these benchmarks show the performance overhead of
// these choices (which turns out to be negligible).

const int logN = 29;
const auto N = 1ull << logN;
const auto N_RUNS = 10;

// Single-round Feistel permutation based on the hash family of BGHT
// Hard-coded parameters (once uniformly drawn) for performance benefit
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

template <bool inv = false>
__global__ void hasher(auto permute, const key_type *keys, key_type *out) {
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	auto pk = permute(0, keys[index]);
	if constexpr (inv) {
		pk = permute.inv(1, pk); // avoid caching / optimizations
	}
	out[index] = pk;
}

void run(auto permute, key_type *keys, key_type *out) {
	Timer timer;

	// Warmup
	for (auto i = 0; i < N_RUNS; i++) hasher<false><<<N / 128, 128>>>(permute, keys, out);
	for (auto i = 0; i < N_RUNS; i++) hasher<true><<<N / 128, 128>>>(permute, keys, out);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());

	hasher<false><<<N / 128, 128>>>(permute, keys, out);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());
	timer.start();
	for (auto i = 0; i < N_RUNS; i++) hasher<false><<<N / 128, 128>>>(permute, keys, out);
	printf("permute:\t %f ms\n", timer.stop() / N_RUNS);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());

	hasher<true><<<N / 128, 128>>>(permute, keys, out);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());
	timer.start();
	for (auto i = 0; i < N_RUNS; i++) hasher<true><<<N / 128, 128>>>(permute, keys, out);
	printf("perm+inv:\t %f ms\n", timer.stop() / N_RUNS);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
	const BasicPermute bpermute(32);
	std::mt19937 rng;
	const RngPermute rpermute(uint8_t(32), rng);

	printf("Comparing permute vs permute+invert for 2^%d = %llu keys\n", logN, N);

	auto _keys = cusp(alloc_dev<key_type>(N));
	auto *keys = _keys.get();
	auto _out = cusp(alloc_dev<key_type>(N));
	auto *out = _out.get();

	thrust::sequence(thrust::device, keys, keys + N,
		key_type(0), key_type(std::numeric_limits<key_type>::max() / N));

	printf("BasicPermute\n");
	run(bpermute, keys, out);
	printf("\nRngPermute\n");
	run(rpermute, keys, out);
}
