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
	std::mt19937 rng;
	const BasicPermute bpermute(32);
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
