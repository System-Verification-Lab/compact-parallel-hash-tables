#include "cuda_util.cuh"
#include "quotient.cuh"
#include "timer.h"
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

template <bool inv = false>
__global__ void hasher(auto permute, const key_type *keys, key_type *out) {
	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	auto pk = permute(0, keys[index]);
	if constexpr (inv) {
		pk = permute.inv(0, pk);
	}
	out[index] = pk;
}

int main(int argc, char **argv) {
	const int logN = 29;
	const auto N = 1ull << logN;
	const auto N_RUNS = 10;
	const BasicPermute permute(32);
	Timer timer;

	printf("Comparing permute vs permute+invert for 2^%d = %llu keys\n", logN, N);

	auto _keys = cusp(alloc_dev<key_type>(N));
	auto *keys = _keys.get();
	auto _out = cusp(alloc_dev<key_type>(N));
	auto *out = _out.get();

	thrust::sequence(thrust::device, keys, keys + N,
		key_type(0), key_type(std::numeric_limits<key_type>::max() / N));

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
