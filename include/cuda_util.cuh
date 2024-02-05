#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>

inline void cuda_assert(cudaError_t code, const char *file, const int line) {
	if (code == cudaSuccess) return;
	fprintf(stderr, "%s:%d CUDA error %s\n", file, line, cudaGetErrorString(code));
	std::exit(1);
}

#define CUDA(check) cuda_assert((check), __FILE__, __LINE__)

// A kernel that calls __device__ function F with given arguments
//
// As __global__ functions cannot be member functions and __device__ functions
// can, this can be used to "wrap" a member __device__ function in a kernel.
template <auto F, class... Args>
__global__ void invoke_device(Args... args) {
	std::invoke(F, args...);
}
