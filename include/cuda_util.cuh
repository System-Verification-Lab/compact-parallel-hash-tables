#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <type_traits>

inline void cuda_assert(cudaError_t code, const char *file, const int line) {
	if (code == cudaSuccess) return;
	fprintf(stderr, "%s:%d CUDA error %s\n", file, line, cudaGetErrorString(code));
	std::abort();
}

#define CUDA(check) cuda_assert((check), __FILE__, __LINE__)

// A shared pointer intended for wrapping a device memory pointer
template <typename T>
using CuSP = std::shared_ptr<T>;

template <typename T>
CuSP<T> cusp(T *t) {
	return CuSP<T>(t, cudaFree);
}

// Device allocation
template <typename T>
T *alloc_dev(size_t count) {
	T *out;
	CUDA(cudaMalloc(&out, sizeof(T) * count));
	return out;
}

// Managed allocation
template <typename T>
T *alloc_man(size_t count) {
	T *out;
	CUDA(cudaMallocManaged(&out, sizeof(T) * count));
	return out;
}

// Allocate managed (managed = true) or ordinary device memory
template <typename T>
T *alloc(size_t count, bool managed) {
	if (managed) return alloc_man<T>(count);
	return alloc_dev<T>(count);
}

// A kernel that calls __device__ function F with given arguments
//
// As __global__ functions cannot be member functions and __device__ functions
// can, this can be used to "wrap" a member __device__ function in a kernel.
template <auto F, class... Args>
__global__ void invoke_device(Args... args) {
	std::invoke(F, args...);
}

// Volatile load
//
// This can be used to load up-to-date values for variables that have been
// modified using atomic operations. (CUDA does not guarantee that normal reads
// following atomic operations are current, they could come from local cache.)
//
// Based on http://wg21.link/P1382R0
template <typename T>
__host__ __device__ constexpr inline T volatile_load(const T *from)
requires std::is_trivially_copyable_v<T> {
	return *(volatile const T*)from;
}
