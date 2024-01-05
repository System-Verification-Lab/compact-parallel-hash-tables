#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void cuda_assert(cudaError_t code, const char *file, const int line) {
	if (code == cudaSuccess) return;
	fprintf(stderr, "%s:%d CUDA error %s\n", file, line, cudaGetErrorString(code));
	std::exit(1);
}

#define CUDA(check) cuda_assert((check), __FILE__, __LINE__)
