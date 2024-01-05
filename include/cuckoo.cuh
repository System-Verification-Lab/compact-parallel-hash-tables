#pragma once
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bit>
#include "quotient.cuh"

// A Cuckoo hash table 
template <typename row_type, uint8_t bucket_size, uint8_t n_hash_functions = 3>
struct Cuckoo {
	static_assert(bucket_size > 0, "bucket size must be nonzero");
	static_assert(32 % bucket_size == 0, "warp/bucket size must divide 32");

public:
	const uint8_t key_width, addr_width, rem_width; // in bits
	const size_t n_rows;
	// status ::= unoccupied | occupied hash_id
	const uint8_t status_width = std::bit_width(n_hash_functions + 1u);

	row_type *rows; // the storage backend

	// Construct a Cuckoo hash table with 2^addr_width buckets
	// key_width and addr_width to be specified in bits
	Cuckoo(const uint8_t key_width, const uint8_t addr_width)
		: key_width(key_width)
		, addr_width(addr_width)
		, rem_width(key_width - addr_width)
		, n_rows((1ull << addr_width) * sizeof(*rows) * bucket_size) {
		// make sure row_type is wide enough
		assert(sizeof(row_type) * 8 >= status_width + rem_width);
		CUDA(cudaMallocManaged(&rows, n_rows));
		thrust::fill(thrust::device, rows, rows + n_rows, 0);
	}
};
