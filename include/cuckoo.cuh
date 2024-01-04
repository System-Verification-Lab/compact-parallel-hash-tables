#pragma once
#include <thrust/fill.h>
#include "bit_width.h"
#include "quotient.h"

// A Cuckoo hash table 
template <typename row_type, typename bucket_size, uint8_t n_hash_functions = 3>
struct Cuckoo {
	static_assert(bucket_size > 0, "bucket size must be nonzero");
	static_assert(32 % bucket_size == 0, "warp/bucket size must divide 32");

public:
	const uint8_t key_width, addr_width, rem_width; // in bits
	const size_t n_rows;
	constexpr uint8_t status_width = bit_width(n_hash_functions + 1); // unoccupied or hash id

	row_type *rows; // the storage backend

	// Construct a Cuckoo hash table with 2^addr_width buckets
	// key_width and addr_width to be specified in bits
	Cuckoo(const uint8_t key_width, const uint8_t addr_width)
		: key_width(key_width)
		, addr_width(addr_width)
		, rem_width(key_width - addr_width)
		, n_rows((1ull << addr_width) * sizeof(*rows) * bucket_size) {
		assert("remainders do not fit in row_type",
			sizeof(row_type) * 8 >= status_width + rem_width);
		cuda(cudaMallocManaged(&rows, n_rows);
		thrust::fill(thrust::device, rows, rows + n_rows, 0);
	}
};
