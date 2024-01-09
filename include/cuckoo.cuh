#pragma once

#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bit>
#include <utility>
#include "bits.h"
#include "cuda_util.cuh"
#include "quotient.cuh"

namespace cg = cooperative_groups;

// A Cuckoo hash table 
template <uint8_t key_width, typename row_type, uint8_t bucket_size, uint8_t n_hash_functions = 3>
class Cuckoo {
	static_assert(bucket_size > 0, "bucket size must be nonzero");
	static_assert(32 % bucket_size == 0, "warp/bucket size must divide 32");

public:
	const uint8_t row_width = sizeof(row_type) * 8;
	const uint8_t addr_width, rem_width; // in bits
	const size_t n_rows;
	// state ::= empty | occupied hash_id
	// TODO: For safety, it must also support `locked hash_id` (effectively an extra lock bit)
	const uint8_t state_width = std::bit_width(n_hash_functions);

	using Tile = cg::thread_block_tile<bucket_size, cg::thread_block>;
	using Permute = BasicPermute<key_width>;
	using AddrRow = std::pair<addr_type, row_type>;

	// The storage backend
	// Row entries are stored as follows:
	// - the most significant state_width bits indicate the state
	//   - state 0 is for empty rows
	//   - state 1 + i indicates hash function i
	// - the least significant rem_width bits indicate the remainder
	row_type *rows;

	// Hash key to an address and a row entry
	__host__ __device__ AddrRow addr_row(const uint8_t hash_id, const key_type k) {
		const auto pk = Permute::permute(hash_id, k);
		const addr_type addr = pk & mask<key_type>(addr_width);
		const auto rem = pk >> addr_width;
		return { addr, ((hash_id + 1) << (row_width - state_width)) | rem };
	}

	// Restore key from address and row
	__host__ __device__ key_type restore_key(const addr_type addr, const row_type row) {
		assert(row != 0);
		const auto hash_id = (row >> (row_width - state_width)) - 1;
		const auto rem = row & mask<row_type>(rem_width);
		const auto pk = (rem << addr_width) | addr;
		return Permute::permute_inv(hash_id, pk);
	}

	// Count the number of occurrences of key k in the table
	__host__ __device__ unsigned count(const key_type k) {
		unsigned count = 0;
		for (auto hid = 0; hid < n_hash_functions; hid++) {
			const auto [addr, row] = addr_row(hid, k);
			for (auto bi = 0; bi < bucket_size; bi++) {
				if (rows[addr * bucket_size + bi] == row) count++;
			}
		}
		return count;
	}

	// Construct a Cuckoo hash table with 2^addr_width buckets
	// addr_width to be specified in bits
	Cuckoo(const uint8_t addr_width)
		: addr_width(addr_width)
		, rem_width(key_width - addr_width)
		, n_rows((1ull << addr_width) * sizeof(*rows) * bucket_size)
	{
		// make sure row_type is wide enough
		assert(sizeof(row_type) * 8 >= state_width + rem_width);
		CUDA(cudaMallocManaged(&rows, n_rows));
		thrust::fill(thrust::device, rows, rows + n_rows, 0);
	}

	~Cuckoo() {
		CUDA(cudaFree(rows));
	}
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("Cuckoo hash table") {
	using Table = Cuckoo<17, uint16_t, 32>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(3);

	CHECK(table->count(0) == 0);

	CUDA(cudaFree(table));
	CHECK(true); // survived destruction
}
#endif
