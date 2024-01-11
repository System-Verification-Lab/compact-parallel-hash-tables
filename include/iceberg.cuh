#pragma once

#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <bit>
#include <utility>
#include "bits.h"
#include "cuda_util.cuh"
#include "quotient.cuh"
#include "table.cuh"

namespace cg = cooperative_groups;

// An Iceberg hash table
//
// For storing keys of width key_width (in bits).
// The primary buckets consist of p_bucket_size rows of type p_row_type.
// The secondary buckets consist of s_bucket_size rows of type s_row_type.
// Keys are permuted with Permute::permute(hash_id, key), and inverted with
// Permute::permute_inv(hash_id, permuted_key).
//
// TODO: There is some code duplication between the two levels,
// which could perhaps be reduced using fancy compile-time abstractions.
// Could also do with more tests.
template <
	uint8_t key_width,
	typename p_row_type, uint8_t p_bucket_size,
	typename s_row_type, uint8_t s_bucket_size,
	typename Permute = BasicPermute<key_width>
>
class Iceberg {
	static_assert(p_bucket_size > 0
		&& s_bucket_size > 0, "bucket size must be nonzero");
	static_assert(32 % p_bucket_size == 0
		&& 32 % s_bucket_size == 0, "warp/bucket size must divide 32");

public:
	// Primary bits
	const uint8_t p_row_width = sizeof(p_row_type) * 8;
	const uint8_t p_addr_width, p_rem_width; // in bits
	const size_t p_n_rows;

	// Secondary bits
	const uint8_t s_row_width = sizeof(s_row_type) * 8;
	const uint8_t s_addr_width, s_rem_width; // in bits
	const size_t s_n_rows;

	// primary state ::= empty | occupied
	const uint8_t p_state_width = 1;

	// secondary state ::= empty | occupied hash_id
	const uint8_t s_state_width = 2;

	using PTile = cg::thread_block_tile<p_bucket_size, cg::thread_block>;
	using STile = cg::thread_block_tile<s_bucket_size, cg::thread_block>;
	using PAddrRow = std::pair<addr_type, p_row_type>;
	using SAddrRow = std::pair<addr_type, s_row_type>;

	// The storage backend
	// Primary rows consist of a remainder
	// Secondary rows are as follows:
	// - the 2 most significant bits indicate the state
	//   - state 0 is for empty rows (thus a row is empty if the row is 0)
	//   - state 1 + i indicates hash function i
	// - the least significant rem_width bits indicate the remainder
	p_row_type *p_rows;
	s_row_type *s_rows;

	// Hash key to an address and a row entry
	__host__ __device__ PAddrRow p_addr_row(const key_type k) {
		const auto pk = Permute::permute(0, k);
		const addr_type addr = pk & mask<key_type>(p_addr_width);
		const auto rem = pk >> p_addr_width;
		return { addr, (p_row_type(1) << (p_row_width - p_state_width)) | rem };
	}

	__host__ __device__ SAddrRow s_addr_row(const uint8_t hash_id, const key_type k) {
		const auto pk = Permute::permute(hash_id, k);
		const addr_type addr = pk & mask<key_type>(s_addr_width);
		const auto rem = pk >> s_addr_width;
		return { addr, s_row_type(hash_id + 1) << (s_row_width - s_state_width) | rem };
	}

	// Restore key from address and row
	__host__ __device__ key_type p_restore_key(const addr_type addr, const p_row_type row) {
		assert(row != 0);
		const auto hash_id = 0;
		const auto rem = row & mask<p_row_type>(p_rem_width);
		const auto pk = (rem << p_addr_width) | addr;
		return Permute::permute_inv(hash_id, pk);
	}

	// Restore key from address and row
	__host__ __device__ key_type s_restore_key(const addr_type addr, const s_row_type row) {
		assert(row != 0);
		const auto hash_id = (row >> (s_row_width - s_state_width)) - 1;
		const auto rem = row & mask<s_row_type>(s_rem_width);
		const auto pk = (rem << s_addr_width) | addr;
		return Permute::permute_inv(hash_id, pk);
	}

	// Count the number of occurrences of key k in the table
	// Unoptimized, for testing only
	__host__ __device__ unsigned count(const key_type k) {
		unsigned count = 0;

		// Primary
		const auto [a0, r0] = p_addr_row(k);
		for (auto bi = 0; bi < p_bucket_size; bi++) {
			if (p_rows[a0 * p_bucket_size + bi] == r0) count++;
		}

		// Secondary
		const auto [a1, r1] = s_addr_row(1, k);
		const auto [a2, r2] = s_addr_row(2, k);
		for (auto bi = 0; bi < p_bucket_size; bi++) {
			if (s_rows[a1 * s_bucket_size + bi] == r1) count++;
			if (s_rows[a2 * s_bucket_size + bi] == r2) count++;
		}

		return count;
	}

	// Cooperatively find-or-put key k
	//
	// All threads in the tile must receive the same parameter k
	__device__ Result coop_find_or_put(const key_type k, PTile tile) {
		using enum Result;
		const auto rank = tile.thread_rank();
		
		// Primary
		auto [a0, r0] = p_addr_row(k);
		while (true) {
			const auto rid = a0 * p_bucket_size + rank;
			auto v = p_rows[rid];
			if (tile.any(v == r0)) return FOUND;

			const auto load = __popc(tile.ballot(v != 0));
			if (load == p_bucket_size) break; // to secondary

			if (rank == load) v = atomicCAS(p_rows + a0 * p_bucket_size + load, p_row_type(0), r0);
			if (tile.shfl(v, load) == 0) return PUT;
		}

		// Secondary level
		// We divide the tile in two subgroups, inspecting one secondary bucket each
		static_assert(s_bucket_size * 2 <= p_bucket_size);
		const auto subgroup = cg::tiled_partition<p_bucket_size / 2>(tile);
		const auto hashid = subgroup.meta_group_rank() + 1;
		const auto subrank = subgroup.thread_rank();
		const bool to_act = subrank < s_bucket_size;

		while (true) {
			// Inspect buckets
			const auto [a1, r1] = s_addr_row(hashid, k);
			auto v = s_rows[a1 * s_bucket_size + subrank];
			const bool found = (v == r1) && to_act;
			if (tile.any(found)) return FOUND;

			// Compare loads
			const auto load = __popc(subgroup.ballot((v != 0) && to_act));
			const auto load1 = tile.shfl(load, 0); // first subgroup
			const auto load2 = tile.shfl(load, p_bucket_size / 2); // second subgroup
			if (load1 == s_bucket_size && load2 == s_bucket_size) return FULL;

			// Insert in least full bucket (when tied, in the second)
			const auto target = (load1 >= s_bucket_size);
			const auto leader = target * p_bucket_size / 2;
			if (rank == leader) {
				v = atomicCAS(s_rows + a1 * s_bucket_size + load, 0, r1);
			}
			if (tile.shfl(v, leader) == 0) return PUT;
		}
	}

	// Find-or-put keys between start (inclusive) and end (exclusive)
	//
	// Assumes a 1d thread layout, and that p_bucket_size divides blockDim.x
	__device__ void find_or_put(const key_type *start, const key_type *end, Result *results) {
		const auto index = blockIdx.x * blockDim.x + threadIdx.x;
		const auto stride = gridDim.x * blockDim.x;
		const auto len = end - start;
		// round to p_bucket_size groups (using integer division)
		const auto max = ((len + p_bucket_size - 1) / p_bucket_size) * p_bucket_size;

		for (auto i = index; i < max; i += stride) {
			key_type k;
			bool to_act = i < len;
			if (to_act) k = start[i];

			// Cooperative processing (group size = p_bucket_size)
			const auto thb = cg::this_thread_block();
			const auto tile = cg::tiled_partition<p_bucket_size>(thb);
			const auto rank = tile.thread_rank();
			while (auto queue = tile.ballot(to_act)) {
				const auto leader = __ffs(queue) - 1;
				const auto res = coop_find_or_put(tile.shfl(k, leader), tile);
				if (rank == leader) {
					results[i] = res;
					to_act = false;
				}
			}
		}
	}

	// Construct an Iceberg hash table with 2^primary_addr_width primary buckets
	// and 2^secondary_addr_width secondary buckets
	//
	// It is very important that buckets fit in a single cache line.
	//
	// See Iceberg paper for their recommendations. In particular:
	// - spend most storage budget on primary buckets (one cache line per bucket)
	// - have about 1/8 as many secondary buckets, with about 1/8 the bucket size
	//
	// Buckets must be aligned to cache lines for efficiency
	Iceberg(const uint8_t primary_addr_width, const uint8_t secondary_addr_width)
		: p_addr_width(primary_addr_width)
		, p_rem_width(key_width - p_addr_width)
		, p_n_rows((1ull << p_addr_width) * sizeof(*p_rows) * p_bucket_size)
		, s_addr_width(secondary_addr_width)
		, s_rem_width(key_width - s_addr_width)
		, s_n_rows((1ull << s_addr_width) * sizeof(*s_rows) * s_bucket_size)
	{
		// make sure row_type is wide enough
		assert(sizeof(p_row_type) * 8 >= p_state_width + p_rem_width);
		assert(sizeof(s_row_type) * 8 >= s_state_width + s_rem_width);
		CUDA(cudaMallocManaged(&p_rows, p_n_rows));
		CUDA(cudaMallocManaged(&s_rows, s_n_rows));
		thrust::fill(thrust::device, p_rows, p_rows + p_n_rows, 0);
		thrust::fill(thrust::device, s_rows, s_rows + s_n_rows, 0);
	}

	~Iceberg() {
		CUDA(cudaFree(p_rows));
		CUDA(cudaFree(s_rows));
	}
};

template <typename Table>
__global__ void find_or_put(Table *table, const key_type *start, const key_type *end, Result *results) {
	table->find_or_put(start, end, results);
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("Iceberg hash table") {
	// TODO: allow to swap out the permute function via template argument,
	// so that we can properly test level 2 behavior (generate conflicts).

	// TODO 16 bit width only on high CC
	using Table = Iceberg<21, uint32_t, 32, uint32_t, 16>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(6, 3);

	CHECK(table->count(0) == 0);

	const auto n = 300;
	key_type *keys;
	Result *results;
	CUDA(cudaMallocManaged(&keys, sizeof(*keys) * n));
	CUDA(cudaMallocManaged(&results, sizeof(*results) * n));
	thrust::fill(thrust::device, keys, keys + n, 0);
	find_or_put<<<2, 512>>>(table, keys, keys + n, results);
	CHECK(!cudaDeviceSynchronize());

	CHECK(table->count(0) == 1);
	CHECK(thrust::count(results, results + n, Result::PUT) == 1);
	CHECK(thrust::count(results, results + n, Result::FOUND) == n - 1);

	CHECK(!cudaFree(table));
}
#endif
