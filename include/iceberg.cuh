#pragma once

#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
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
// The permutation object permute is initialzed as Permute(key_width, rng).
// Keys are permuted with permute(hash_id, key), and inverted with
// permute.inv(hash_id, permuted_key).
//
// TODO: There is some code duplication between the two levels,
// which could perhaps be reduced using fancy compile-time abstractions.
// Could also do with more tests.
template <
	typename p_row_type, uint8_t p_bucket_size,
	typename s_row_type, uint8_t s_bucket_size,
	class Permute = RngPermute,
	bool unified_memory = false // useful for debugging purposes
>
class Iceberg {
	static_assert(p_bucket_size > 0
		&& s_bucket_size > 0, "bucket size must be nonzero");
	static_assert(32 % p_bucket_size == 0
		&& 32 % s_bucket_size == 0, "warp/bucket size must divide 32");

public:
	static constexpr int block_size = 128;
	static_assert(block_size % p_bucket_size == 0);

	const Permute permute;

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
	CuSP<p_row_type> _p_rows; // shared pointer handling memory
	s_row_type *s_rows;
	CuSP<s_row_type> _s_rows; // shared pointer handling memory

	// Hash key to an address and a row entry
	__host__ __device__ PAddrRow p_addr_row(const key_type k) {
		const auto pk = permute(0, k);
		const addr_type addr = pk & mask<key_type>(p_addr_width);
		const auto rem = pk >> p_addr_width;
		return { addr, (p_row_type(1) << (p_row_width - p_state_width)) | rem };
	}

	__host__ __device__ SAddrRow s_addr_row(const uint8_t hash_id, const key_type k) {
		const auto pk = permute(hash_id, k);
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
		return permute.inv(hash_id, pk);
	}

	// Restore key from address and row
	__host__ __device__ key_type s_restore_key(const addr_type addr, const s_row_type row) {
		assert(row != 0);
		const auto hash_id = (row >> (s_row_width - s_state_width)) - 1;
		const auto rem = row & mask<s_row_type>(s_rem_width);
		const auto pk = (rem << s_addr_width) | addr;
		return permute.inv(hash_id, pk);
	}

	// Count the number of occurrences of key k in the table
	//
	// Only works from host when unified_memory is used
	__host__ __device__ unsigned count(const key_type k) {
#ifndef  __CUDA_ARCH__
		assert(unified_memory);
#endif
		unsigned count = 0;

		// Primary
		const auto [a0, r0] = p_addr_row(k);
		for (auto bi = 0; bi < p_bucket_size; bi++) {
			if (p_rows[a0 * p_bucket_size + bi] == r0) count++;
		}

		// Secondary
		const auto [a1, r1] = s_addr_row(1, k);
		const auto [a2, r2] = s_addr_row(2, k);
		for (auto bi = 0; bi < s_bucket_size; bi++) {
			if (s_rows[a1 * s_bucket_size + bi] == r1) count++;
			if (a1 != a2 && s_rows[a2 * s_bucket_size + bi] == r2) count++;
		}

		return count;
	}

	// Divide work between tiled threads
	//
	// For every key, F(key, tile) is called by some PTile,
	// its return value stored in results
	//
	// Assumes a 1d thread layout, and that p_bucket_size divides blockDim.x
	template <auto F, class KeyIt, class ResIt>
	__device__ void coop(const KeyIt start, const KeyIt end, ResIt results) {
		const auto index = blockIdx.x * blockDim.x + threadIdx.x;
		const auto stride = gridDim.x * blockDim.x;
		const auto len = end - start;
		const auto max = ((len + p_bucket_size - 1) / p_bucket_size) * p_bucket_size;

		for (auto i = index; i < max; i += stride) {
			key_type k;
			bool to_act = i < len;
			if (to_act) k = start[i];

			const auto thb = cg::this_thread_block();
			const auto tile = cg::tiled_partition<p_bucket_size>(thb);
			const auto rank = tile.thread_rank();
			while (auto queue = tile.ballot(to_act)) {
				const auto leader = __ffs(queue) - 1;
				const auto res = std::invoke(F, this,
						tile.shfl(k, leader), tile);
				if (rank == leader) {
					results[i] = res;
					to_act = false;
				}
			}
		}
	}

	// Cooperatively find-or-put key k
	//
	// All threads in the tile must receive the same parameter k
	__device__ Result coop_find_or_put(const key_type k, PTile tile) {
		using enum Result;
		const auto rank = tile.thread_rank();

		// Primary
		{
		auto [a0, r0] = p_addr_row(k);
		p_row_type v = 0;
		while (true) {
			const auto rid = a0 * p_bucket_size + rank;
			if (v == 0) v = volatile_load(p_rows + rid);
			if (tile.any(v == r0)) return FOUND;

			const auto load = __popc(tile.ballot(v != 0));
			if (load == p_bucket_size) break; // to secondary

			if (rank == load) {
				v = atomicCAS(p_rows + a0 * p_bucket_size + load, p_row_type(0), r0);
			}
			if (tile.shfl(v, load) == 0) return PUT;
		}
		}

		// Secondary level
		// We divide the tile in two subgroups,
		// inspecting one secondary bucket each.
		//
		// NOTE: we assume that a thread is in tile
		//	rank / (p_bucket_size / 2).
		// This seems to be the case, but it isn't well documented.
		static_assert(s_bucket_size * 2 <= p_bucket_size);
		const auto subgroup = cg::tiled_partition<p_bucket_size / 2>(tile);
		const auto hashid = subgroup.meta_group_rank() + 1;
		const auto subrank = subgroup.thread_rank();
		const bool to_act = subrank < s_bucket_size;

		const auto [a1, r1] = s_addr_row(hashid, k);
		s_row_type v = 0;
		while (true) {
			// Inspect buckets
			if (to_act && v == 0) v = volatile_load(s_rows + a1 * s_bucket_size + subrank);
			const bool found = (v == r1) && to_act;
			if (tile.any(found)) return FOUND;

			// Compare loads
			const auto load = __popc(subgroup.ballot((v != 0) && to_act));
			const auto load1 = tile.shfl(load, 0); // first subgroup
			const auto load2 = tile.shfl(load, p_bucket_size / 2); // second subgroup
			if (load1 == s_bucket_size && load2 == s_bucket_size) return FULL;

			// Insert in least full bucket (when tied, in the second)
			// This is where we use the assumption on partition tiling.
			const auto leader = load1 < load2 ? load1 : p_bucket_size / 2 + load2;
			if (rank == leader) {
				v = atomicCAS(s_rows + a1 * s_bucket_size + load, 0, r1);
			}
			if (tile.shfl(v, leader) == 0) return PUT;
		}
	}

	// Find-or-put keys between start (inclusive) and end (exclusive)
	//
	// Assumes a 1d thread layout, and that p_bucket_size divides blockDim.x
	__device__ void _find_or_put(const key_type *start, const key_type *end, Result *results) {
		return coop<&Iceberg::coop_find_or_put>(start, end, results);
	}

	// Find-or-put keys between start (inclusive) and end (exclusive)
	// If sync is true (by default, it is), a cuda device synchronization is performed
	//
	// To control thread layout, use the find_or_put kernel directly
	template <class KeyIt, class ResIt>
	void find_or_put(const KeyIt start, const KeyIt end, ResIt results, bool sync = true) {
		const int n_blocks = ((end - start) + block_size - 1) / block_size;
		invoke_device<&Iceberg::coop<&Iceberg::coop_find_or_put, KeyIt, ResIt>>
			<<<n_blocks, block_size>>>(*this, start, end, results);
		if (sync) CUDA(cudaDeviceSynchronize());
	}

	__device__ bool coop_find(const key_type k, const PTile tile) {
		const auto rank = tile.thread_rank();

		// Primary
		const auto [a0, r0] = p_addr_row(k);
		const auto rid = a0 * p_bucket_size + rank;
		const auto row = p_rows[rid];
		if (tile.any(row == r0)) return true;
		// Here we use the assumption that no keys are ever (re)moved
		// (so if we find an empty spot in level 1, we are done)
		if (tile.any(row == 0)) return false;

		// Secondary level
		// We divide the tile in two groups, inspecting one bucket each
		static_assert(s_bucket_size * 2 <= p_bucket_size);
		const auto subgroup = cg::tiled_partition<p_bucket_size / 2>(tile);
		const auto hashid = subgroup.meta_group_rank() + 1;
		const auto subrank = subgroup.thread_rank();
		const bool to_act = subrank < s_bucket_size;
		const auto [a1, r1] = s_addr_row(hashid, k);

		p_row_type v;
		if (to_act) v = s_rows[a1 * s_bucket_size + subrank];
		const bool found = (v == r1) && to_act;
		if (tile.any(found)) return true;

		return false;
	}

	// Find keys
	template <class KeyIt, class BoolIt>
	void find(const KeyIt start, const KeyIt end, BoolIt results, bool sync = true) {
		const int n_blocks = ((end - start) + block_size - 1) / block_size;
		invoke_device<&Iceberg::coop<&Iceberg::coop_find, KeyIt, BoolIt>>
			<<<n_blocks, block_size>>>(*this, start, end, results);
		if (sync) CUDA(cudaDeviceSynchronize());
	}

	// Put key WITHOUT checking if the key is already in the table
	__device__ Result coop_put(const key_type k, const PTile tile) {
		using enum Result;
		const auto rank = tile.thread_rank();

		// Primary
		{
		auto [a0, r0] = p_addr_row(k);
		p_row_type v = 0;
		while (true) {
			const auto rid = a0 * p_bucket_size + rank;
			if (v == 0) v = volatile_load(p_rows + rid);
			const auto load = __popc(tile.ballot(v != 0));
			if (load == p_bucket_size) break; // to secondary

			if (rank == load) {
				v = atomicCAS(p_rows + a0 * p_bucket_size + load, p_row_type(0), r0);
			}
			if (tile.shfl(v, load) == 0) return PUT;
		}
		}

		// Secondary level
		// We divide the tile in two subgroups,
		// inspecting one secondary bucket each.
		//
		// NOTE: we assume that a thread is in tile
		//	rank / (p_bucket_size / 2).
		// This seems to be the case, but it isn't well documented.
		static_assert(s_bucket_size * 2 <= p_bucket_size);
		const auto subgroup = cg::tiled_partition<p_bucket_size / 2>(tile);
		const auto hashid = subgroup.meta_group_rank() + 1;
		const auto subrank = subgroup.thread_rank();
		const bool to_act = subrank < s_bucket_size;

		const auto [a1, r1] = s_addr_row(hashid, k);
		s_row_type v = 0;
		while (true) {
			// Inspect buckets
			if (to_act && v == 0) v = volatile_load(s_rows + a1 * s_bucket_size + subrank);

			// Compare loads
			const auto load = __popc(subgroup.ballot((v != 0) && to_act));
			const auto load1 = tile.shfl(load, 0); // first subgroup
			const auto load2 = tile.shfl(load, p_bucket_size / 2); // second subgroup
			if (load1 == s_bucket_size && load2 == s_bucket_size) return FULL;

			// Insert in least full bucket (when tied, in the second)
			// This is where we use the assumption on partition tiling.
			const auto leader = load1 < load2 ? load1 : p_bucket_size / 2 + load2;
			if (rank == leader) {
				v = atomicCAS(s_rows + a1 * s_bucket_size + load, 0, r1);
			}
			if (tile.shfl(v, leader) == 0) return PUT;
		}
	}

	// Put keys WITHOUT checking if the key is already in the table
	template <class KeyIt, class ResIt>
	void put(const KeyIt start, const KeyIt end, ResIt results, bool sync = true) {
		const int n_blocks = ((end - start) + block_size - 1) / block_size;
		invoke_device<&Iceberg::coop<&Iceberg::coop_put, KeyIt, ResIt>>
			<<<n_blocks, block_size>>>(*this, start, end, results);
		if (sync) CUDA(cudaDeviceSynchronize());
	}

	// Clears all rows
	void clear() {
		thrust::fill(thrust::device, p_rows, p_rows + p_n_rows, 0);
		thrust::fill(thrust::device, s_rows, s_rows + s_n_rows, 0);
	}

	// Construct an Iceberg hash table with 2^primary_addr_width primary buckets
	// and 2^secondary_addr_width secondary buckets. Optionally, the
	// permutation is initialized with the given RNG.
	//
	// It is very important that buckets fit in a single cache line.
	//
	// See Iceberg paper for their recommendations. In particular:
	// - spend most storage budget on primary buckets (one cache line per bucket)
	// - have about 1/8 as many secondary buckets, with about 1/8 the bucket size
	//
	// Buckets must be aligned to cache lines for efficiency
	Iceberg(const uint8_t key_width,
		const uint8_t primary_addr_width, const uint8_t secondary_addr_width,
		std::optional<Rng> rng = std::nullopt)
		: permute(key_width, rng)
		, p_addr_width(primary_addr_width)
		, p_rem_width(key_width - p_addr_width)
		, p_n_rows((1ull << p_addr_width) * sizeof(*p_rows) * p_bucket_size)
		, s_addr_width(secondary_addr_width)
		, s_rem_width(key_width - s_addr_width)
		, s_n_rows((1ull << s_addr_width) * sizeof(*s_rows) * s_bucket_size)
	{
		// make sure row_type is wide enough
		assert(sizeof(p_row_type) * 8 >= p_state_width + p_rem_width);
		assert(sizeof(s_row_type) * 8 >= s_state_width + s_rem_width);
		_p_rows = cusp(alloc<p_row_type>(p_n_rows, unified_memory));
		p_rows = _p_rows.get();
		_s_rows = cusp(alloc<s_row_type>(s_n_rows, unified_memory));
		s_rows = _s_rows.get();
		clear();
	}
};

#ifdef DOCTEST_LIBRARY_INCLUDED
#include <thrust/count.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

using namespace kh;

TEST_CASE("Iceberg hash table") {
	// TODO 16 bit width only on high CC
	using Table = Iceberg<uint32_t, 32, uint32_t, 16, RngPermute, true>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(21, 6, 3);

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

	table->~Table();
	CUDA(cudaFree(keys));
	CUDA(cudaFree(results));
	CHECK(!cudaFree(table));
}

TEST_CASE("Iceberg convenience find_or_put member function") {
	constexpr auto n = 1000;
	key_type *keys;
	Result *results;
	CUDA(cudaMallocManaged(&keys, sizeof(*keys) * n));
	CUDA(cudaMallocManaged(&results, sizeof(*results) * n));
	thrust::sequence(keys, keys + n);

	using Table = Iceberg<uint32_t, 32, uint32_t, 16>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(21, 5, 3);

	table->find_or_put(keys, keys + n, results);
	CHECK(thrust::all_of(thrust::device, keys, keys + n,
		[table, results] __device__ (auto key) {
			return table->count(key) == 1 && results[key] == Result::PUT;
		}));

	table->~Table();
	CUDA(cudaFree(keys));
	CUDA(cudaFree(results));
	CUDA(cudaFree(table));
}

TEST_CASE("Iceberg: put and find") {
	constexpr auto n = 1000;
	key_type *keys;
	Result *results;
	bool *found;
	CUDA(cudaMallocManaged(&keys, sizeof(*keys) * n));
	CUDA(cudaMallocManaged(&found, sizeof(*found) * n));
	CUDA(cudaMallocManaged(&results, sizeof(*results) * n));
	thrust::sequence(keys, keys + n);

	using Table = Iceberg<uint32_t, 32, uint32_t, 16>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(21, 5, 3);

	table->put(keys, keys + n / 2, results);
	table->find(keys, keys + n, found);
	CHECK(thrust::all_of(thrust::device, keys, keys + n,
		[table, found, results] __device__ (auto key) {
			auto c = table->count(key);
			return (key < n / 2)
				? (c == 1 && found[key] && results[key] == Result::PUT)
				: (c == 0 && !found[key]);
		}));

	table->~Table();
	CUDA(cudaFree(keys));
	CUDA(cudaFree(results));
	CUDA(cudaFree(table));
}

TEST_CASE("Iceberg: 16-bit") {
	const auto n = 1000;
	auto table = Iceberg<uint16_t, 32, uint16_t, 16, RngPermute, true>(21, 6, 7);
	auto _keys = cusp(alloc_man<key_type>(n));
	auto *keys = _keys.get();
	auto _results = cusp(alloc_man<Result>(n));
	auto *results = _results.get();
	auto _found = cusp(alloc_man<bool>(n));
	auto *found = _found.get();
	thrust::sequence(keys, keys + n);
	table.put(keys, keys + n / 2, results);
	table.find(keys, keys + n, found);
	CHECK(thrust::all_of(keys, keys + n,
		[&table, found, results] (auto key) {
			auto c = table.count(key);
			return (key < n / 2)
				? (c == 1 && found[key] && results[key] == Result::PUT)
				: (c == 0 && !found[key]);
		}));
}

// We test level 2 using a small table
// with 1 primary bucket of size 4 and two secondary of size 2
// The permutation ensures that all keys hash to both secondary buckets
struct SmallPermute {
	__host__ __device__ key_type operator()(const uint8_t hid, const key_type k) const {
		if (hid == 0) return k;
		if (hid == 1) return k;
		return k ^ 1;
	}

	SmallPermute(auto, auto) {}
};

TEST_CASE("Iceberg hash table: level 2 find-or-put") {
	using Table = Iceberg<uint32_t, 4, uint32_t, 2, SmallPermute, true>;
	Table *table;
	CHECK(!cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(21, 0, 1); // one primary bucket, two secondary

	// We'll work with keys [0, 8], of which [0, 7] will just fit and 8 will not
	// To test for potential race problems, we insert each key many times
	const auto last_val = 9;
	const auto dups = 10;
	const auto N = last_val * dups;
	key_type *keys;
	Result *results;
	CUDA(cudaMallocManaged(&keys, sizeof(*keys) * N));
	CUDA(cudaMallocManaged(&results, sizeof(*results) * N));

	// The first dups keys will be 0, the next dups keys 1, etc.
	for (auto i = 0; i < N; i++) keys[i] = i / dups;

	// Fill the primary bucket with [0, 3]
	find_or_put<<<2, 64>>>(table, keys, keys + 4 * dups, results);
	CHECK(!cudaDeviceSynchronize());
	for (auto i = 0; i < 4; i++) CHECK(table->count(i) == 1);
	for (auto i = 4; i < 9; i++) CHECK(table->count(i) == 0);
	CHECK(thrust::count(results, results + 4 * dups, Result::PUT) == 4);
	CHECK(thrust::count(results, results + 4 * dups, Result::FOUND) == 4 * (dups - 1));

	// Now we fill both secondary buckets with [4, 7]
	find_or_put<<<2, 64>>>(table, keys, keys + 8 * dups, results);
	CHECK(!cudaDeviceSynchronize());
	CHECK(thrust::count(results, results + 8 * dups, Result::PUT) == 4);
	CHECK(thrust::count(results, results + 8 * dups, Result::FULL) == 0);
	for (auto i = 0; i < 8; i++) CHECK(table->count(i) == 1);

	// All buckets are full, so number 8 cannot be added
	find_or_put<<<2, 64>>>(table, keys, keys + 9 * dups, results);
	CHECK(!cudaDeviceSynchronize());
	CHECK(thrust::count(results, results + 8 * dups, Result::FOUND) == 8 * dups);
	CHECK(thrust::count(results, results + 9 * dups, Result::FULL) == dups);

	for (auto i = 0; i < 8; i++) CHECK(table->count(i) == 1);
	CHECK(table->count(8) == 0);

	CUDA(cudaFree(keys));
	CUDA(cudaFree(results));
	CUDA(cudaFree(table));
}

#endif
