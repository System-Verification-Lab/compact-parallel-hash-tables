// Example of using iceberg table from the host
//
// nvcc iceberg.cu -I../include --std=c++20 --expt-relaxed-constexpr -arch=sm_75 -o iceberg

#include "iceberg.cuh"

int main() {
	auto table = Iceberg<
		uint16_t, 32, // Primary slots are 16 bits wide, and there are 32 slots per bucket
		uint16_t, 16 // Secondary slots are 16 bits wide, and there are 16 slots per bucket
	>(20, 10, 7); // The keys are (at most) 20 bits wide, there are 2^10 primary buckets, and 2^7 secondary buckets.

	// Generate some keys with duplicates
	const auto n_keys  = 200;
	key_type *keys;
	cudaMallocManaged(&keys, sizeof(*keys) * n_keys);
	for (auto i = 0; i < n_keys; i++) keys[i] = (i / 2) << 10;

	// Find-or-put the keys, storing the results
	Result *results;
	cudaMallocManaged(&results, sizeof(*results) * n_keys);
	table.find_or_put(keys, keys + n_keys, results);

	// The concurrent writes did not introduce duplicates
	for (auto i = 0; i < n_keys / 2; i++) {
		auto a = i * 2;
		auto b = i * 2 + 1;
		assert((results[a] == Result::PUT && results[b] == Result::FOUND)
			|| (results[b] == Result::PUT && results[a] == Result::FOUND));
	}

	cudaFree(keys); cudaFree(results);
}
