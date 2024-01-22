#include <cuda_util.cuh>
#include <cuckoo.cuh>
#include <iceberg.cuh>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>

// Runs per benchmark
static constexpr auto N_RUNS = 10; 
// Steps in find-or-put process. First run is always discarded.
static constexpr auto N_STEPS = 10;

struct Timer {
	cudaEvent_t before, after;
	float time;

	inline void start() {
		cudaEventRecord(before);
	}

	inline float stop() {
		cudaEventRecord(after);
		cudaEventSynchronize(after);
		cudaEventElapsedTime(&time, before, after);
		return time;
	}

	Timer() {
		cudaEventCreate(&before);
		cudaEventCreate(&after);
	}

	~Timer() {
		cudaEventDestroy(before);
		cudaEventDestroy(after);
	}
};

struct BenchResult {
	float average_ms;
};

/* TODO: not sure whether this organisation is worth it

struct BenchmarkDescription {
	std::string title;
	size_t num_keys;
	unsigned key_width;
	std::string keys_file;

	unsigned cuckoo_address_width;
	unsigned cuckoo_bucket_size;
	typename cuckoo_row_type;

	unsigned primary_address_width;
	unsigned primary_bucket_size;
	typename primary_row_type;
	unsigned secondary_address_width;
	unsigned secondary_bucket_size;
	typename secondary_row_type;
};

struct Benchmark {
	size_t num_keys;
	key_type *keys;
	BenchmarkResult run();
};

BenchmarkResult Benchmark::run() {
}

constexpr auto benchmarks {
	{ "Uniformly drawn", 20'000'000, 45, "benchmarks/data/1",
		25, 32, uint32_t, // 2*25 * 2*5 = 2^30 rows
		24, 32, uint32_t, 21, 16, uint64 }, // 2^20 * 2^5 + 2*21 * 2^4 = 2^29 + 2^25
};

void run_benchmark(BenchmarkDescription benchmark) {

}

int main() {
	for (auto benchmark : benchmarks) {
		run_benchmark(benchmark);
	}
}*/

template <uint8_t key_width, uint8_t address_width, uint8_t bucket_size, typename row_type>
BenchResult bench_cuckoo(key_type *keys, key_type *keys_end) {
	const auto len = keys_end - keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	using Table = Cuckoo<row_type, bucket_size>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(key_width, address_width);

	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));
	key_type *tmp;
	CUDA(cudaMallocManaged(&tmp, sizeof(*tmp) * len * 2));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		for (auto n = 0; n < len; n += len / N_STEPS) {
			table->find_or_put(keys, keys + n, tmp, results, false);
		}
		times_ms[i] = timer.stop();

		bool full = thrust::find(results, results + len, Result::FULL) != results + len;
		if (full) {
			fprintf(stderr, "bench: table was full! Results are not trustworthy");
			std::exit(1);
		}
	}

	CUDA(cudaFree(table));
	CUDA(cudaFree(results));
	CUDA(cudaFree(tmp));

	return { std::accumulate(times_ms + 1, times_ms + N_RUNS, 0.f) / (N_RUNS - 1) };
};

template <
	uint8_t key_width,
	uint8_t p_address_width, uint8_t p_bucket_size, typename p_row_type,
	uint8_t s_address_width, uint8_t s_bucket_size, typename s_row_type>
BenchResult bench_iceberg(key_type *keys, key_type *keys_end) {
	const auto len = keys_end - keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	using Table = Iceberg<p_row_type, p_bucket_size,
		s_row_type, s_bucket_size>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(key_width, p_address_width, s_address_width);

	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		for (auto n = 0; n < len; n += len / N_STEPS) {
			table->find_or_put(keys, keys + n, results, false);
		}
		times_ms[i] = timer.stop();

		bool full = thrust::find(results, results + len, Result::FULL) != results + len;
		if (full) {
			fprintf(stderr, "bench: table was full! Results are not trustworthy");
			std::exit(1);
		}
	}

	CUDA(cudaFree(table));
	CUDA(cudaFree(results));

	return { std::accumulate(times_ms + 1, times_ms + N_RUNS, 0.f) / (N_RUNS - 1) };
};

int main() {
	const std::string data_file = "benchmarks/data/1.bin";
	const size_t num_keys = 20'000'000;
	const uint8_t key_width = 45;

	// Cuckoo
	const unsigned cuckoo_address_width = 25;
	const uint8_t cuckoo_bucket_size = 32;
	using cuckoo_row_type = uint32_t;

	// Iceberg
	const unsigned primary_address_width = 24;
	const uint8_t primary_bucket_size = 32;
	using primary_row_type = uint32_t;
	const unsigned secondary_address_width = 21;
	const uint8_t secondary_bucket_size = 16;
	using secondary_row_type = long long unsigned; // 64 bits

	// Read keys (and a bit of warmup)
	// Assumes the file contains a binary dump of a key_type array
	fprintf(stderr, "Reading keys from %s... ", data_file.c_str());
	std::ifstream input(data_file, std::ios::in | std::ios::binary);
	key_type *keys, *keys_end;
	CUDA(cudaMallocManaged(&keys, sizeof(*keys) * num_keys));
	keys_end = keys + num_keys;
	if (!input.read((char*)keys, num_keys * sizeof(*keys))) {
		fprintf(stderr, "failed\n");
		std::exit(1);
	}
	thrust::all_of(thrust::device, keys, keys_end, thrust::identity<key_type>());
	fprintf(stderr, "done.\n");

	printf("Cuckoo average..."); fflush(stdout);
	auto cuckoo_result = bench_cuckoo<key_width, cuckoo_address_width,
		cuckoo_bucket_size, cuckoo_row_type>(keys, keys_end);
	printf(" %f ms\n", cuckoo_result.average_ms);

	printf("Iceberg average..."); fflush(stdout);
	auto iceberg_result = bench_iceberg<key_width,
		primary_address_width, primary_bucket_size, primary_row_type,
		secondary_address_width, secondary_bucket_size, secondary_row_type>
		(keys, keys_end);
	printf(" %f ms\n", iceberg_result.average_ms);

	CUDA(cudaFree(keys));
}
