#include <cuda_util.cuh>
#include <cuckoo.cuh>
#include <iceberg.cuh>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
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

enum TableType {
	CUCKOO,
	ICEBERG,
};

std::string to_string(TableType type) {
	switch (type) {
		case CUCKOO: return "Cuckoo";
		case ICEBERG: return "Iceberg";
		default: assert(false);
	}
}

struct TableConfig {
	TableType type;

	uint8_t p_addr_width;
	uint8_t p_row_width;
	uint8_t p_bucket_size;

	// Iceberg only
	uint8_t s_addr_width;
	uint8_t s_row_width;
	uint8_t s_bucket_size;

	// TODO: some stringstream magic
	std::string describe() const {
		return to_string(type);
	}
};

struct Benchmark {
	TableConfig config;
	uint8_t key_width;
	key_type *keys;
	key_type *keys_end;
};

template <typename row_type, uint8_t bucket_size>
BenchResult bench_cuckoo(Benchmark bench) {
	const auto len = bench.keys_end - bench.keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	using Table = Cuckoo<row_type, bucket_size>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(bench.key_width, bench.config.p_addr_width);

	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));
	key_type *tmp;
	CUDA(cudaMallocManaged(&tmp, sizeof(*tmp) * len * 2));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		for (auto n = 0; n < len; n += len / N_STEPS) {
			table->find_or_put(bench.keys, bench.keys + n, tmp, results, false);
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

template <typename p_row_type, uint8_t p_bucket_size,
	typename s_row_type, uint8_t s_bucket_size>
BenchResult bench_iceberg(Benchmark bench) {
	const auto len = bench.keys_end - bench.keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	using Table = Iceberg<p_row_type, p_bucket_size, s_row_type, s_bucket_size>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	new (table) Table(bench.key_width, bench.config.p_addr_width, bench.config.s_addr_width);

	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		for (auto n = 0; n < len; n += len / N_STEPS) {
			table->find_or_put(bench.keys, bench.keys + n, results, false);
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

BenchResult bench_table(TableConfig config, uint8_t key_width, key_type *keys, key_type *keys_end) {
	assert(config.p_row_width % 32 == 0);
	assert(config.s_row_width % 32 == 0);
	if (config.type == ICEBERG) {
		assert(32 % config.p_bucket_size == 0);
		assert(32 % config.s_bucket_size == 0);
	}

	struct Table {
		TableType type;
		uint8_t p_row_width; uint8_t p_bucket_size;
		uint8_t s_row_width; uint8_t s_bucket_size;
		auto operator<=>(const Table&) const = default;
	} table {
		config.type,
		config.p_row_width, config.p_bucket_size,
		config.s_row_width, config.s_bucket_size
	};

	using Runner = std::function<BenchResult(Benchmark)>;
	const std::map<Table, Runner> runners = {
		{{ CUCKOO, 32, 32}, bench_cuckoo<uint32_t, 32>},
		{{ ICEBERG, 32, 32, 32, 16 }, bench_iceberg<uint32_t, 32, uint32_t, 16>},
	};

	if (auto runner = runners.find(table); runner != runners.end()) {
		return runner->second({ config, key_width, keys, keys_end });
	}

	std::cerr << "Unsupported table configuration" << std::endl;
	std::exit(1);
}

struct Suite {
	std::string name;
	uint8_t key_width;
	size_t num_keys;
	std::string keyfile;
	std::vector<TableConfig> tables;

	using Results = std::vector<std::pair<TableConfig, BenchResult>>;
	Results run() {
		Results results;

		std::cout << "Starting benchmark suite " << name << std::endl;

		std::cerr << "\tReading " << num_keys << " keys from " << keyfile << "...";
		std::ifstream input(keyfile, std::ios::in | std::ios::binary);
		key_type *keys, *keys_end;
		CUDA(cudaMallocManaged(&keys, sizeof(*keys) * num_keys));
		keys_end = keys + num_keys;
		if (!input.read((char*)keys, num_keys * sizeof(*keys))) {
			std::cerr << "failed." << std::endl;
			std::exit(1);
		}
		thrust::all_of(thrust::device, keys, keys_end, thrust::identity<key_type>());
		std::cerr << "done." << std::endl;

		for (auto config : tables) {
			std::cout << "\t" << config.describe() << ": " << std::flush;
			auto res = bench_table(config, key_width, keys, keys_end);
			std::cout << res.average_ms << " ms" << std::endl;
			results.push_back({config, res});
		}

		std::cout << std::endl;
		CUDA(cudaFree(keys));
		return results;
	}
};

int main() {
	Suite s {
		"20 million keys of width 45",
		45, 20'000'000, "benchmarks/data/1.bin",
		{ { CUCKOO, 25, 32, 32 }
		, { ICEBERG, 24, 32, 32, 21, 32, 16 }
		}
	};
	s.run();
}
