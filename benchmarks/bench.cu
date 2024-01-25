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
#include <sstream>
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
		std::ostringstream out;

		switch (type) {
			case CUCKOO: // + here for casting uint8_t (char) to int
				out << "Cuckoo (aw " << +p_addr_width
					<< ", rw " << +p_row_width
					<< ", bs " << +p_bucket_size << ")";
				break;
			case ICEBERG:
				out << "Iceberg (paw " << +p_addr_width
					<< ", prw " << +p_row_width
					<< ", pbs " << +p_bucket_size
					<< ", saw " << +s_addr_width
					<< ", srw " << +s_row_width
					<< ", sbs " << +s_bucket_size << ")";
				break;
			default: assert(false);
		}
		return out.str();
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

	table->~Table();
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

	table->~Table();
	CUDA(cudaFree(table));
	CUDA(cudaFree(results));

	return { std::accumulate(times_ms + 1, times_ms + N_RUNS, 0.f) / (N_RUNS - 1) };
};

namespace runnerlink {

}

using Runner = std::function<BenchResult(Benchmark)>;
Runner get_runner(TableConfig config) {
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

	using u32 = uint32_t;
	using u64 = long long unsigned;
#define REG_CUCKOO(rw, bs)\
	{{ CUCKOO, rw, bs }, bench_cuckoo<u##rw, bs> }
#define REG_ICEBERG(prw, pbs, srw, sbs)\
	{{ ICEBERG, prw, pbs, srw, sbs }, bench_iceberg<u##prw, pbs, u##srw, sbs> }
	const std::map<Table, Runner> runners = {
		REG_CUCKOO(32, 32),
		REG_CUCKOO(32, 16),
		REG_CUCKOO(64, 16),
		REG_CUCKOO(64, 8),

		REG_ICEBERG(32, 32, 32, 16),
		REG_ICEBERG(32, 32, 32, 8),
		REG_ICEBERG(32, 16, 32, 8),
		REG_ICEBERG(64, 32, 64, 16),
		REG_ICEBERG(64, 16, 64, 8),
		REG_ICEBERG(64, 8, 64, 4),
	};
#undef REG_CUCKOO
#undef REG_ICEBERG

	if (auto runner = runners.find(table); runner != runners.end()) {
		//return runner->second({ config, key_width, keys, keys_end });
		return runner->second;
	}

	std::cerr << "Unsupported table configuration: "
		<< config.describe() << std::endl;
	std::exit(1);
}

struct Suite {
	std::string name;
	uint8_t key_width;
	size_t num_keys;
	std::string keyfile;
	std::vector<TableConfig> tables;

	using Results = std::vector<std::pair<TableConfig, BenchResult>>;

	// Assert all table configurations are sound
	void assert_sound() {
		for (auto config : tables) get_runner(config);
	}

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
			auto res = get_runner(config)({ config, key_width, keys, keys_end });
			std::cout << res.average_ms << " ms" << std::endl;
			results.push_back({config, res});
		}

		std::cout << std::endl;
		CUDA(cudaFree(keys));
		return results;
	}
};

// TODO: need some way to check that all table kinds are registered,
// don't want to find out mid-run
int main() {
	const Suite suites[] {
		{ "20 million keys of width 45",
			45, 20'000'000, "benchmarks/data/1.bin",
			{
				//{ CUCKOO, 26, 64, 16 }
				// type, (addr_width, row_width, bucket_size)...
				{ CUCKOO, 25, 64, 16 }, // non-compact
				{ CUCKOO, 26, 64, 8 }, // non-compact, smaller buckets
				{ CUCKOO, 25, 32, 16 }, // compact, save space
				{ CUCKOO, 25, 32, 32 }, // compact, larger buckets
				{ CUCKOO, 26, 32, 16 }, // compact, more buckets

				{ ICEBERG, 24, 64, 16, 21, 64, 8 }, // non-compact
				{ ICEBERG, 25, 64, 8, 22, 64, 4 }, // non-compact, smaller buckets
				{ ICEBERG, 24, 32, 16, 21, 32, 8 }, // compact, save space
				{ ICEBERG, 24, 32, 32, 21, 32, 16 }, // compact, larger buckets
				{ ICEBERG, 24, 32, 32, 21, 32, 8 }, // compact, only larger p buck
				{ ICEBERG, 25, 32, 16, 22, 32, 8 }, // compact, more buckets
			}
		},
		{ "20 million keys of width 45, many duplicates",
			45, 20'000'000, "benchmarks/data/dups.bin",
			{
				//{ CUCKOO, 26, 64, 16 }
				{ CUCKOO, 25, 32, 32 },
				{ ICEBERG, 25, 64, 16, 22, 64, 8 },
				{ ICEBERG, 25, 32, 32, 21, 32, 16 },
			}
		},
	};
	for (auto s : suites) s.assert_sound();
	for (auto s : suites) s.run();
}
