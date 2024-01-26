#include <argparse/argparse.hpp>
#include <cuda_util.cuh>
#include <cuckoo.cuh>
#include <iceberg.cuh>
#define JSON_HAS_RANGES 0
#include <nlohmann/json.hpp>
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

using json = nlohmann::json;

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
NLOHMANN_JSON_SERIALIZE_ENUM(TableType, {
	{ TableType::CUCKOO, "CUCKOO" },
	{ TableType::ICEBERG, "ICEBERG" },
})

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
	uint8_t s_addr_width = 0;
	uint8_t s_row_width = 0;
	uint8_t s_bucket_size = 0;

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

// We read TableConfig as
// { "type": "CUCKOO", "arw": [aw, rw, bs] }
// { "type": "ICEBERG", "arw": [[paw, prw, pbs], [saw, srw, sbs]] }
void from_json(const json& j, TableConfig& c) {
	using ARB = std::tuple<uint8_t, uint8_t, uint8_t>;
	ARB arb;
	std::pair<ARB, ARB> arbs;

	j.at("type").get_to(c.type);
	switch (c.type) {
	case CUCKOO:
		j.at("arb").get_to(arb);
		std::tie(c.p_addr_width, c.p_row_width, c.p_bucket_size) = arb;
		break;
	case ICEBERG:
		j.at("arb").get_to(arbs);
		std::tie(
			c.p_addr_width, c.p_row_width, c.p_bucket_size,
			c.s_addr_width, c.s_row_width, c.s_bucket_size
		) = std::tuple_cat(arbs.first, arbs.second);
		break;
	}
}

// Not implemented, but necessary for framework
void to_json(json& j, const TableConfig &c) {
	assert(false);
}

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
	using u64 = unsigned long long;
#define REG_CUCKOO(rw, bs)\
	{{ CUCKOO, rw, bs }, bench_cuckoo<u##rw, bs> }
#define REG_ICEBERG(prw, pbs, srw, sbs)\
	{{ ICEBERG, prw, pbs, srw, sbs }, bench_iceberg<u##prw, pbs, u##srw, sbs> }
	const std::map<Table, Runner> runners = {
		REG_CUCKOO(32, 32),
		REG_CUCKOO(32, 16),
		REG_CUCKOO(32,  8),
		REG_CUCKOO(32,  4),
		REG_CUCKOO(32,  2),
		REG_CUCKOO(32,  1),

		REG_CUCKOO(64, 32),
		REG_CUCKOO(64, 16),
		REG_CUCKOO(64,  8),
		REG_CUCKOO(64,  4),
		REG_CUCKOO(64,  2),
		REG_CUCKOO(64,  1),

		REG_ICEBERG(32, 32, 32, 16),
		REG_ICEBERG(32, 16, 32,  8),
		REG_ICEBERG(32,  8, 32,  4),
		REG_ICEBERG(32,  4, 32,  2),
		REG_ICEBERG(32,  2, 32,  1),

		REG_ICEBERG(64, 32, 64, 16),
		REG_ICEBERG(64, 16, 64,  8),
		REG_ICEBERG(64,  8, 64,  4),
		REG_ICEBERG(64,  4, 64,  2),
		REG_ICEBERG(64,  2, 64,  1),
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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Suite, name, key_width, num_keys, keyfile, tables)

// TODO: need some way to check that all table kinds are registered,
// don't want to find out mid-run
int main(int argc, char **argv) {
	using argparse::ArgumentParser;
	ArgumentParser program(argv[0]);
	ArgumentParser cmd_suite("suite");

	cmd_suite.add_description("Benchmark a suite");
	cmd_suite.add_argument("suite_file");
	program.add_subparser(cmd_suite);

	try {
		program.parse_args(argc, argv);
	} catch (const std::exception &err) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

	if (program.is_subcommand_used(cmd_suite)) {
		auto path = cmd_suite.get("suite_file");
		std::ifstream inp(path);
		if (!inp) {
			std::cerr << "error opening " << path << std::endl;
			return 1;
		}
		std::list<Suite> suites = json::parse(inp, nullptr, true, true);
		for (auto s : suites) s.assert_sound();
		for (auto s : suites) s.run();
		return 0;
	}

	// No subcommand, print help
	std::cout << program;
}
