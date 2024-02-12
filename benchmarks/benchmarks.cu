#include "cuckoo.cuh"
#include "iceberg.cuh"
#include "benchmarks.h"
#include <thrust/find.h>
#include <map>
#include <numeric>
#include <type_traits>

// Runs per benchmark. First run is discarded.
constexpr auto N_RUNS = 10;
// Steps in fop-benchmark
constexpr auto N_STEPS = 10;

// Wrapper for CUDA GPU timers
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

//
// Runners
//

template <TableSpec spec>
requires (spec.p_row_width == 64 || spec.p_row_width == 32)
using p_row_t = std::conditional<spec.p_row_width == 64, unsigned long long, uint32_t>::type;

template <TableSpec spec>
requires (spec.s_row_width == 64 || spec.s_row_width == 32
	|| (spec.type == TableType::CUCKOO && spec.s_row_width == 0))
using s_row_t = std::conditional<spec.s_row_width == 64, unsigned long long, uint32_t>::type;

// We use here that the current _implementation_ of std::conditional short
// circuits on the false path (here: if type is CUCKOO, then we get no Iceberg
// template errors for having a s_row_width of 0).
template <TableSpec spec>
using Table = std::conditional_t<spec.type == TableType::ICEBERG,
	Iceberg<p_row_t<spec>, spec.p_bucket_size, s_row_t<spec>, spec.s_bucket_size>,
	Cuckoo<p_row_t<spec>, spec.p_bucket_size>>;

template <TableSpec spec>
static Table<spec> *new_table(TableConfig conf) {
	using Table = Table<spec>;
	Table *table;
	CUDA(cudaMallocManaged(&table, sizeof(*table)));
	if constexpr (spec.type == TableType::CUCKOO) {
		new (table) Table(conf.key_width, conf.p_addr_width);
	} else {
		new (table) Table(conf.key_width, conf.p_addr_width, conf.s_addr_width);
	}
	return table;
}

template <TableSpec s>
FindResult find_runner(TableConfig config, FindBenchmark bench) {
	assert(false);
}

template <TableSpec spec>
FopResult fop_runner(TableConfig conf, FopBenchmark bench) {
	using T = Table<spec>;
	const auto len = bench.keys_end - bench.keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	T *table = new_table<spec>(conf);
	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));
	key_type *tmp;
	CUDA(cudaMallocManaged(&tmp, sizeof(*tmp) * len * 2));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		for (auto n = 0; n < len; n += len / N_STEPS) {
			if constexpr (spec.type == TableType::CUCKOO) {
				table->find_or_put(bench.keys, bench.keys + n, tmp, results, false);
			} else {
				table->find_or_put(bench.keys, bench.keys + n, results, false);
			}
		}
		times_ms[i] = timer.stop();

		CUDA(cudaDeviceSynchronize());
		bool full = thrust::find(thrust::device,
			results, results + len, Result::FULL) != results + len;
		if (full) return FopResult { {} };
	}

	table->~T();
	CUDA(cudaFree(table));
	CUDA(cudaFree(results));
	CUDA(cudaFree(tmp));

	return FopResult {
		std::accumulate(times_ms + 1, times_ms + N_RUNS, 0.f) / (N_RUNS - 1)
	};
}

template <TableSpec spec>
PutResult put_runner(TableConfig conf, PutBenchmark bench) {
	using T = Table<spec>;
	const auto len = bench.keys_end - bench.keys;
	assert(len % N_STEPS == 0);
	float times_ms[N_RUNS];

	T *table = new_table<spec>(conf);
	Result *results;
	CUDA(cudaMallocManaged(&results, sizeof(*results) * len));

	Timer timer;
	for (auto i = 0; i < N_RUNS; i++) {
		table->clear();

		timer.start();
		table->put(bench.keys, bench.keys_end, results, false);
		times_ms[i] = timer.stop();

		CUDA(cudaDeviceSynchronize());
		bool full = thrust::find(thrust::device,
			results, results + len, Result::FULL) != results + len;
		if (full) return PutResult { {} };
	}

	table->~T();
	CUDA(cudaFree(table));
	CUDA(cudaFree(results));

	return PutResult {
		std::accumulate(times_ms + 1, times_ms + N_RUNS, 0.f) / (N_RUNS - 1)
	};
}

//
// Runners registry
//

template <TableSpec spec>
static Runners make_runners() {
	return Runners {
		find_runner<spec>,
		fop_runner<spec>,
		put_runner<spec>,
	};
}

template <TableType type, uint8_t prw, uint8_t pbs, uint8_t srw, uint8_t sbs>
static std::pair<TableSpec, Runners> registration() {
	constexpr TableSpec spec { type, prw, pbs, srw, sbs };
	return { spec, make_runners<spec>() };
}

template <uint8_t prw, uint8_t pbs>
static auto cuckoo = registration<TableType::CUCKOO, prw, pbs, 0, 0>();

template <uint8_t prw, uint8_t pbs, uint8_t srw, uint8_t sbs>
static auto iceberg = registration<TableType::ICEBERG, prw, pbs, srw, sbs>();

static const std::map<TableSpec, Runners> registry {
	cuckoo<32, 32>,
	cuckoo<32, 16>,
	cuckoo<32,  8>,
	cuckoo<32,  4>,
	cuckoo<32,  2>,
	cuckoo<32,  1>,

	cuckoo<64, 32>,
	cuckoo<64, 16>,
	cuckoo<64,  8>,
	cuckoo<64,  4>,
	cuckoo<64,  2>,
	cuckoo<64,  1>,

	iceberg<32, 32, 32, 16>,
	iceberg<32, 16, 32,  8>,
	iceberg<32, 16, 32,  4>,
	iceberg<32, 16, 32,  2>,
	iceberg<32,  8, 32,  4>,
	iceberg<32,  8, 32,  2>,
	iceberg<32,  4, 32,  2>,
	iceberg<32,  2, 32,  1>,

	iceberg<64, 32, 64, 16>,
	iceberg<64, 16, 64,  8>,
	iceberg<64,  8, 64,  4>,
	iceberg<64,  4, 64,  2>,
	iceberg<64,  2, 64,  1>,
};

Runners get_runners(TableSpec spec) {
	return registry.at(spec);
}

bool has_runners(TableSpec spec) {
	return registry.contains(spec);
}
