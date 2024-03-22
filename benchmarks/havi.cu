#include "benchmarks.h"
#include "cuda_util.cuh"
#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

// This is benchmark code for the havi data.
//
// This benchmark is based on the code of rates.cu, but there are differences.
// Here, we measure runtime against processed ratio of input
// (so _not_ against fill factor), because of the duplicates.
// There are more duplicates here, this is a much smaller benchmark,
// and the table does not get _that_ full. That's real-world data for you.
//
// To use this for different real-world data sets, vary the n_keys variable.

const uint8_t key_width = 24;
const auto n_keys = 71053459;

const auto p_log_rows = 24;
const auto s_log_rows = p_log_rows - 3;
const size_t n_rows_cuckoo = (1ull << p_log_rows);
const size_t n_rows_iceberg = (1ull << p_log_rows) + (1ull << s_log_rows);
const auto fill_ratios = { 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0 };
const uint8_t row_widths[] = { 16, 32, 64 };
// How many times the random values should be shuffled. Note that each
// individual "measurement" contains many iterations, see benchmarks.cu.
const auto n_measurements = 5;

// Init RNG for measurement i
auto rng_init(auto i) {
	std::mt19937 rng;
	rng.discard(i * 6); // each measurement uses 6 values
	return rng;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "supply filename of havi data (binary)" << std::endl;
		std::abort();
	}
	const auto filename = argv[1];

	std::ifstream input(filename, std::ios::in | std::ios::binary);
	if (!input) {
		std::cerr << "error opening " << filename << std::endl;
		std::abort();
	}

	std::cerr << "Reading " << n_keys << " keys of width "
		<< +key_width << " from " << filename << "..." << std::flush;
	auto _keys = cusp(alloc_man<key_type>(n_keys));
	auto *keys = _keys.get();
	if (!input.read((char*)keys, n_keys * sizeof(*keys))) {
		std::cerr << "error!" << std::endl;
		std::abort();
	}
	std::cerr << "done." << std::endl;


	using Table = std::pair<TableSpec, TableConfig>;
	std::vector<Table> tables;
	for (uint8_t p_log_bs = 3; p_log_bs < 6; p_log_bs++) {
		const uint8_t s_log_bs = p_log_bs - 1;
		const uint8_t p_addr_width = p_log_rows - p_log_bs;
		const uint8_t s_addr_width = s_log_rows - s_log_bs; 
		for (auto rw : row_widths) {
			tables.emplace_back(
				TableSpec { TableType::ICEBERG,
					rw, uint8_t(1 << p_log_bs),
					uint8_t(rw),
					uint8_t(1 << s_log_bs )},
				TableConfig { key_width, p_addr_width, s_addr_width }
			);
			// Cuckoo does not support small rows
			if (rw < 32) continue;
			tables.emplace_back(
				TableSpec { TableType::CUCKOO,
					rw, uint8_t(1 << p_log_bs) },
				TableConfig { key_width, p_addr_width }
			);
		}
	}
	for (const auto [s, c] : tables) assert(spec_fits_config(s, c));

	std::cout << "table,operation,positive_ratio,rw,pbs,sbs";
	for (auto r : fill_ratios) printf(",%g", r);
	printf("\n");
	for (auto [spec, conf] : tables) {
		const auto n_rows =
			spec.type == TableType::ICEBERG ? n_rows_iceberg : n_rows_cuckoo;
		auto runners = get_runners(spec);
		auto print_table = [&]() {
			printf("%2d, %2d, %2d", spec.p_row_width,
				spec.p_bucket_size, spec.s_bucket_size);
		};
		const auto type_str =
			spec.type == TableType::ICEBERG ? "iceberg" : "cuckoo";

		for (auto i = 0; i < n_measurements; i++) {
			conf.rng = rng_init(i);

			printf("%s,havi,,", type_str); print_table();
			for (auto fr : fill_ratios) {
				auto res = runners.one_fop(conf, OneFopBenchmark {
					keys, keys + 1,
					keys, keys + size_t(fr * n_keys)

				});
				printf(", %f", res.average_ms.value_or(NAN));

			}
			printf("\n");
		}
	}
}
