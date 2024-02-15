#include "benchmarks.h"
#include "cuda_util.cuh"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// NOTE: in BGHT (Awad et al.), ratios are varied by varying the table size.
// Because we currently only support table sizes in powers of 2, we vary the
// number of keys instead.
// TODO: consider making the table size more flexible and follow BGHT
//
// The arithmetic on keys to be inserted based on floating point ratios might
// be off by one or two keys at times, but this does not matter when we are
// speaking of tens of millions of keys.

const auto p_log_rows = 26;
const auto s_log_rows = 20;
const size_t n_rows = (1ull << p_log_rows) + (1ull << s_log_rows);
const size_t n_keys = 0.9 * n_rows;
const uint8_t key_width = 45;
const auto fill_ratios = { 0.5, 0.6, 0.7, 0.8, 0.9 };
const auto positive_query_ratios = { 0., 0.5, 0.75, 1.};
const uint8_t row_widths[] = { 32, 64 };

int main(int argc, char** argv) {
/*	assert(argc == 4);
	const auto n_keys = std::stoull(argv[1]);
	const auto n_keys = std::stoull(argv[2]);
	const auto filename = argv[3];*/

	assert(argc == 2);
	const auto filename = argv[1];
	std::ifstream input(filename, std::ios::in | std::ios::binary);
	assert(input);

	std::cerr << "Reading " << n_keys * 2 << " keys of width "
		<< +key_width << "from " << filename << "..." << std::flush;
	auto _keys = cusp(alloc_man<key_type>(n_keys * 2));
	auto *keys = _keys.get();
	assert(input.read((char*)keys, n_keys * sizeof(*keys)));
	std::cerr << "done." << std::endl;

	using Table = std::pair<TableSpec, TableConfig>;
	std::vector<Table> tables;
	for (uint8_t p_log_bs = 3; p_log_bs < 6; p_log_bs++) {
		const uint8_t s_log_bs = p_log_bs - 1;
		const uint8_t p_addr_width = p_log_rows - p_log_bs;
		const uint8_t s_addr_width = s_log_rows - s_log_bs; 
		for (auto rw : row_widths) {
			tables.emplace_back(
				TableSpec { TableType::ICEBERG, rw,
					uint8_t(1 << p_log_bs),
					rw,
					uint8_t(1 << s_log_bs )},
				TableConfig { key_width, p_addr_width, s_addr_width }
			);
		}
	}

	std::cout << "# Iceberg find benchmark" << std::endl;
	// No spaces in header
	// (some software reads this as column names starting with space)
	std::cout << "positive_ratio,rw,pbs,sbs";
	for (auto r : fill_ratios) printf(",%g", r);
	printf("\n");
	for (auto [spec, conf] : tables) {
		auto runners = get_runners(spec);

		for (auto pr : positive_query_ratios) {
			printf("%4g, %2d, %2d, %2d", pr, spec.p_row_width,
				spec.p_bucket_size, spec.s_bucket_size);
			for (auto r : fill_ratios) {
				const size_t n_insert = n_keys * r;
				const size_t start = n_keys - n_insert;
				const size_t query_start = n_keys - (n_insert * pr);
				auto findres = runners.find(conf, FindBenchmark {
					keys + start, keys + n_keys,
					keys + query_start, keys + query_start + n_insert
				});
				printf(", %f", findres.time_ms.value_or(NAN));
			}
			printf("\n");
		}
	}
}
