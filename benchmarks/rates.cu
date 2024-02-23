#include "benchmarks.h"
#include "cuda_util.cuh"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
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
const auto s_log_rows = 23;
const size_t n_rows = (1ull << p_log_rows) + (1ull << s_log_rows);
const uint8_t key_width = 45;
const auto fill_ratios = { 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95 };
const auto positive_query_ratios = { 0., 0.5, 0.75, 1.};
const uint8_t row_widths[] = { 32, 64 };

int main(int argc, char** argv) {
	assert(argc == 2);
	const auto filename = argv[1];
	std::ifstream input(filename, std::ios::in | std::ios::binary);
	assert(input);

	std::cerr << "Reading " << n_rows * 2 << " keys of width "
		<< +key_width << " from " << filename << "..." << std::flush;
	auto _keys = cusp(alloc_man<key_type>(n_rows * 2));
	auto *keys = _keys.get();
	assert(input.read((char*)keys, 2 * n_rows * sizeof(*keys)));
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
					rw, uint8_t(1 << s_log_bs )},
				TableConfig { key_width, p_addr_width, s_addr_width }
			);
			tables.emplace_back(
				TableSpec { TableType::CUCKOO,
					rw, uint8_t(1 << p_log_bs) },
				TableConfig { key_width, p_addr_width }
			);
		}
	}

	printf("# Benchmark with %zu rows (2^%d primary, 2^%d secondary)\n",
		n_rows, +p_log_rows, +s_log_rows);
	printf("# Keys (expected to be unique) taken from %s\n", filename);
	// No spaces in header
	// (some software reads this as column names starting with space)
	std::cout << "table,operation,positive_ratio,rw,pbs,sbs";
	for (auto r : fill_ratios) printf(",%g", r);
	printf("\n");
	for (auto [spec, conf] : tables) {
		auto runners = get_runners(spec);

		auto print_table = [&]() {
			printf("%2d, %2d, %2d", spec.p_row_width,
				spec.p_bucket_size, spec.s_bucket_size);
		};
		const auto type_str =
			spec.type == TableType::ICEBERG ? "iceberg" : "cuckoo";

		// Find
		// Insert n_rows * fill_ratio in the table,
		// then query n_rows * 0.5 keys with a positive ratio of pr
		// (keeping the number of queried keys constant allows for
		// comparing fill ratios)
		for (auto pr : positive_query_ratios) {
			printf("%s,find,%4g,", type_str, pr); print_table();
			for (auto r : fill_ratios) {
				assert(r >= .5);
				const size_t n_query = 0.5 * n_rows;
				const size_t n_positive = n_query * pr;
				const size_t n_insert = n_rows * r;
				const auto *before_start = keys + n_rows - n_insert;
				const auto *before_end = before_start + n_insert;
				const auto *query_start = before_end - n_positive;
				auto findres = runners.find(conf, FindBenchmark {
					before_start, before_end,
					query_start, query_start + n_query
				});
				printf(", %f", findres.average_ms.value_or(NAN));
			}
			printf("\n");
		}

		// Put
		// Put fill_ratio * n_rows keys in the table
		printf("%s,put,,", type_str); print_table();
		for (auto r : fill_ratios) {
			const size_t n_insert = n_rows * r;
			auto res = runners.put(conf, PutBenchmark {
				keys, keys + n_insert
			});
			printf(", %f", res.average_ms.value_or(NAN));
		}
		printf("\n");

		// TODO: check this
		// Let's try to keep the number of inputs constant: n_rows
		// (this is realistic I think)
		// First fill table up to before_ratio
		// Then query n_rows many keys,
		// so that at the end of the fop, fill_ratio keys are in the table
		// (fop queries are evenly over already inserted keys and to insert keys)
		for (auto br : positive_query_ratios) {
			printf("%s,fop,%4g,", type_str, br); print_table(); // TODO
			const size_t n_before = n_rows * br;
			const auto *before_start = keys + n_rows - n_before;
			const auto *before_end = keys + n_rows;
			for (auto fr : fill_ratios) {
				const size_t n_new = std::max(0., fr - br) * n_rows;
				auto _to_fop = cusp(alloc_dev<key_type>(n_rows));
				key_type *tof = _to_fop.get();
				key_type *tof_end = tof + n_rows;
				thrust::counting_iterator tofi(0);
				thrust::for_each_n(thrust::device, tofi, n_rows,
					[tof, before_start, n_before, n_new]
					__device__ (auto i) {
						tof[i] = before_start[i % (n_before + n_new)];
					});

				auto res = runners.one_fop(conf, OneFopBenchmark {
					before_start, before_end,
					tof, tof_end

				});
				printf(", %f", res.average_ms.value_or(NAN));
			}
			printf("\n");
		}
	}
}
