#include "benchmarks.h"
#include "cuda_util.cuh"
#include <argparse/argparse.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <vector>

// NOTE: in BGHT (Awad et al.), ratios are varied by varying the table size.
// Because we currently only support table sizes in powers of 2, we vary the
// number of keys instead.
// TODO: consider making the table size more flexible and follow BGHT
//
// The arithmetic on keys to be inserted based on floating point ratios might
// be off by one or two keys at times, but this does not matter when we are
// speaking of tens of millions of keys.
//
// We need to be slightly careful with the occupancy calculations: n_rows
// assumes that there is a backyard, but Cuckoo does not have one
//
// NOTE: there is a situation with the 16-bit rows:
// - only iceberg is supported
// - the backyard uses 32-bit rows

const auto fill_ratios = { 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95 };
const auto positive_query_ratios = { 0., 0.5, 0.75, 1.};
const uint8_t row_widths[] = { 16, 32, 64 };

// Init RNG for measurement i
auto rng_init(auto i) {
	std::mt19937 rng;
	rng.discard(i * 6); // each measurement uses 6 values
	return rng;
}

int main(int argc, char** argv) {
	// Command-line arguments
	argparse::ArgumentParser args;
	args.add_argument("keys")
		.help("binary file containing unique random keys (uint64_t[])");
	args.add_argument("--key-width", "-w")
		.help("width of keys in bits")
		.default_value(39u)
		.nargs(1)
		.scan<'u', unsigned>();
	args.add_argument("--p-log-entries", "-p")
		.help("2-log of the number of primary entries")
		.default_value(29u)
		.nargs(1)
		.scan<'u', unsigned>();
	args.add_argument("--s-log-entries", "-s")
		.help("2-log of the number of secondary entries [default: p-log-entries - 3]")
		.nargs(1)
		.scan<'u', unsigned>();
	args.add_argument("--measurements", "-n")
		.help("number of measurements per configuration")
		.default_value(5u)
		.nargs(1)
		.scan<'u', unsigned>();
	const auto tabletypes = { "cuckoo", "iceberg" };
	const auto benchmarks = { "find", "fop", "put" };
	for (auto &t: tabletypes) for (auto &b : benchmarks) {
		args.add_argument(std::string("--skip-") + t + "-" + b)
			.flag();
	}
	args.add_argument("--verify")
		.help("verify uniqueness and length of keys")
		.flag();
	try {
		args.parse_args(argc, argv);
	} catch (std::exception &err) {
		std::cerr << err.what() << std::endl << args;
		return 1;
	}

	const auto filename = args.get("keys");
	const int p_log_rows = args.get<unsigned>("--p-log-entries");
	const int s_log_rows = [&]() -> int {
		if (auto slr = args.present<unsigned>("--s-log-entries")) return *slr;
		else return p_log_rows - 3;
	}();
	const size_t n_rows_cuckoo = (1ull << p_log_rows);
	const size_t n_rows_iceberg = (1ull << p_log_rows) + (1ull << s_log_rows);
	const uint8_t key_width = args.get<unsigned>("--key-width");
	// How many times the random values should be shuffled. Note that each
	// individual "measurement" contains many iterations, see benchmarks.cu.
	const int n_measurements = args.get<unsigned>("--measurements");
	const auto verify = args.get<bool>("--verify");
	// In the future, this could be much nicer with argparse's store_into()
	auto skip = [&args](auto table, auto bench) -> bool {
		return args.get<bool>(std::string("--skip-") + table + "-" + bench);
	};

	// Parse input
	std::ifstream input(filename, std::ios::in | std::ios::binary);
	if (!input) {
		std::cerr << "error opening " << filename << std::endl;
		std::abort();
	}
	const auto n_keys = n_rows_iceberg * 2;
	std::cerr << "Reading " << n_keys << " keys of width "
		<< +key_width << " from " << filename << "..." << std::flush;
	auto _keys = cusp(alloc_man<key_type>(n_keys));
	auto *keys = _keys.get();
	if (!input.read((char*)keys, n_keys * sizeof(*keys))) {
		std::cerr << "error!" << std::endl;
		std::abort();
	}
	std::cerr << "done." << std::endl;

	if (verify) {
		auto _copy = cusp(alloc_man<key_type>(n_keys));
		auto *copy = _copy.get();
		thrust::copy(thrust::device, keys, keys + n_keys, copy);
		thrust::sort(thrust::device, copy, copy + n_keys);
		auto uend = thrust::unique(thrust::device, copy, copy + n_keys);
		assert(uend = copy + n_keys);
		assert(thrust::all_of(thrust::device, keys, keys + n_keys,
			[keys, key_width] __device__ (auto key) {
				return key < (key_type(1) << key_width);
			}));
	}

	// Generate table specifications to test
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
					uint8_t(rw == 16 ? 32 : rw),
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
	for (const auto &[s, c] : tables) if (!spec_fits_config(s, c)) {
		std::cerr << "table specification does not fit config" << std::endl;
		std::abort();
	}

	printf("# Benchmark with %zu rows (2^%d primary, 2^%d secondary)\n",
		n_rows_iceberg, +p_log_rows, +s_log_rows);
	printf("# Keys (expected to be unique) taken from %s\n",
		filename.c_str());
	// No spaces in header
	// (some software reads this as column names starting with space)
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

		// Find
		// Insert n_rows * fill_ratio in the table,
		// then query n_rows * 0.5 keys with a positive ratio of pr
		// (keeping the number of queried keys constant allows for
		// comparing fill ratios)
		if (!skip(type_str, "find"))
		for (auto i = 0; i < n_measurements; i++) {
			conf.rng = rng_init(i);
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
		}

		// Put
		// Put fill_ratio * n_rows keys in the table
		if (!skip(type_str, "put"))
		for (auto i = 0; i < n_measurements; i++) {
			conf.rng = rng_init(i);

			printf("%s,put,,", type_str); print_table();
			for (auto r : fill_ratios) {
				const size_t n_insert = n_rows * r;
				auto res = runners.put(conf, PutBenchmark {
					keys, keys + n_insert
				});
				printf(", %f", res.average_ms.value_or(NAN));
			}
			printf("\n");
		}

		// Find-or-put
		//
		// TODO: check this
		// Let's try to keep the number of inputs constant: n_rows
		// (this is realistic I think)
		// First fill table up to before_ratio
		// Then query n_rows many keys,
		// so that at the end of the fop, fill_ratio keys are in the table
		// (fop queries are evenly over already inserted keys and to insert keys)
		if (!skip(type_str, "fop"))
		for (auto i = 0; i < n_measurements; i++) {
			conf.rng = rng_init(i);
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
}
