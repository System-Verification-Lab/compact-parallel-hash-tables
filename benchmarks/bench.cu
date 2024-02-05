#include "cuda_util.cuh"
#include "benchmarks.h"
#include <argparse/argparse.hpp>
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
#include <optional>
#include <sstream>
#include <string>

using json = nlohmann::json;

NLOHMANN_JSON_SERIALIZE_ENUM(TableType, {
	{ TableType::CUCKOO, "CUCKOO" },
	{ TableType::ICEBERG, "ICEBERG" },
});

std::string to_string(TableType type) {
	switch (type) {
		case TableType::CUCKOO: return "Cuckoo";
		case TableType::ICEBERG: return "Iceberg";
	}
	assert(false);
}

struct TableDesc {
	TableSpec spec;
	TableConfig conf;

	std::string describe() const {
		std::ostringstream out;

		switch (spec.type) {
		case TableType::CUCKOO: // + here for casting uint8_t (char) to int
			out << "Cuckoo (aw " << +conf.p_addr_width
				<< ", rw " << +spec.p_row_width
				<< ", bs " << +spec.p_bucket_size << ")";
			break;
		case TableType::ICEBERG:
			out << "Iceberg (paw " << +conf.p_addr_width
				<< ", prw " << +spec.p_row_width
				<< ", pbs " << +spec.p_bucket_size
				<< ", saw " << +conf.s_addr_width
				<< ", srw " << +spec.s_row_width
				<< ", sbs " << +spec.s_bucket_size << ")";
			break;
		}
		return out.str();
	}
};

// We read TableConfig as
// { "type": "CUCKOO", "arw": [aw, rw, bs] }
// { "type": "ICEBERG", "arw": [[paw, prw, pbs], [saw, srw, sbs]] }
void from_json(const json& j, TableDesc& d) {
	using ARB = std::tuple<uint8_t, uint8_t, uint8_t>;
	TableConfig &c(d.conf);
	TableSpec &s(d.spec);

	ARB arb;
	std::pair<ARB, ARB> arbs;

	j.at("type").get_to(d.spec.type);
	switch (d.spec.type) {
	case TableType::CUCKOO:
		j.at("arb").get_to(arb);
		std::tie(c.p_addr_width, s.p_row_width, s.p_bucket_size) = arb;
		break;
	case TableType::ICEBERG:
		j.at("arb").get_to(arbs);
		std::tie(
			c.p_addr_width, s.p_row_width, s.p_bucket_size,
			c.s_addr_width, s.s_row_width, s.s_bucket_size
		) = std::tuple_cat(arbs.first, arbs.second);
		break;
	}
}

// Not implemented, but necessary for framework
void to_json(json& j, const TableDesc &c) {
	assert(false);
}

struct Suite {
	std::string name;
	uint8_t key_width;
	size_t num_keys;
	std::string keyfile;
	std::vector<TableDesc> tables;

	using Results = std::vector<std::pair<TableDesc, FopResult>>;

	// Assert all table configurations are sound
	void assert_sound() {
		for (auto desc : tables) {
			if (!has_runners(desc.spec)) {
				std::cerr << "Unsupported table type: "
					<< desc.describe() << std::endl;
				std::exit(1);
			}
		}
	}

	Results run() {
		// Inform descriptions about key_width
		for (auto &desc : tables) desc.conf.key_width = key_width;

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

		for (auto desc : tables) {
			std::cout << "\t" << desc.describe() << ": " << std::flush;
			auto runners = get_runners(desc.spec);
			auto res = runners.fop(desc.conf, FopBenchmark { keys, keys_end });
			if (res.average_ms) {
				std::cout << res.average_ms.value() << " ms" << std::endl;
			} else {
				std::cout << "FULL" << std::endl;
			}
			results.push_back({desc, res});
		}

		std::cout << std::endl;
		CUDA(cudaFree(keys));
		return results;
	}
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Suite, name, key_width, num_keys, keyfile, tables)

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
