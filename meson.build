# NOTE!!! Currently does not rebuild if headers have changed.
# The next version of meson should support this.
# See https://github.com/mesonbuild/meson/issues/11969

project(
	'compact-parallel-hash-tables',
	'cpp', 'cuda',
	default_options: [
		'cpp_std=c++20',
		'werror=true',
	],
)

add_project_arguments('--std=c++20', language: 'cuda')
add_project_arguments('--expt-relaxed-constexpr', language: 'cuda')
add_project_arguments('--extended-lambda', language: 'cuda')

subproject('doctest')
doctest = dependency('doctest')

cpht_inc = include_directories('include')

tests = executable('tests',
	'test/test.cu',
	include_directories: cpht_inc,
	dependencies: [doctest])
test('tests', tests)

bench = executable('bench',
	'benchmarks/bench.cu',
	include_directories: cpht_inc,
	dependencies: [doctest, argparse])
benchmark('bench', bench)
