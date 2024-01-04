# NOTE!!! Currently does not rebuild if headers have changed.
# The next version of meson should support this.
# See https://github.com/mesonbuild/meson/issues/11969

project(
	'compact-parallel-hash-tables',
	'cpp', 'cuda',
	default_options: [
		'cpp_std=c++14',
		'werror=true',
	],
)

add_project_arguments('--std=c++14', language: 'cuda')
# We only need this because std::make_pair is constexpr in C++14...
add_project_arguments('--expt-relaxed-constexpr', language: 'cuda')

subproject('doctest')
doctest = dependency('doctest')

cpht_inc = include_directories('include')
cpht_source = []
cpht = library('cpht',
	cpht_source,
	include_directories: cpht_inc,
	dependencies: [])

tests = executable('tests',
	'test/test.cu',
	include_directories: cpht_inc,
	dependencies: [doctest])

test('tests', tests)
