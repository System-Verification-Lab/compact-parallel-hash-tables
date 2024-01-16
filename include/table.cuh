#pragma once

// Result type for find-or-put operations
enum class Result { FOUND, PUT, FULL };

// Helpers for directing kernels to device member functions
// (because __global__ functions cannot be member functions)
namespace kh {
	template <class Table>
	__global__ void find(Table *table, const key_type *start, const key_type *end, bool *results) {
		table->_find(start, end, results);
	}

	template <class Table>
	__global__ void find_or_put(Table *table, const key_type *start, const key_type *end, Result *results) {
		table->_find_or_put(start, end, results);
	}

	template <class Table>
	__global__ void put(Table *table, const key_type *start, const key_type *end, Result *results) {
		table->_put(start, end, results);
	}
}
