#pragma once

template <typename T>
inline constexpr T mask(const uint8_t width) {
	return (static_cast<T>(1) << width) - 1;
}
static_assert(mask<int>(3) == 0b111);
