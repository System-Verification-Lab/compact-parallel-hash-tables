#pragma once

// Limited reimplementation of std::bit_width from C++20
// (awaiting availability of C++20)
constexpr unsigned bit_width(unsigned x) {
    if (x == 0) return 0;
    auto w = 1;
    while (x = x >> 1) w++;
    return w;
}
