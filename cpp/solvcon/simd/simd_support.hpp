#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>

namespace solvcon
{

namespace simd
{

namespace detail
{

enum SimdFeature : std::uint8_t
{
    SIMD_NONE = 0,
    SIMD_NEON,
    SIMD_SSE,
    SIMD_SSE2,
    SIMD_SSE3,
    SIMD_SSSE3,
    SIMD_SSE41,
    SIMD_SSE42,
    SIMD_AVX,
    SIMD_AVX2,
    SIMD_AVX512,
    SIMD_UNKNOWN
};

#ifdef __aarch64__
inline constexpr SimdFeature detect_simd() noexcept
{
    return SIMD_NEON;
}
#else
SimdFeature detect_simd();
#endif

} /* end namespace detail */

} /* end namespace simd */

} /* end namespace solvcon */
