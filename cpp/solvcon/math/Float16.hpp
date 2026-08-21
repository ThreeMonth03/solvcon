#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Portable IEEE 754 binary16 storage type.
 *
 * @ingroup group_core
 */

#include <half.hpp>

#include <type_traits>

namespace solvcon
{

using Float16 = half_float::half;

static_assert(sizeof(Float16) == 2);
static_assert(std::is_trivially_copyable_v<Float16>);

template <typename T>
inline constexpr bool is_float16_v = std::is_same_v<std::remove_cv_t<T>, Float16>;

template <typename T>
Float16 float16_cast(T const & value)
{
    return Float16(static_cast<float>(value));
}

template <typename T>
inline constexpr bool is_floating_number_v = std::is_floating_point_v<T> || is_float16_v<T>;

template <typename T>
inline constexpr bool is_arithmetic_number_v = std::is_arithmetic_v<T> || is_float16_v<T>;

template <typename T>
inline constexpr bool is_signed_number_v = std::is_signed_v<T> || is_float16_v<T>;

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
