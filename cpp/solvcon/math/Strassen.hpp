#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Internal Strassen GEMM kernel and reusable workspace.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/math/blas_compat.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

template <typename T>
class StrassenWorkspace
{
public:
    void resize(ssize_t rows, ssize_t columns, ssize_t inner_size, size_t depth);

private:
    template <typename U, typename Leaf>
    friend void gemm_strassen_with_leaf(
        ssize_t rows,
        ssize_t columns,
        ssize_t inner_size,
        U const * lhs,
        U const * rhs,
        U * output,
        size_t depth,
        StrassenWorkspace<U> & workspace,
        Leaf const & leaf);

    std::unique_ptr<T[]> m_storage;
    size_t m_capacity = 0;
    size_t m_size = 0;
}; /* end class StrassenWorkspace */

template <typename T>
struct StrassenConstMatrixView
{
    T const * m_data;
    ssize_t m_row_stride;
}; /* end struct StrassenConstMatrixView */

template <typename T>
struct StrassenMatrixView
{
    T * m_data;
    ssize_t m_row_stride;
}; /* end struct StrassenMatrixView */

template <typename T>
class StrassenArena
{
public:
    StrassenArena(T * data, size_t size)
        : m_data(data)
        , m_size(size)
    {
    }

    size_t mark() const noexcept { return m_offset; }
    T * allocate(size_t count);
    void rewind(size_t mark) noexcept { m_offset = mark; }

private:
    T * m_data;
    size_t m_size;
    size_t m_offset = 0;
}; /* end class StrassenArena */

template <typename T>
T * StrassenArena<T>::allocate(size_t count)
{
    if (count > m_size - m_offset)
    {
        throw std::runtime_error("Strassen workspace is too small");
    }
    T * ret = m_data + m_offset;
    m_offset += count;
    return ret;
}

inline size_t strassen_workspace_size(ssize_t rows, ssize_t columns, ssize_t inner_size, size_t depth)
{
    if (depth == 0)
    {
        return 0;
    }

    size_t const half_rows = static_cast<size_t>(rows / 2);
    size_t const half_columns = static_cast<size_t>(columns / 2);
    size_t const half_inner = static_cast<size_t>(inner_size / 2);
    return half_rows * half_inner + half_inner * half_columns + half_rows * half_columns +
           strassen_workspace_size(rows / 2, columns / 2, inner_size / 2, depth - 1);
}

inline void validate_strassen_shape(ssize_t rows, ssize_t columns, ssize_t inner_size, size_t depth)
{
    if (rows <= 0 || columns <= 0 || inner_size <= 0)
    {
        throw std::invalid_argument("gemm_strassen(): dimensions must be positive");
    }
    if (depth >= std::numeric_limits<size_t>::digits)
    {
        throw std::invalid_argument("gemm_strassen(): depth is too large");
    }
    size_t const divisor = size_t{1} << depth;
    if (rows % divisor != 0 || columns % divisor != 0 || inner_size % divisor != 0)
    {
        throw std::invalid_argument("gemm_strassen(): dimensions must be divisible by 2^depth");
    }
}

template <typename T>
StrassenConstMatrixView<T> subview(StrassenConstMatrixView<T> matrix, ssize_t row, ssize_t column)
{
    return {matrix.m_data + row * matrix.m_row_stride + column, matrix.m_row_stride};
}

template <typename T>
StrassenMatrixView<T> subview(StrassenMatrixView<T> matrix, ssize_t row, ssize_t column)
{
    return {matrix.m_data + row * matrix.m_row_stride + column, matrix.m_row_stride};
}

template <typename T>
void combine(
    StrassenConstMatrixView<T> lhs,
    StrassenConstMatrixView<T> rhs,
    T * output,
    ssize_t rows,
    ssize_t columns,
    T rhs_scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T const * lhs_row = lhs.m_data + row * lhs.m_row_stride;
        T const * rhs_row = rhs.m_data + row * rhs.m_row_stride;
        T * output_row = output + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] = lhs_row[column] + rhs_scale * rhs_row[column];
        }
    }
}

template <typename T>
void assign(StrassenMatrixView<T> output, T const * input, ssize_t rows, ssize_t columns)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output.m_data + row * output.m_row_stride;
        T const * input_row = input + row * columns;
        std::copy_n(input_row, columns, output_row);
    }
}

template <typename T>
void accumulate(StrassenMatrixView<T> output, T const * input, ssize_t rows, ssize_t columns, T scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output.m_data + row * output.m_row_stride;
        T const * input_row = input + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] += scale * input_row[column];
        }
    }
}

template <typename T, typename Leaf>
void multiply_strassen(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    StrassenConstMatrixView<T> lhs,
    StrassenConstMatrixView<T> rhs,
    StrassenMatrixView<T> output,
    size_t depth,
    StrassenArena<T> & arena,
    Leaf const & leaf)
{
    if (depth == 0)
    {
        leaf(rows, columns, inner_size, lhs, rhs, output);
        return;
    }

    ssize_t const half_rows = rows / 2;
    ssize_t const half_columns = columns / 2;
    ssize_t const half_inner = inner_size / 2;
    size_t const mark = arena.mark();
    T * lhs_sum = arena.allocate(static_cast<size_t>(half_rows * half_inner));
    T * rhs_sum = arena.allocate(static_cast<size_t>(half_inner * half_columns));
    T * product = arena.allocate(static_cast<size_t>(half_rows * half_columns));

    auto const a11 = subview(lhs, 0, 0);
    auto const a12 = subview(lhs, 0, half_inner);
    auto const a21 = subview(lhs, half_rows, 0);
    auto const a22 = subview(lhs, half_rows, half_inner);
    auto const b11 = subview(rhs, 0, 0);
    auto const b12 = subview(rhs, 0, half_columns);
    auto const b21 = subview(rhs, half_inner, 0);
    auto const b22 = subview(rhs, half_inner, half_columns);
    auto const c11 = subview(output, 0, 0);
    auto const c12 = subview(output, 0, half_columns);
    auto const c21 = subview(output, half_rows, 0);
    auto const c22 = subview(output, half_rows, half_columns);
    StrassenConstMatrixView<T> const compact_lhs{lhs_sum, half_inner};
    StrassenConstMatrixView<T> const compact_rhs{rhs_sum, half_columns};
    StrassenMatrixView<T> const compact_product{product, half_columns};

    combine(a11, a22, lhs_sum, half_rows, half_inner, T{1});
    combine(b11, b22, rhs_sum, half_inner, half_columns, T{1});
    multiply_strassen(
        half_rows,
        half_columns,
        half_inner,
        compact_lhs,
        compact_rhs,
        compact_product,
        depth - 1,
        arena,
        leaf);
    assign(c11, product, half_rows, half_columns);
    assign(c22, product, half_rows, half_columns);

    combine(a21, a22, lhs_sum, half_rows, half_inner, T{1});
    multiply_strassen(
        half_rows, half_columns, half_inner, compact_lhs, b11, compact_product, depth - 1, arena, leaf);
    assign(c21, product, half_rows, half_columns);
    accumulate(c22, product, half_rows, half_columns, T{-1});

    combine(b12, b22, rhs_sum, half_inner, half_columns, T{-1});
    multiply_strassen(
        half_rows, half_columns, half_inner, a11, compact_rhs, compact_product, depth - 1, arena, leaf);
    assign(c12, product, half_rows, half_columns);
    accumulate(c22, product, half_rows, half_columns, T{1});

    combine(b21, b11, rhs_sum, half_inner, half_columns, T{-1});
    multiply_strassen(
        half_rows, half_columns, half_inner, a22, compact_rhs, compact_product, depth - 1, arena, leaf);
    accumulate(c11, product, half_rows, half_columns, T{1});
    accumulate(c21, product, half_rows, half_columns, T{1});

    combine(a11, a12, lhs_sum, half_rows, half_inner, T{1});
    multiply_strassen(
        half_rows, half_columns, half_inner, compact_lhs, b22, compact_product, depth - 1, arena, leaf);
    accumulate(c11, product, half_rows, half_columns, T{-1});
    accumulate(c12, product, half_rows, half_columns, T{1});

    combine(a21, a11, lhs_sum, half_rows, half_inner, T{-1});
    combine(b11, b12, rhs_sum, half_inner, half_columns, T{1});
    multiply_strassen(
        half_rows,
        half_columns,
        half_inner,
        compact_lhs,
        compact_rhs,
        compact_product,
        depth - 1,
        arena,
        leaf);
    accumulate(c22, product, half_rows, half_columns, T{1});

    combine(a12, a22, lhs_sum, half_rows, half_inner, T{-1});
    combine(b21, b22, rhs_sum, half_inner, half_columns, T{1});
    multiply_strassen(
        half_rows,
        half_columns,
        half_inner,
        compact_lhs,
        compact_rhs,
        compact_product,
        depth - 1,
        arena,
        leaf);
    accumulate(c11, product, half_rows, half_columns, T{1});

    arena.rewind(mark);
}

template <typename T>
void StrassenWorkspace<T>::resize(ssize_t rows, ssize_t columns, ssize_t inner_size, size_t depth)
{
    validate_strassen_shape(rows, columns, inner_size, depth);
    m_size = strassen_workspace_size(rows, columns, inner_size, depth);
    if (m_size > m_capacity)
    {
        m_storage = std::make_unique_for_overwrite<T[]>(m_size);
        m_capacity = m_size;
    }
}

template <typename T, typename Leaf>
void gemm_strassen_with_leaf(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    T const * lhs,
    T const * rhs,
    T * output,
    size_t depth,
    StrassenWorkspace<T> & workspace,
    Leaf const & leaf)
{
    workspace.resize(rows, columns, inner_size, depth);
    StrassenArena<T> arena(workspace.m_storage.get(), workspace.m_size);
    StrassenConstMatrixView<T> const lhs_view{lhs, inner_size};
    StrassenConstMatrixView<T> const rhs_view{rhs, columns};
    StrassenMatrixView<T> const output_view{output, columns};
    multiply_strassen(rows, columns, inner_size, lhs_view, rhs_view, output_view, depth, arena, leaf);
}

template <typename T>
void gemm_strassen(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    T const * lhs,
    T const * rhs,
    T * output,
    size_t depth,
    StrassenWorkspace<T> & workspace)
{
    auto const leaf = [](
                          ssize_t leaf_rows,
                          ssize_t leaf_columns,
                          ssize_t leaf_inner_size,
                          StrassenConstMatrixView<T> leaf_lhs,
                          StrassenConstMatrixView<T> leaf_rhs,
                          StrassenMatrixView<T> leaf_output)
    {
        BlasMatrixView<T> const lhs_blas{
            leaf_lhs.m_data, leaf_lhs.m_row_stride, BlasTranspose::None};
        BlasMatrixView<T> const rhs_blas{
            leaf_rhs.m_data, leaf_rhs.m_row_stride, BlasTranspose::None};
        gemm_blas(
            leaf_rows,
            leaf_columns,
            leaf_inner_size,
            lhs_blas,
            rhs_blas,
            leaf_output.m_data);
    };
    gemm_strassen_with_leaf(
        rows, columns, inner_size, lhs, rhs, output, depth, workspace, leaf);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
