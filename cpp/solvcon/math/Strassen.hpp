#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Implement a fixed-depth rectangular Strassen GEMM kernel.
 *
 * For C = A B, each recursion splits the operands into four blocks and
 * evaluates Strassen's seven products:
 *
 * P1 = (A11 + A22)(B11 + B22), P2 = (A21 + A22)B11
 * P3 = A11(B12 - B22), P4 = A22(B21 - B11)
 * P5 = (A11 + A12)B22, P6 = (A21 - A11)(B11 + B12)
 * P7 = (A12 - A22)(B21 + B22)
 *
 * C11 = P1 + P4 - P5 + P7, C12 = P3 + P5
 * C21 = P2 + P4, C22 = P1 - P2 + P3 + P6
 *
 * Inputs are row-major non-transposed views. The output is compact row-major
 * storage and must not overlap either input.
 *
 * Reference:
 * Volker Strassen, "Gaussian elimination is not optimal," Numerische
 * Mathematik 13(4), 354-356 (1969).
 *
 * @see https://doi.org/10.1007/BF02165411
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

namespace strassen
{

template <typename T>
struct Gemm
{
    ssize_t m_rows;
    ssize_t m_columns;
    ssize_t m_inner_size;
    BlasMatrixView<T> m_lhs;
    BlasMatrixView<T> m_rhs;
    T * m_output;
}; /* end struct Gemm */

template <typename T>
class Workspace
{
public:
    void prepare(size_t required_size);
    size_t mark() const noexcept { return m_offset; }
    T * allocate(size_t count);
    void rewind(size_t mark) noexcept { m_offset = mark; }
    size_t capacity() const noexcept { return m_capacity; }

private:
    std::unique_ptr<T[]> m_storage; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    size_t m_capacity = 0;
    size_t m_limit = 0;
    size_t m_offset = 0;
}; /* end class Workspace */

template <typename T>
void Workspace<T>::prepare(size_t required_size)
{
    if (required_size > m_capacity)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        m_storage = std::make_unique_for_overwrite<T[]>(required_size);
        m_capacity = required_size;
    }
    m_limit = required_size;
    m_offset = 0;
}

template <typename T>
T * Workspace<T>::allocate(size_t count)
{
    if (count > m_limit - m_offset)
    {
        throw std::runtime_error("Strassen workspace is too small");
    }
    T * block = m_storage.get() + m_offset;
    m_offset += count;
    return block;
}

inline size_t workspace_size(ssize_t rows, ssize_t columns, ssize_t inner_size, size_t depth)
{
    size_t required_size = 0;
    for (size_t level = 0; level < depth; ++level)
    {
        rows /= 2;
        columns /= 2;
        inner_size /= 2;
        auto const block_rows = static_cast<size_t>(rows);
        auto const block_columns = static_cast<size_t>(columns);
        auto const block_inner_size = static_cast<size_t>(inner_size);
        required_size += block_rows * block_inner_size + block_inner_size * block_columns +
                         block_rows * block_columns;
    }
    return required_size;
}

template <typename T>
void validate(Gemm<T> const & gemm, size_t depth)
{
    if (gemm.m_rows <= 0 || gemm.m_columns <= 0 || gemm.m_inner_size <= 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be positive");
    }
    if (depth >= std::numeric_limits<size_t>::digits)
    {
        throw std::invalid_argument("Strassen GEMM depth is too large");
    }
    size_t const divisor = size_t{1} << depth;
    if (gemm.m_rows % divisor != 0 || gemm.m_columns % divisor != 0 || gemm.m_inner_size % divisor != 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be divisible by 2^depth");
    }
    if (gemm.m_lhs.m_transpose != BlasTranspose::None || gemm.m_rhs.m_transpose != BlasTranspose::None)
    {
        throw std::invalid_argument("Strassen GEMM does not support transposed input views");
    }
    if (gemm.m_lhs.m_leading_dimension < gemm.m_inner_size ||
        gemm.m_rhs.m_leading_dimension < gemm.m_columns)
    {
        throw std::invalid_argument("Strassen GEMM input leading dimensions are too small");
    }
}

template <typename T>
BlasMatrixView<T> make_subview(BlasMatrixView<T> matrix, ssize_t row, ssize_t column)
{
    return {matrix.m_data + row * matrix.m_leading_dimension + column,
            matrix.m_leading_dimension,
            BlasTranspose::None};
}

template <typename T>
void combine_block(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T * output, ssize_t rows, ssize_t columns, T rhs_scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T const * lhs_row = lhs.m_data + row * lhs.m_leading_dimension;
        T const * rhs_row = rhs.m_data + row * rhs.m_leading_dimension;
        T * output_row = output + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] = lhs_row[column] + rhs_scale * rhs_row[column];
        }
    }
}

template <typename T>
void copy_block(T * output, ssize_t output_stride, T const * input, ssize_t rows, ssize_t columns)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output + row * output_stride;
        T const * input_row = input + row * columns;
        std::copy_n(input_row, columns, output_row);
    }
}

template <typename T>
void add_block(T * output, ssize_t output_stride, T const * input, ssize_t rows, ssize_t columns, T scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output + row * output_stride;
        T const * input_row = input + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] += scale * input_row[column];
        }
    }
}

template <typename T>
struct Partition
{
    ssize_t m_block_rows;
    ssize_t m_block_columns;
    ssize_t m_block_inner_size;
    ssize_t m_output_stride;
    BlasMatrixView<T> m_a11;
    BlasMatrixView<T> m_a12;
    BlasMatrixView<T> m_a21;
    BlasMatrixView<T> m_a22;
    BlasMatrixView<T> m_b11;
    BlasMatrixView<T> m_b12;
    BlasMatrixView<T> m_b21;
    BlasMatrixView<T> m_b22;
    T * m_c11;
    T * m_c12;
    T * m_c21;
    T * m_c22;
    T * m_lhs_block;
    T * m_rhs_block;
    T * m_product;
    BlasMatrixView<T> m_compact_lhs;
    BlasMatrixView<T> m_compact_rhs;
}; /* end struct Partition */

template <typename T, typename Leaf>
struct Recursion
{
    size_t m_depth;
    Workspace<T> & m_workspace;
    Leaf const & m_leaf;
}; /* end struct Recursion */

template <typename T>
Partition<T> make_partition(Gemm<T> const & gemm, Workspace<T> & workspace)
{
    ssize_t const block_rows = gemm.m_rows / 2;
    ssize_t const block_columns = gemm.m_columns / 2;
    ssize_t const block_inner_size = gemm.m_inner_size / 2;
    T * lhs_block = workspace.allocate(static_cast<size_t>(block_rows * block_inner_size));
    T * rhs_block = workspace.allocate(static_cast<size_t>(block_inner_size * block_columns));
    T * product = workspace.allocate(static_cast<size_t>(block_rows * block_columns));
    T * c21 = gemm.m_output + block_rows * gemm.m_columns;

    return {
        .m_block_rows = block_rows,
        .m_block_columns = block_columns,
        .m_block_inner_size = block_inner_size,
        .m_output_stride = gemm.m_columns,
        .m_a11 = make_subview(gemm.m_lhs, 0, 0),
        .m_a12 = make_subview(gemm.m_lhs, 0, block_inner_size),
        .m_a21 = make_subview(gemm.m_lhs, block_rows, 0),
        .m_a22 = make_subview(gemm.m_lhs, block_rows, block_inner_size),
        .m_b11 = make_subview(gemm.m_rhs, 0, 0),
        .m_b12 = make_subview(gemm.m_rhs, 0, block_columns),
        .m_b21 = make_subview(gemm.m_rhs, block_inner_size, 0),
        .m_b22 = make_subview(gemm.m_rhs, block_inner_size, block_columns),
        .m_c11 = gemm.m_output,
        .m_c12 = gemm.m_output + block_columns,
        .m_c21 = c21,
        .m_c22 = c21 + block_columns,
        .m_lhs_block = lhs_block,
        .m_rhs_block = rhs_block,
        .m_product = product,
        .m_compact_lhs = {lhs_block, block_inner_size, BlasTranspose::None},
        .m_compact_rhs = {rhs_block, block_columns, BlasTranspose::None},
    };
}

template <typename T, typename Leaf>
void recurse(Gemm<T> const & gemm, Recursion<T, Leaf> const & recursion);

template <typename T, typename Leaf>
void multiply_product(Partition<T> const & partition,
                      BlasMatrixView<T> lhs,
                      BlasMatrixView<T> rhs,
                      Recursion<T, Leaf> const & recursion)
{
    Gemm<T> const product{
        partition.m_block_rows,
        partition.m_block_columns,
        partition.m_block_inner_size,
        lhs,
        rhs,
        partition.m_product,
    };
    Recursion<T, Leaf> const child{
        recursion.m_depth - 1,
        recursion.m_workspace,
        recursion.m_leaf,
    };
    recurse(product, child);
}

template <typename T>
void combine_lhs(Partition<T> const & partition, BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale)
{
    combine_block(
        lhs, rhs, partition.m_lhs_block, partition.m_block_rows, partition.m_block_inner_size, rhs_scale);
}

template <typename T>
void combine_rhs(Partition<T> const & partition, BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale)
{
    combine_block(
        lhs, rhs, partition.m_rhs_block, partition.m_block_inner_size, partition.m_block_columns, rhs_scale);
}

template <typename T>
void copy_product(Partition<T> const & partition, T * output)
{
    copy_block(
        output, partition.m_output_stride, partition.m_product, partition.m_block_rows, partition.m_block_columns);
}

template <typename T>
void add_product(Partition<T> const & partition, T * output, T scale)
{
    add_block(output,
              partition.m_output_stride,
              partition.m_product,
              partition.m_block_rows,
              partition.m_block_columns,
              scale);
}

template <typename T, typename Leaf>
void evaluate_products(Partition<T> const & partition, Recursion<T, Leaf> const & recursion)
{
    // P1
    combine_lhs(partition, partition.m_a11, partition.m_a22, T{1});
    combine_rhs(partition, partition.m_b11, partition.m_b22, T{1});
    multiply_product(partition, partition.m_compact_lhs, partition.m_compact_rhs, recursion);
    copy_product(partition, partition.m_c11);
    copy_product(partition, partition.m_c22);

    // P2
    combine_lhs(partition, partition.m_a21, partition.m_a22, T{1});
    multiply_product(partition, partition.m_compact_lhs, partition.m_b11, recursion);
    copy_product(partition, partition.m_c21);
    add_product(partition, partition.m_c22, T{-1});

    // P3
    combine_rhs(partition, partition.m_b12, partition.m_b22, T{-1});
    multiply_product(partition, partition.m_a11, partition.m_compact_rhs, recursion);
    copy_product(partition, partition.m_c12);
    add_product(partition, partition.m_c22, T{1});

    // P4
    combine_rhs(partition, partition.m_b21, partition.m_b11, T{-1});
    multiply_product(partition, partition.m_a22, partition.m_compact_rhs, recursion);
    add_product(partition, partition.m_c11, T{1});
    add_product(partition, partition.m_c21, T{1});

    // P5
    combine_lhs(partition, partition.m_a11, partition.m_a12, T{1});
    multiply_product(partition, partition.m_compact_lhs, partition.m_b22, recursion);
    add_product(partition, partition.m_c11, T{-1});
    add_product(partition, partition.m_c12, T{1});

    // P6
    combine_lhs(partition, partition.m_a21, partition.m_a11, T{-1});
    combine_rhs(partition, partition.m_b11, partition.m_b12, T{1});
    multiply_product(partition, partition.m_compact_lhs, partition.m_compact_rhs, recursion);
    add_product(partition, partition.m_c22, T{1});

    // P7
    combine_lhs(partition, partition.m_a12, partition.m_a22, T{-1});
    combine_rhs(partition, partition.m_b21, partition.m_b22, T{1});
    multiply_product(partition, partition.m_compact_lhs, partition.m_compact_rhs, recursion);
    add_product(partition, partition.m_c11, T{1});
}

template <typename T, typename Leaf>
void recurse(Gemm<T> const & gemm, Recursion<T, Leaf> const & recursion)
{
    if (recursion.m_depth == 0)
    {
        recursion.m_leaf(gemm);
        return;
    }

    size_t const mark = recursion.m_workspace.mark();
    Partition<T> const partition = make_partition(gemm, recursion.m_workspace);
    evaluate_products(partition, recursion);
    recursion.m_workspace.rewind(mark);
}

template <typename T, typename Leaf>
void multiply(Gemm<T> const & gemm, size_t depth, Workspace<T> & workspace, Leaf const & leaf)
{
    validate(gemm, depth);
    workspace.prepare(workspace_size(gemm.m_rows, gemm.m_columns, gemm.m_inner_size, depth));
    Recursion<T, Leaf> const recursion{depth, workspace, leaf};
    recurse(gemm, recursion);
}

} /* end namespace strassen */

template <typename T>
void gemm_strassen(strassen::Gemm<T> const & gemm, size_t depth, strassen::Workspace<T> & workspace)
{
    auto const leaf = [](strassen::Gemm<T> const & leaf_gemm)
    {
        gemm_blas(
            leaf_gemm.m_rows,
            leaf_gemm.m_columns,
            leaf_gemm.m_inner_size,
            leaf_gemm.m_lhs,
            leaf_gemm.m_rhs,
            leaf_gemm.m_output);
    };
    strassen::multiply(gemm, depth, workspace, leaf);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
