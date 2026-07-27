#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/loop.hpp>
#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

template <typename T>
inline constexpr bool can_matmul_blas_v = std::is_same_v<T, float> ||
                                          std::is_same_v<T, double> ||
                                          std::is_same_v<T, Complex<float>> ||
                                          std::is_same_v<T, Complex<double>>;

/**
 * @brief Describe matmul operands as an execution-independent contraction.
 *
 * MatmulPlan interprets trailing axes as vector or matrix roles and leading
 * axes as the batch domain. It validates the contracted dimension, derives
 * the result shape, and records the signed-stride mappings used to locate
 * operands in the result batch domain. Broadcast batch axes are represented
 * by zero strides, so an executor can advance operand offsets without
 * reinterpreting ranks or broadcasting rules.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the plan records batch shape `(2,5)`, M=3,
 * N=6, K=4, output shape `(2,5,3,6)`, and zero-stride batch mappings for
 * the broadcast axes. It does not allocate the output or evaluate the ten
 * matrix pairs.
 *
 * Vector operands contribute no batch axes. Their removed matrix axis is
 * represented by a unit row or column extent, so every operand-role pairing
 * uses the same contraction and batch traversal.
 */
class MatmulPlan
{
public:
    using shape_type = small_vector<ssize_t>;

    enum class BatchOperand : std::uint8_t
    {
        Output,
        Lhs,
        Rhs,
    };

    template <typename Array>
    static MatmulPlan make(Array const & lhs, Array const & rhs);

    shape_type const & output_shape() const noexcept { return m_output_shape; }
    ssize_t rows() const noexcept { return m_contraction.m_rows; }
    ssize_t columns() const noexcept { return m_contraction.m_columns; }
    ssize_t inner_size() const noexcept { return m_contraction.m_inner_size; }
    ssize_t lhs_row_stride() const noexcept { return m_strides.m_lhs_row_stride; }
    ssize_t lhs_inner_stride() const noexcept { return m_strides.m_lhs_inner_stride; }
    ssize_t rhs_inner_stride() const noexcept { return m_strides.m_rhs_inner_stride; }
    ssize_t rhs_column_stride() const noexcept { return m_strides.m_rhs_column_stride; }
    bool lhs_vector() const noexcept { return m_roles.m_lhs_vector; }
    bool rhs_vector() const noexcept { return m_roles.m_rhs_vector; }

    bool has_batch_axes() const noexcept { return m_batch.m_domain.rank() != 0; }
    size_t batch_rank() const noexcept { return m_batch.m_domain.rank(); }
    size_t batch_size() const noexcept { return m_batch.m_domain.size(); }
    ssize_t batch_extent(size_t axis) const noexcept
    {
        return m_batch.m_domain.extent(axis);
    }
    ssize_t batch_stride(BatchOperand operand, size_t axis) const noexcept
    {
        return m_batch.m_mappings[std::to_underlying(operand)].stride(axis);
    }
    MappedOffsetCursor batch_cursor() const & { return MappedOffsetCursor(m_batch.m_domain, m_batch.m_mappings); }
    MappedOffsetCursor batch_cursor() const && = delete;

private:
    using batch_stride_type = OperandMapping::stride_type;
    using mapping_type = MappedOffsetCursor::mapping_type;

    struct Contraction
    {
        ssize_t m_rows;
        ssize_t m_columns;
        ssize_t m_inner_size;
    }; /* end struct Contraction */

    struct MatrixStrides
    {
        ssize_t m_lhs_row_stride;
        ssize_t m_lhs_inner_stride;
        ssize_t m_rhs_inner_stride;
        ssize_t m_rhs_column_stride;
    }; /* end struct MatrixStrides */

    struct OperandRoles
    {
        bool m_lhs_vector;
        bool m_rhs_vector;
    }; /* end struct OperandRoles */

    struct BatchMappings
    {
        LoopDomain m_domain;
        mapping_type m_mappings;
    }; /* end struct BatchMappings */

    MatmulPlan(
        shape_type output_shape,
        Contraction contraction,
        MatrixStrides strides,
        OperandRoles roles,
        BatchMappings batch);

    template <typename Array>
    static Contraction make_contraction(Array const & lhs, Array const & rhs, OperandRoles const & roles);

    template <typename Array>
    static BatchMappings make_batch_mappings(
        Array const & lhs,
        Array const & rhs,
        OperandRoles const & roles,
        ssize_t output_matrix_size);

    template <typename Array>
    static shape_type make_batch_shape(Array const & lhs, Array const & rhs, OperandRoles const & roles);

    template <typename Array>
    static OperandMapping make_batch_mapping(Array const & operand, bool vector, LoopDomain const & domain);

    static shape_type make_output_shape(
        BatchMappings const & batch,
        Contraction const & contraction,
        OperandRoles const & roles);

    template <typename Array>
    static std::string shape_string(Array const & array);

    shape_type m_output_shape;
    Contraction m_contraction;
    MatrixStrides m_strides;
    OperandRoles m_roles;
    BatchMappings m_batch;
}; /* end class MatmulPlan */

/**
 * @brief Execute a MatmulPlan with a layout-appropriate contraction route.
 *
 * MatmulExecutor maps vector and matrix roles to DOT, GEMV, or GEMM, then
 * selects generic, tiled, direct BLAS, or pack-once BLAS execution. It owns
 * BLAS eligibility, packing reuse, and size thresholds; these decisions do
 * not change the plan. The constructor only binds a plan and caller-owned
 * arrays; execute() evaluates the plan.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the executor visits ten batch offsets and
 * evaluates one `(3,4) @ (4,6)` contraction at each offset. The results are
 * written into the allocated `(2,5,3,6)` output.
 *
 * Generic traversal remains the fallback for small contractions and types
 * without a BLAS backend. BLAS-compatible layouts are consumed directly,
 * while unsupported core layouts are packed once before batch traversal.
 */
template <typename Array>
class MatmulExecutor
{
public:
    MatmulExecutor(MatmulPlan const & plan, Array & output, Array const & lhs, Array const & rhs);

    void execute();

private:
    using value_type = typename Array::value_type;

    static constexpr size_t BLAS_MINIMUM_WORK = 4096;
    static constexpr size_t DIRECT_MATRIX_BLAS_MINIMUM_WORK = 512;
    static constexpr size_t PACKED_VECTOR_BLAS_MINIMUM_WORK = 1024;
    static constexpr size_t PACKED_VECTOR_BLAS_MINIMUM_BATCHES = 4;
    static constexpr size_t REUSED_VECTOR_BLAS_MINIMUM_WORK = 576;
    static constexpr size_t REUSED_VECTOR_BLAS_MINIMUM_INTENSITY = 128;

    class BlasMatrixLayout
    {
    public:
        BlasMatrixLayout(bool transpose, ssize_t leading_dimension)
            : m_transpose(transpose)
            , m_leading_dimension(leading_dimension)
        {
        }

        bool transpose() const noexcept { return m_transpose; }
        ssize_t leading_dimension() const noexcept { return m_leading_dimension; }

    private:
        bool m_transpose;
        ssize_t m_leading_dimension;
    }; /* end class BlasMatrixLayout */

    static std::optional<BlasMatrixLayout> lhs_blas_layout(MatmulPlan const & plan);
    static std::optional<BlasMatrixLayout> rhs_blas_layout(MatmulPlan const & plan);

    bool use_blas() const;
    bool use_small_batched_vector_blas(size_t matrix_work) const;
    void execute_generic();
    void execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);
    void execute_blas_dispatch();
    void execute_direct_blas();
    void execute_matrix_blas(
        BlasMatrixLayout const & lhs_layout,
        BlasMatrixLayout const & rhs_layout);
    void execute_matrix_blas(
        ssize_t output_base,
        ssize_t lhs_base,
        ssize_t rhs_base,
        BlasMatrixLayout const & lhs_layout,
        BlasMatrixLayout const & rhs_layout);
    void execute_batched_matrix_blas(
        BlasMatrixLayout const & lhs_layout,
        BlasMatrixLayout const & rhs_layout);
    void execute_vector_blas();
    void execute_vector_blas(
        ssize_t output_base,
        ssize_t lhs_base,
        ssize_t rhs_base,
        BlasMatrixLayout const & matrix_layout);
    bool can_execute_affine_vector_blas() const;
    void execute_affine_vector_blas(BlasMatrixLayout const & matrix_layout);
    void execute_packed_blas(bool pack_lhs, bool pack_rhs);

    MatmulPlan const & m_plan;
    Array & m_output;
    Array const & m_lhs;
    Array const & m_rhs;
    value_type * m_output_data;
}; /* end class MatmulExecutor */

inline MatmulPlan::MatmulPlan(
    shape_type output_shape,
    Contraction contraction,
    MatrixStrides strides,
    OperandRoles roles,
    BatchMappings batch)
    : m_output_shape(std::move(output_shape))
    , m_contraction(contraction)
    , m_strides(strides)
    , m_roles(roles)
    , m_batch(std::move(batch))
{
}

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() == 0 || rhs.ndim() == 0)
    {
        throw std::invalid_argument(
            "planned matmul requires non-scalar operands");
    }

    OperandRoles const roles{
        .m_lhs_vector = lhs.ndim() == 1,
        .m_rhs_vector = rhs.ndim() == 1,
    };
    Contraction const contraction = make_contraction(lhs, rhs, roles);
    MatrixStrides const strides{
        .m_lhs_row_stride = roles.m_lhs_vector ? 0 : lhs.stride(lhs.ndim() - 2),
        .m_lhs_inner_stride = lhs.stride(lhs.ndim() - 1),
        .m_rhs_inner_stride = roles.m_rhs_vector ? rhs.stride(0) : rhs.stride(rhs.ndim() - 2),
        .m_rhs_column_stride = roles.m_rhs_vector ? 0 : rhs.stride(rhs.ndim() - 1),
    };
    BatchMappings batch = make_batch_mappings(
        lhs,
        rhs,
        roles,
        contraction.m_rows * contraction.m_columns);
    shape_type output_shape = make_output_shape(batch, contraction, roles);
    return MatmulPlan{
        std::move(output_shape),
        contraction,
        strides,
        roles,
        std::move(batch),
    };
}

template <typename Array>
MatmulPlan::Contraction MatmulPlan::make_contraction(
    Array const & lhs,
    Array const & rhs,
    OperandRoles const & roles)
{
    ssize_t const inner_size = lhs.shape(lhs.ndim() - 1);
    ssize_t const rhs_inner_size = roles.m_rhs_vector
                                       ? rhs.shape(0)
                                       : rhs.shape(rhs.ndim() - 2);
    if (inner_size != rhs_inner_size)
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul_planned(): shape mismatch: "
                        "this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }
    return Contraction{
        .m_rows = roles.m_lhs_vector ? 1 : lhs.shape(lhs.ndim() - 2),
        .m_columns = roles.m_rhs_vector ? 1 : rhs.shape(rhs.ndim() - 1),
        .m_inner_size = inner_size,
    };
}

template <typename Array>
MatmulPlan::BatchMappings MatmulPlan::make_batch_mappings(
    Array const & lhs,
    Array const & rhs,
    OperandRoles const & roles,
    ssize_t output_matrix_size)
{
    LoopDomain domain{make_batch_shape(lhs, rhs, roles)};
    mapping_type mappings{
        OperandMapping::contiguous(domain, output_matrix_size),
        make_batch_mapping(lhs, roles.m_lhs_vector, domain),
        make_batch_mapping(rhs, roles.m_rhs_vector, domain),
    };
    return BatchMappings{
        .m_domain = std::move(domain),
        .m_mappings = std::move(mappings),
    };
}

template <typename Array>
MatmulPlan::shape_type MatmulPlan::make_batch_shape(
    Array const & lhs,
    Array const & rhs,
    OperandRoles const & roles)
{
    size_t const lhs_batch_rank = roles.m_lhs_vector ? 0 : lhs.ndim() - 2;
    size_t const rhs_batch_rank = roles.m_rhs_vector ? 0 : rhs.ndim() - 2;
    size_t const batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
    shape_type shape(batch_rank, 1);
    for (size_t offset = 0; offset < batch_rank; ++offset)
    {
        ssize_t const lhs_extent = offset < lhs_batch_rank ? lhs.shape(lhs_batch_rank - offset - 1) : 1;
        ssize_t const rhs_extent = offset < rhs_batch_rank ? rhs.shape(rhs_batch_rank - offset - 1) : 1;
        if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
        {
            throw std::invalid_argument(
                std::format("SimpleArray::matmul_planned(): batch shape "
                            "mismatch: this={} other={}",
                            shape_string(lhs),
                            shape_string(rhs)));
        }
        shape[batch_rank - offset - 1] = lhs_extent == 1 ? rhs_extent : lhs_extent;
    }
    return shape;
}

template <typename Array>
OperandMapping MatmulPlan::make_batch_mapping(
    Array const & operand,
    bool vector,
    LoopDomain const & domain)
{
    size_t const batch_rank = vector ? 0 : operand.ndim() - 2;
    size_t const rank_delta = domain.rank() - batch_rank;
    batch_stride_type strides(domain.rank(), 0);
    for (size_t axis = 0; axis < batch_rank; ++axis)
    {
        size_t const domain_axis = rank_delta + axis;
        if (operand.shape(axis) == domain.extent(domain_axis))
        {
            strides[domain_axis] = operand.stride(axis);
        }
    }
    return OperandMapping(std::move(strides));
}

inline MatmulPlan::shape_type MatmulPlan::make_output_shape(
    BatchMappings const & batch,
    Contraction const & contraction,
    OperandRoles const & roles)
{
    size_t const batch_rank = batch.m_domain.rank();
    size_t const result_rank = roles.m_lhs_vector || roles.m_rhs_vector ? 1 : 2;
    shape_type output_shape(batch_rank + result_rank);
    std::ranges::copy(batch.m_domain.shape(), output_shape.begin());
    if (roles.m_lhs_vector && roles.m_rhs_vector)
    {
        output_shape[batch_rank] = 1;
    }
    else if (roles.m_lhs_vector)
    {
        output_shape[batch_rank] = contraction.m_columns;
    }
    else if (roles.m_rhs_vector)
    {
        output_shape[batch_rank] = contraction.m_rows;
    }
    else
    {
        output_shape[batch_rank] = contraction.m_rows;
        output_shape[batch_rank + 1] = contraction.m_columns;
    }
    return output_shape;
}

template <typename Array>
std::string MatmulPlan::shape_string(Array const & array)
{
    std::string result = "(";
    for (size_t axis = 0; axis < array.ndim(); ++axis)
    {
        if (axis > 0)
        {
            result += ",";
        }
        result += std::to_string(array.shape(axis));
    }
    result += ")";
    return result;
}

template <typename Array>
MatmulExecutor<Array>::MatmulExecutor(MatmulPlan const & plan, Array & output, Array const & lhs, Array const & rhs)
    : m_plan(plan)
    , m_output(output)
    , m_lhs(lhs)
    , m_rhs(rhs)
    , m_output_data(output.logical_data())
{
}

template <typename Array>
std::optional<typename MatmulExecutor<Array>::BlasMatrixLayout>
MatmulExecutor<Array>::lhs_blas_layout(MatmulPlan const & plan)
{
    if (plan.lhs_inner_stride() == 1 &&
        plan.lhs_row_stride() >= plan.inner_size())
    {
        return BlasMatrixLayout(false, plan.lhs_row_stride());
    }
    if (plan.lhs_row_stride() == 1 &&
        plan.lhs_inner_stride() >= plan.rows())
    {
        return BlasMatrixLayout(true, plan.lhs_inner_stride());
    }
    return std::nullopt;
}

template <typename Array>
std::optional<typename MatmulExecutor<Array>::BlasMatrixLayout>
MatmulExecutor<Array>::rhs_blas_layout(MatmulPlan const & plan)
{
    if (plan.rhs_column_stride() == 1 &&
        plan.rhs_inner_stride() >= plan.columns())
    {
        return BlasMatrixLayout(false, plan.rhs_inner_stride());
    }
    if (plan.rhs_inner_stride() == 1 &&
        plan.rhs_column_stride() >= plan.inner_size())
    {
        return BlasMatrixLayout(true, plan.rhs_column_stride());
    }
    return std::nullopt;
}

template <typename Array>
bool MatmulExecutor<Array>::use_blas() const
{
    size_t const matrix_work =
        static_cast<size_t>(m_plan.rows()) *
        static_cast<size_t>(m_plan.columns()) *
        static_cast<size_t>(m_plan.inner_size());
    if (matrix_work >= BLAS_MINIMUM_WORK)
    {
        return true;
    }
    if (!m_plan.lhs_vector() && !m_plan.rhs_vector())
    {
        if (lhs_blas_layout(m_plan) && rhs_blas_layout(m_plan) &&
            matrix_work >= DIRECT_MATRIX_BLAS_MINIMUM_WORK)
        {
            return true;
        }
        return m_plan.has_batch_axes() &&
               matrix_work * m_plan.batch_size() >=
                   BLAS_MINIMUM_WORK;
    }
    if (m_plan.lhs_vector() && m_plan.rhs_vector())
    {
        return false;
    }
    return use_small_batched_vector_blas(matrix_work);
}

template <typename Array>
bool MatmulExecutor<Array>::use_small_batched_vector_blas(
    size_t matrix_work) const
{
    if (!m_plan.has_batch_axes())
    {
        return false;
    }
    std::optional<BlasMatrixLayout> const matrix_layout =
        m_plan.lhs_vector() ? rhs_blas_layout(m_plan)
                            : lhs_blas_layout(m_plan);
    if (!matrix_layout)
    {
        return false;
    }

    ssize_t const vector_stride =
        m_plan.lhs_vector() ? m_plan.lhs_inner_stride()
                            : m_plan.rhs_inner_stride();
    if (vector_stride > 0)
    {
        return true;
    }
    if (matrix_work >= PACKED_VECTOR_BLAS_MINIMUM_WORK &&
        m_plan.batch_size() >= PACKED_VECTOR_BLAS_MINIMUM_BATCHES)
    {
        return true;
    }
    if (matrix_work < REUSED_VECTOR_BLAS_MINIMUM_WORK)
    {
        return false;
    }

    size_t const output_extent = static_cast<size_t>(
        m_plan.lhs_vector() ? m_plan.columns() : m_plan.rows());
    if (output_extent == 0)
    {
        return false;
    }
    size_t const minimum_batches =
        REUSED_VECTOR_BLAS_MINIMUM_INTENSITY / output_extent +
        (REUSED_VECTOR_BLAS_MINIMUM_INTENSITY % output_extent != 0);
    return m_plan.batch_size() >= minimum_batches;
}

template <typename Array>
void MatmulExecutor<Array>::execute()
{
    if constexpr (can_matmul_blas_v<value_type>)
    {
        if (use_blas())
        {
            execute_blas_dispatch();
            return;
        }
    }
    execute_generic();
}

template <typename Array>
void MatmulExecutor<Array>::execute_generic()
{
    if (!m_plan.has_batch_axes())
    {
        execute_generic(0, 0, 0);
        return;
    }

    for (MappedOffsetCursor cursor = m_plan.batch_cursor(); cursor; cursor.advance())
    {
        execute_generic(
            cursor.offset(MatmulPlan::BatchOperand::Output),
            cursor.offset(MatmulPlan::BatchOperand::Lhs),
            cursor.offset(MatmulPlan::BatchOperand::Rhs));
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base)
{
    value_type const * lhs_data = m_lhs.logical_data();
    value_type const * rhs_data = m_rhs.logical_data();
    for (ssize_t row = 0; row < m_plan.rows(); ++row)
    {
        ssize_t const lhs_row_base = lhs_base + row * m_plan.lhs_row_stride();
        ssize_t const output_row_base = output_base + row * m_plan.columns();
        for (ssize_t column = 0; column < m_plan.columns(); ++column)
        {
            value_type total{};
            ssize_t lhs_offset = lhs_row_base;
            ssize_t rhs_offset = rhs_base + column * m_plan.rhs_column_stride();
            for (ssize_t inner = 0; inner < m_plan.inner_size(); ++inner)
            {
                total += lhs_data[lhs_offset] * rhs_data[rhs_offset];
                lhs_offset += m_plan.lhs_inner_stride();
                rhs_offset += m_plan.rhs_inner_stride();
            }
            m_output_data[output_row_base + column] = total;
        }
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_blas_dispatch()
{
    bool const pack_lhs =
        m_plan.lhs_vector()
            ? m_plan.lhs_inner_stride() <= 0
            : !lhs_blas_layout(m_plan);
    bool const pack_rhs =
        m_plan.rhs_vector()
            ? m_plan.rhs_inner_stride() <= 0
            : !rhs_blas_layout(m_plan);
    if (pack_lhs || pack_rhs)
    {
        execute_packed_blas(pack_lhs, pack_rhs);
        return;
    }
    execute_direct_blas();
}

template <typename Array>
void MatmulExecutor<Array>::execute_direct_blas()
{
    if (m_plan.lhs_vector() || m_plan.rhs_vector())
    {
        execute_vector_blas();
        return;
    }

    std::optional<BlasMatrixLayout> const lhs_layout =
        lhs_blas_layout(m_plan);
    std::optional<BlasMatrixLayout> const rhs_layout =
        rhs_blas_layout(m_plan);
    if (!lhs_layout || !rhs_layout)
    {
        throw std::logic_error(
            "direct planned matmul requires BLAS matrix layouts");
    }
    execute_matrix_blas(lhs_layout.value(), rhs_layout.value());
}

template <typename Array>
void MatmulExecutor<Array>::execute_matrix_blas(
    BlasMatrixLayout const & lhs_layout,
    BlasMatrixLayout const & rhs_layout)
{
    if (m_plan.has_batch_axes())
    {
        execute_batched_matrix_blas(lhs_layout, rhs_layout);
        return;
    }
    execute_matrix_blas(0, 0, 0, lhs_layout, rhs_layout);
}

template <typename Array>
void MatmulExecutor<Array>::execute_matrix_blas(
    ssize_t output_base,
    ssize_t lhs_base,
    ssize_t rhs_base,
    BlasMatrixLayout const & lhs_layout,
    BlasMatrixLayout const & rhs_layout)
{
    gemm_blas(
        m_plan.rows(),
        m_plan.columns(),
        m_plan.inner_size(),
        m_lhs.logical_data() + lhs_base,
        m_rhs.logical_data() + rhs_base,
        m_output_data + output_base,
        lhs_layout.transpose(),
        rhs_layout.transpose(),
        lhs_layout.leading_dimension(),
        rhs_layout.leading_dimension(),
        0,
        0,
        0,
        1);
}

template <typename Array>
void MatmulExecutor<Array>::execute_batched_matrix_blas(
    BlasMatrixLayout const & lhs_layout,
    BlasMatrixLayout const & rhs_layout)
{
    size_t const inner_axis = m_plan.batch_rank() - 1;
    size_t const inner_batch_size = static_cast<size_t>(
        m_plan.batch_extent(inner_axis));
    MappedOffsetCursor cursor = m_plan.batch_cursor();
    while (cursor)
    {
        gemm_blas(
            m_plan.rows(),
            m_plan.columns(),
            m_plan.inner_size(),
            m_lhs.logical_data() +
                cursor.offset(MatmulPlan::BatchOperand::Lhs),
            m_rhs.logical_data() +
                cursor.offset(MatmulPlan::BatchOperand::Rhs),
            m_output_data +
                cursor.offset(MatmulPlan::BatchOperand::Output),
            lhs_layout.transpose(),
            rhs_layout.transpose(),
            lhs_layout.leading_dimension(),
            rhs_layout.leading_dimension(),
            m_plan.batch_stride(
                MatmulPlan::BatchOperand::Lhs, inner_axis),
            m_plan.batch_stride(
                MatmulPlan::BatchOperand::Rhs, inner_axis),
            m_plan.batch_stride(
                MatmulPlan::BatchOperand::Output, inner_axis),
            inner_batch_size);
        for (size_t batch = 0; batch < inner_batch_size; ++batch)
        {
            cursor.advance();
        }
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_vector_blas()
{
    if (m_plan.lhs_vector() && m_plan.rhs_vector())
    {
        m_output_data[0] = dot_blas(
            m_plan.inner_size(),
            m_lhs.logical_data(),
            m_plan.lhs_inner_stride(),
            m_rhs.logical_data(),
            m_plan.rhs_inner_stride());
        return;
    }

    std::optional<BlasMatrixLayout> const matrix_layout =
        m_plan.lhs_vector() ? rhs_blas_layout(m_plan)
                            : lhs_blas_layout(m_plan);
    if (!matrix_layout)
    {
        throw std::logic_error(
            "direct planned matmul requires a BLAS matrix layout");
    }
    if (can_execute_affine_vector_blas())
    {
        execute_affine_vector_blas(matrix_layout.value());
        return;
    }
    if (!m_plan.has_batch_axes())
    {
        execute_vector_blas(0, 0, 0, matrix_layout.value());
        return;
    }

    for (MappedOffsetCursor cursor = m_plan.batch_cursor(); cursor; cursor.advance())
    {
        execute_vector_blas(
            cursor.offset(MatmulPlan::BatchOperand::Output),
            cursor.offset(MatmulPlan::BatchOperand::Lhs),
            cursor.offset(MatmulPlan::BatchOperand::Rhs),
            matrix_layout.value());
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_vector_blas(
    ssize_t output_base,
    ssize_t lhs_base,
    ssize_t rhs_base,
    BlasMatrixLayout const & matrix_layout)
{
    if (m_plan.lhs_vector())
    {
        bool const transpose = !matrix_layout.transpose();
        ssize_t const rows =
            matrix_layout.transpose() ? m_plan.columns() : m_plan.inner_size();
        ssize_t const columns =
            matrix_layout.transpose() ? m_plan.inner_size() : m_plan.columns();
        gemv_blas(
            rows,
            columns,
            m_rhs.logical_data() + rhs_base,
            m_lhs.logical_data() + lhs_base,
            m_output_data + output_base,
            transpose,
            matrix_layout.leading_dimension(),
            m_plan.lhs_inner_stride(),
            0,
            0,
            1);
        return;
    }

    ssize_t const rows =
        matrix_layout.transpose() ? m_plan.inner_size() : m_plan.rows();
    ssize_t const columns =
        matrix_layout.transpose() ? m_plan.rows() : m_plan.inner_size();
    gemv_blas(
        rows,
        columns,
        m_lhs.logical_data() + lhs_base,
        m_rhs.logical_data() + rhs_base,
        m_output_data + output_base,
        matrix_layout.transpose(),
        matrix_layout.leading_dimension(),
        m_plan.rhs_inner_stride(),
        0,
        0,
        1);
}

template <typename Array>
bool MatmulExecutor<Array>::can_execute_affine_vector_blas() const
{
    Array const & matrix =
        m_plan.lhs_vector() ? m_rhs : m_lhs;
    return matrix.is_c_contiguous() && m_output.is_c_contiguous();
}

template <typename Array>
void MatmulExecutor<Array>::execute_affine_vector_blas(
    BlasMatrixLayout const & matrix_layout)
{
    ssize_t const matrix_size =
        m_plan.rows() * m_plan.inner_size() * m_plan.columns();
    ssize_t const output_size = static_cast<ssize_t>(
        m_plan.lhs_vector() ? m_plan.columns() : m_plan.rows());
    if (m_plan.lhs_vector())
    {
        bool const transpose = !matrix_layout.transpose();
        ssize_t const rows =
            matrix_layout.transpose() ? m_plan.columns() : m_plan.inner_size();
        ssize_t const columns =
            matrix_layout.transpose() ? m_plan.inner_size() : m_plan.columns();
        gemv_blas(
            rows,
            columns,
            m_rhs.logical_data(),
            m_lhs.logical_data(),
            m_output_data,
            transpose,
            matrix_layout.leading_dimension(),
            m_plan.lhs_inner_stride(),
            matrix_size,
            output_size,
            m_plan.batch_size());
        return;
    }

    ssize_t const rows =
        matrix_layout.transpose() ? m_plan.inner_size() : m_plan.rows();
    ssize_t const columns =
        matrix_layout.transpose() ? m_plan.rows() : m_plan.inner_size();
    gemv_blas(
        rows,
        columns,
        m_lhs.logical_data(),
        m_rhs.logical_data(),
        m_output_data,
        matrix_layout.transpose(),
        matrix_layout.leading_dimension(),
        m_plan.rhs_inner_stride(),
        matrix_size,
        output_size,
        m_plan.batch_size());
}

template <typename Array>
void MatmulExecutor<Array>::execute_packed_blas(
    bool pack_lhs,
    bool pack_rhs)
{
    std::optional<Array> packed_lhs;
    std::optional<Array> packed_rhs;
    Array const * ready_lhs = &m_lhs;
    Array const * ready_rhs = &m_rhs;
    if (pack_lhs)
    {
        packed_lhs.emplace(m_lhs.to_row_major());
        ready_lhs = &packed_lhs.value();
    }
    if (pack_rhs)
    {
        packed_rhs.emplace(m_rhs.to_row_major());
        ready_rhs = &packed_rhs.value();
    }

    MatmulPlan const ready_plan =
        MatmulPlan::make(*ready_lhs, *ready_rhs);
    MatmulExecutor ready_executor(
        ready_plan, m_output, *ready_lhs, *ready_rhs);
    ready_executor.execute_direct_blas();
}

template <typename A, typename T>
class SimpleArrayMatmulHelper
{

public:

    using value_type = T;
    using shape_type = typename A::shape_type;

    SimpleArrayMatmulHelper() = delete;
    SimpleArrayMatmulHelper(A const & lhs, A const & rhs);
    SimpleArrayMatmulHelper(A const & lhs,
                            A const & rhs,
                            ssize_t tile_x,
                            ssize_t tile_y,
                            ssize_t tile_z);
    ~SimpleArrayMatmulHelper() = default;

    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper &&) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper &&) = delete;

    A matmul();
    A matmul_fast();
    A matmul_blas();

private:

    static std::string shape_str(A const & arr);
    void check_dims() const;
    void check_inner(size_t lhs_idx, size_t rhs_idx) const;
    void check_tiles() const;
    A matmul_vec_vec();
    A matmul_vec_vec_blas();
    A matmul_vec_mat();
    A matmul_vec_mat_blas();
    A matmul_mat_vec();
    A matmul_mat_vec_blas();
    A matmul_mat_mat();
    A matmul_mat_mat_blas();
    A pack_rhs(ssize_t n, ssize_t k);
    void accumulate_tile(A const & packed_rhs,
                         ssize_t row_begin,
                         ssize_t row_end,
                         ssize_t col_begin,
                         ssize_t col_end,
                         ssize_t inner_begin,
                         ssize_t inner_end);
    A matmul_mat_mat_tiled();

    A const & m_lhs;
    A const & m_rhs;
    A m_result;
    ssize_t m_tile_x;
    ssize_t m_tile_y;
    ssize_t m_tile_z;

}; /* end class SimpleArrayMatmulHelper */

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs, A const & rhs)
    : SimpleArrayMatmulHelper(lhs, rhs, 0, 0, 0)
{
}

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs,
                                                       A const & rhs,
                                                       ssize_t tile_x,
                                                       ssize_t tile_y,
                                                       ssize_t tile_z)
    : m_lhs(lhs)
    , m_rhs(rhs)
    , m_tile_x(tile_x)
    , m_tile_y(tile_y)
    , m_tile_z(tile_z)
{
    check_dims();

    size_t const lhs_ndim = m_lhs.ndim();
    size_t const rhs_ndim = m_rhs.ndim();

    if (lhs_ndim == 1 && rhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(1);
        return;
    }

    if (lhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(m_rhs.shape(1));
        return;
    }

    if (rhs_ndim == 1)
    {
        check_inner(1, 0);
        m_result = A(m_lhs.shape(0));
        return;
    }

    check_inner(1, 0);
    shape_type const result_shape{m_lhs.shape(0), m_rhs.shape(1)};
    m_result = A(result_shape);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat();
}

/**
 * Perform fast matrix multiplication for SimpleArrays.
 * This implementation currently uses tiling for 2D x 2D matrix multiplication.
 * Future optimizations may add other techniques such as SIMD kernels.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_fast()
{
    check_tiles();

    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat_tiled();
}

/**
 * Perform matrix multiplication using vendor BLAS when available.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_blas()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec_blas();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat_blas();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec_blas();
    }

    return matmul_mat_mat_blas();
}

/**
 * Format shape for matrix multiplication diagnostics.
 */
template <typename A, typename T>
std::string SimpleArrayMatmulHelper<A, T>::shape_str(A const & arr)
{
    if (arr.ndim() == 0)
    {
        return "()";
    }

    std::string result = "(";
    for (size_t i = 0; i < arr.ndim(); ++i)
    {
        if (i > 0)
        {
            result += ",";
        }
        result += std::to_string(arr.shape(i));
    }
    result += ")";
    return result;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_dims() const
{
    bool const lhs_is_supported = m_lhs.ndim() == 1 || m_lhs.ndim() == 2;
    bool const rhs_is_supported = m_rhs.ndim() == 1 || m_rhs.ndim() == 2;
    if (lhs_is_supported && rhs_is_supported)
    {
        return;
    }

    std::string const err = std::format("SimpleArray::matmul(): unsupported dimensions: "
                                        "this={} other={}. SimpleArray must be 1D or 2D.",
                                        shape_str(m_lhs),
                                        shape_str(m_rhs));
    throw std::out_of_range(err);
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_inner(size_t lhs_idx, size_t rhs_idx) const
{
    if (m_lhs.shape(lhs_idx) == m_rhs.shape(rhs_idx))
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul(): shape mismatch: this={} other={}",
                    shape_str(m_lhs),
                    shape_str(m_rhs)));
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_tiles() const
{
    if (m_tile_x > 0 && m_tile_y > 0 && m_tile_z > 0)
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul_fast(): tile sizes must be positive: "
                    "tile_x={} tile_y={} tile_z={}",
                    m_tile_x,
                    m_tile_y,
                    m_tile_z));
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec()
{
    ssize_t const k = m_lhs.shape(0);
    value_type v = 0;
    for (ssize_t i = 0; i < k; ++i)
    {
        v += m_lhs(i) * m_rhs(i);
    }
    m_result.data(0) = v;
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_lhs.shape(0);
        m_result.data(0) = dot_blas(k, m_lhs.data(), m_rhs.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat()
{
    ssize_t const n = m_result.shape(0);
    ssize_t const k = m_lhs.shape(0);
    for (ssize_t j = 0; j < n; ++j)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(l) * m_rhs(l, j);
        }
        m_result.data(j) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_rhs.shape(0);
        ssize_t const n = m_rhs.shape(1);
        bool const transpose_matrix = true;
        gemv_blas(k,
                  n,
                  m_rhs.data(),
                  m_lhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(i, l) * m_rhs(l);
        }
        m_result.data(i) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_lhs.shape(0);
        ssize_t const k = m_lhs.shape(1);
        bool const transpose_matrix = false;
        gemv_blas(m,
                  k,
                  m_lhs.data(),
                  m_rhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            value_type v = 0;
            for (ssize_t l = 0; l < k; ++l)
            {
                v += m_lhs(i, l) * m_rhs(l, j);
            }
            m_result(i, j) = v;
        }
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_result.shape(0);
        ssize_t const n = m_result.shape(1);
        ssize_t const k = m_lhs.shape(1);
        gemm_blas(m, n, k, m_lhs.data(), m_rhs.data(), m_result.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::pack_rhs(ssize_t n, ssize_t k)
{
    shape_type const packing_shape{n, k};
    A packing(packing_shape);
    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t j = 0; j < k; ++j)
        {
            packing(i, j) = m_rhs(j, i);
        }
    }
    return packing;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::accumulate_tile(A const & packed_rhs,
                                                    ssize_t row_begin,
                                                    ssize_t row_end,
                                                    ssize_t col_begin,
                                                    ssize_t col_end,
                                                    ssize_t inner_begin,
                                                    ssize_t inner_end)
{
    for (ssize_t i = row_begin; i < row_end; ++i)
    {
        for (ssize_t j = col_begin; j < col_end; ++j)
        {
            value_type v = m_result(i, j);
            for (ssize_t l = inner_begin; l < inner_end; ++l)
            {
                v += m_lhs(i, l) * packed_rhs(j, l);
            }
            m_result(i, j) = v;
        }
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_tiled()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    A packed_rhs = pack_rhs(n, k);
    for (size_t i = 0; i < m_result.size(); ++i)
    {
        m_result.data(i) = value_type{0};
    }
    for (ssize_t row = 0; row < m; row += m_tile_x)
    {
        ssize_t const row_end = std::min(row + m_tile_x, m);
        for (ssize_t col = 0; col < n; col += m_tile_y)
        {
            ssize_t const col_end = std::min(col + m_tile_y, n);
            for (ssize_t inner = 0; inner < k; inner += m_tile_z)
            {
                ssize_t const inner_end = std::min(inner + m_tile_z, k);
                accumulate_tile(packed_rhs, row, row_end, col, col_end, inner, inner_end);
            }
        }
    }
    return std::move(m_result);
}

} /* end namespace detail */

} /* end namespace solvcon */
