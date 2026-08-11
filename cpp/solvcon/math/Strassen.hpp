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
 * Inputs are row-major non-transposed views. The output is a row-major view
 * whose leading dimension is at least its column count, and must not overlap
 * either input.
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
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace solvcon
{

namespace detail
{

namespace strassen
{

template <typename T>
struct Gemm
{
    ssize_t rows;
    ssize_t columns;
    ssize_t inner_size;
    BlasMatrixView<T> lhs;
    BlasMatrixView<T> rhs;
    BlasOutputView<T> output;
    T alpha;
    T beta;
}; /* end struct Gemm */

/**
 * @brief Reuse scratch storage along one Strassen recursion path.
 *
 * Workspace owns a grow-only buffer and provides stack-like allocation.
 * prepare() resets the cursor for one multiplication, while mark() and
 * rewind() bound the storage used by each recursion level. Because each Step
 * evaluates P1-P7 sequentially, sibling products reuse the same child storage
 * instead of reserving space for the full recursion tree.
 *
 * At depth 2, the buffer holds scratch blocks for two simultaneously active
 * Steps, rather than all eight non-leaf Steps or 49 leaf products. Its
 * capacity remains available for later multiplications.
 */
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

constexpr size_t leaf_workspace_size(size_t lhs_size, size_t rhs_size, size_t product_size) noexcept
{
    return std::max({lhs_size + rhs_size, rhs_size + product_size, lhs_size + product_size});
}

inline constexpr size_t DEPTH1_TRANSFORM_LANE_COUNT = 4;
inline constexpr size_t DEPTH1_TRANSFORM_MIN_ELEMENTS = size_t{1} << 20;

/**
 * @brief Select serial or parallel execution for depth-1 block transforms.
 */
enum class TransformSchedule
{
    Serial,
    PreferParallel,
}; /* end enum class TransformSchedule */

/**
 * @brief Reuse one worker team across the transforms of a depth-1 step.
 *
 * The caller occupies one lane. The remaining workers sleep while a leaf
 * callback runs, then wake for the next row-independent block transform.
 * One caller may submit one transform at a time. The row callback executes
 * concurrently across the caller and worker lanes.
 */
class TransformTeam
{
public:
    TransformTeam();
    ~TransformTeam();

    TransformTeam(TransformTeam const &) = delete;
    TransformTeam & operator=(TransformTeam const &) = delete;
    TransformTeam(TransformTeam &&) = delete;
    TransformTeam & operator=(TransformTeam &&) = delete;

    /**
     * Execute a row callback across the fixed transform team.
     *
     * @tparam Function Callable accepting the half-open row range `[first, last)`.
     * @param rows Number of rows to partition.
     * @param function Row callback invoked concurrently by each lane.
     */
    template <typename Function>
    void run(ssize_t rows, Function const & function);

private:
    using invoke_type = void (*)(void const *, ssize_t, ssize_t);

    template <typename Function>
    static void invoke(void const * function, ssize_t first, ssize_t last);

    void run_lane(size_t lane) noexcept;
    void worker_loop(size_t lane);
    void stop() noexcept;

    std::mutex m_mutex;
    std::condition_variable m_start;
    std::condition_variable m_done;
    void const * m_function = nullptr;
    invoke_type m_invoke = nullptr;
    ssize_t m_rows = 0;
    size_t m_generation = 0;
    size_t m_completed_workers = 0;
    bool m_stopping = false;
    std::exception_ptr m_errors[DEPTH1_TRANSFORM_LANE_COUNT];
    // Keep workers last so they join before the shared job state is destroyed.
    std::jthread m_workers[DEPTH1_TRANSFORM_LANE_COUNT - 1];
}; /* end class TransformTeam */

template <typename Function>
void TransformTeam::run(ssize_t rows, Function const & function)
{
    {
        std::lock_guard lock(m_mutex);
        for (std::exception_ptr & error : m_errors)
        {
            error = {};
        }
        m_function = &function;
        m_invoke = invoke<Function>;
        m_rows = rows;
        m_completed_workers = 0;
        ++m_generation;
    }
    m_start.notify_all();
    run_lane(DEPTH1_TRANSFORM_LANE_COUNT - 1);

    std::unique_lock lock(m_mutex);
    m_done.wait(lock, [this]
                { return m_completed_workers == DEPTH1_TRANSFORM_LANE_COUNT - 1; });
    lock.unlock();
    for (std::exception_ptr const & error : m_errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }
}

template <typename Function>
void TransformTeam::invoke(void const * function, ssize_t first, ssize_t last)
{
    (*static_cast<Function const *>(function))(first, last);
}

template <typename Function>
void run_transform_rows(
    TransformTeam * team,
    ssize_t rows,
    ssize_t columns,
    Function const & function)
{
    size_t const element_count = static_cast<size_t>(rows) * static_cast<size_t>(columns);
    if (team && element_count >= DEPTH1_TRANSFORM_MIN_ELEMENTS)
    {
        team->run(rows, function);
        return;
    }
    function(0, rows);
}

template <size_t Depth>
size_t workspace_size(ssize_t rows, ssize_t columns, ssize_t inner_size)
{
    if constexpr (Depth == 0)
    {
        return 0;
    }
    else
    {
        rows /= 2;
        columns /= 2;
        inner_size /= 2;
        auto const block_rows = static_cast<size_t>(rows);
        auto const block_columns = static_cast<size_t>(columns);
        auto const block_inner_size = static_cast<size_t>(inner_size);
        size_t const lhs_size = block_rows * block_inner_size;
        size_t const rhs_size = block_inner_size * block_columns;
        size_t const product_size = block_rows * block_columns;
        if constexpr (Depth > 1)
        {
            return lhs_size + rhs_size + product_size +
                   workspace_size<Depth - 1>(rows, columns, inner_size);
        }
        else
        {
            return leaf_workspace_size(lhs_size, rhs_size, product_size);
        }
    }
}

template <size_t Depth, typename T>
void validate(Gemm<T> const & gemm)
{
    static_assert(Depth <= 2, "Strassen GEMM supports recursion depths 0, 1, and 2");
    if (gemm.rows <= 0 || gemm.columns <= 0 || gemm.inner_size <= 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be positive");
    }
    size_t const divisor = size_t{1} << Depth;
    if (gemm.rows % divisor != 0 || gemm.columns % divisor != 0 || gemm.inner_size % divisor != 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be divisible by 2^depth");
    }
    if (gemm.lhs.m_transpose != BlasTranspose::None || gemm.rhs.m_transpose != BlasTranspose::None)
    {
        throw std::invalid_argument("Strassen GEMM does not support transposed input views");
    }
    if (gemm.lhs.m_leading_dimension < gemm.inner_size ||
        gemm.rhs.m_leading_dimension < gemm.columns)
    {
        throw std::invalid_argument("Strassen GEMM input leading dimensions are too small");
    }
    if (gemm.output.m_leading_dimension < gemm.columns)
    {
        throw std::invalid_argument("Strassen GEMM output leading dimension is too small");
    }
    if constexpr (Depth > 0)
    {
        if (gemm.alpha != T{1} || gemm.beta != T{0})
        {
            throw std::invalid_argument("Strassen recursion requires alpha=1 and beta=0");
        }
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
BlasOutputView<T> make_subview(BlasOutputView<T> matrix, ssize_t row, ssize_t column)
{
    return {matrix.m_data + row * matrix.m_leading_dimension + column,
            matrix.m_leading_dimension};
}

template <typename T>
void combine_block(
    TransformTeam * team,
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    T * output,
    ssize_t rows,
    ssize_t columns,
    T rhs_scale)
{
    auto const combine_rows = [=](ssize_t first, ssize_t last)
    {
        for (ssize_t row = first; row < last; ++row)
        {
            T const * lhs_row = lhs.m_data + row * lhs.m_leading_dimension;
            T const * rhs_row = rhs.m_data + row * rhs.m_leading_dimension;
            T * output_row = output + row * columns;
            for (ssize_t column = 0; column < columns; ++column)
            {
                output_row[column] = lhs_row[column] + rhs_scale * rhs_row[column];
            }
        }
    };
    run_transform_rows(team, rows, columns, combine_rows);
}

template <typename T>
void copy_block(
    TransformTeam * team,
    BlasOutputView<T> output,
    BlasOutputView<T> input,
    ssize_t rows,
    ssize_t columns)
{
    auto const copy_rows = [=](ssize_t first, ssize_t last)
    {
        for (ssize_t row = first; row < last; ++row)
        {
            T * output_row = output.m_data + row * output.m_leading_dimension;
            T const * input_row = input.m_data + row * input.m_leading_dimension;
            std::copy_n(input_row, columns, output_row);
        }
    };
    run_transform_rows(team, rows, columns, copy_rows);
}

template <typename T>
void add_block(
    TransformTeam * team,
    BlasOutputView<T> output,
    BlasOutputView<T> input,
    ssize_t rows,
    ssize_t columns,
    T scale)
{
    auto const add_rows = [=](ssize_t first, ssize_t last)
    {
        for (ssize_t row = first; row < last; ++row)
        {
            T * output_row = output.m_data + row * output.m_leading_dimension;
            T const * input_row = input.m_data + row * input.m_leading_dimension;
            for (ssize_t column = 0; column < columns; ++column)
            {
                output_row[column] += scale * input_row[column];
            }
        }
    };
    run_transform_rows(team, rows, columns, add_rows);
}

template <typename T>
void add_block_to_pair(
    TransformTeam * team,
    BlasOutputView<T> output1,
    T scale1,
    BlasOutputView<T> output2,
    T scale2,
    BlasOutputView<T> input,
    ssize_t rows,
    ssize_t columns)
{
    auto const add_rows = [=](ssize_t first, ssize_t last)
    {
        for (ssize_t row = first; row < last; ++row)
        {
            T * output1_row = output1.m_data + row * output1.m_leading_dimension;
            T * output2_row = output2.m_data + row * output2.m_leading_dimension;
            T const * input_row = input.m_data + row * input.m_leading_dimension;
            for (ssize_t column = 0; column < columns; ++column)
            {
                T const value = input_row[column];
                output1_row[column] += scale1 * value;
                output2_row[column] += scale2 * value;
            }
        }
    };
    run_transform_rows(team, rows, columns, add_rows);
}

/**
 * @brief Evaluate one level of Strassen matrix multiplication.
 *
 * Step divides one Gemm descriptor into lhs, rhs, and output quadrants.
 * evaluate() keeps lhs, rhs, and product blocks live for recursive execution.
 * evaluate_leaf() sends P1-P3 and P6-P7 to output quadrants, while P4 and P5
 * share product scratch. Each product repartitions the same arena for the
 * two scratch blocks that coexist. The caller-provided callback decides whether a product
 * recurses or reaches the leaf backend.
 *
 * For an `8 x 12 x 16` contraction, one Step forms seven `4 x 6 x 8`
 * products. Its leaf execution needs at most two of the three block types at
 * once.
 */
template <typename T>
class Step
{
public:
    Step(Gemm<T> const & gemm, Workspace<T> & workspace);

    template <typename Multiply>
    void evaluate(Multiply const & multiply) const;

    template <typename Multiply>
    void evaluate_leaf(Multiply const & multiply, TransformTeam * team) const;

private:
    struct Product
    {
        ssize_t rows;
        ssize_t columns;
        ssize_t inner_size;
    }; /* end struct Product */

    size_t lhs_size() const { return static_cast<size_t>(m_product.rows * m_product.inner_size); }
    size_t rhs_size() const { return static_cast<size_t>(m_product.inner_size * m_product.columns); }
    size_t product_size() const { return static_cast<size_t>(m_product.rows * m_product.columns); }
    BlasMatrixView<T> lhs_block(T * data) const { return {data, m_product.inner_size, BlasTranspose::None}; }
    BlasMatrixView<T> rhs_block(T * data) const { return {data, m_product.columns, BlasTranspose::None}; }
    BlasOutputView<T> product_block(T * data) const { return {data, m_product.columns}; }
    void form_lhs(
        TransformTeam * team,
        T * output,
        BlasMatrixView<T> lhs,
        BlasMatrixView<T> rhs,
        T rhs_scale) const
    {
        combine_block(team, lhs, rhs, output, m_product.rows, m_product.inner_size, rhs_scale);
    }
    void form_rhs(
        TransformTeam * team,
        T * output,
        BlasMatrixView<T> lhs,
        BlasMatrixView<T> rhs,
        T rhs_scale) const
    {
        combine_block(team, lhs, rhs, output, m_product.inner_size, m_product.columns, rhs_scale);
    }

    template <typename Multiply>
    void multiply_product(
        BlasMatrixView<T> lhs,
        BlasMatrixView<T> rhs,
        BlasOutputView<T> output,
        Multiply const & multiply) const;

    template <typename Multiply>
    void multiply_into(
        BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, BlasOutputView<T> output, T alpha, T beta, Multiply const & multiply) const;

    void add_output_pair(
        TransformTeam * team,
        BlasOutputView<T> destination1,
        T scale1,
        BlasOutputView<T> destination2,
        T scale2,
        BlasOutputView<T> source) const
    {
        add_block_to_pair(
            team, destination1, scale1, destination2, scale2, source, m_product.rows, m_product.columns);
    }
    void copy_output(
        TransformTeam * team,
        BlasOutputView<T> destination,
        BlasOutputView<T> source) const
    {
        copy_block(team, destination, source, m_product.rows, m_product.columns);
    }
    void add_output(
        TransformTeam * team,
        BlasOutputView<T> destination,
        BlasOutputView<T> source,
        T scale) const
    {
        add_block(team, destination, source, m_product.rows, m_product.columns, scale);
    }
    Workspace<T> & m_workspace;
    Product m_product;
    BlasMatrixView<T> m_a11;
    BlasMatrixView<T> m_a12;
    BlasMatrixView<T> m_a21;
    BlasMatrixView<T> m_a22;
    BlasMatrixView<T> m_b11;
    BlasMatrixView<T> m_b12;
    BlasMatrixView<T> m_b21;
    BlasMatrixView<T> m_b22;
    BlasOutputView<T> m_c11;
    BlasOutputView<T> m_c12;
    BlasOutputView<T> m_c21;
    BlasOutputView<T> m_c22;
}; /* end class Step */

template <typename T>
Step<T>::Step(Gemm<T> const & gemm, Workspace<T> & workspace)
    : m_workspace(workspace)
    , m_product{
          gemm.rows / 2,
          gemm.columns / 2,
          gemm.inner_size / 2,
      }
    , m_a11(make_subview(gemm.lhs, 0, 0))
    , m_a12(make_subview(gemm.lhs, 0, m_product.inner_size))
    , m_a21(make_subview(gemm.lhs, m_product.rows, 0))
    , m_a22(make_subview(gemm.lhs, m_product.rows, m_product.inner_size))
    , m_b11(make_subview(gemm.rhs, 0, 0))
    , m_b12(make_subview(gemm.rhs, 0, m_product.columns))
    , m_b21(make_subview(gemm.rhs, m_product.inner_size, 0))
    , m_b22(make_subview(gemm.rhs, m_product.inner_size, m_product.columns))
    , m_c11(make_subview(gemm.output, 0, 0))
    , m_c12(make_subview(gemm.output, 0, m_product.columns))
    , m_c21(make_subview(gemm.output, m_product.rows, 0))
    , m_c22(make_subview(gemm.output, m_product.rows, m_product.columns))
{
}

template <typename T>
template <typename Multiply>
void Step<T>::multiply_product(
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    BlasOutputView<T> output,
    Multiply const & multiply) const
{
    multiply_into(lhs, rhs, output, T{1}, T{0}, multiply);
}

template <typename T>
template <typename Multiply>
void Step<T>::multiply_into(
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    BlasOutputView<T> output,
    T alpha,
    T beta,
    Multiply const & multiply) const
{
    Gemm<T> const product{
        m_product.rows,
        m_product.columns,
        m_product.inner_size,
        lhs,
        rhs,
        output,
        alpha,
        beta,
    };
    multiply(product);
}

template <typename T>
template <typename Multiply>
void Step<T>::evaluate(Multiply const & multiply) const
{
    T * const lhs_scratch = m_workspace.allocate(lhs_size() + rhs_size() + product_size());
    T * const rhs_scratch = lhs_scratch + lhs_size();
    BlasOutputView<T> const product = product_block(rhs_scratch + rhs_size());

    // P1 = (A11 + A22)(B11 + B22); initialize C11 and C22.
    form_lhs(nullptr, lhs_scratch, m_a11, m_a22, T{1});
    form_rhs(nullptr, rhs_scratch, m_b11, m_b22, T{1});
    multiply_product(lhs_block(lhs_scratch), rhs_block(rhs_scratch), product, multiply);
    copy_output(nullptr, m_c11, product);
    copy_output(nullptr, m_c22, product);

    // P2 = (A21 + A22)B11; initialize C21 and subtract from C22.
    form_lhs(nullptr, lhs_scratch, m_a21, m_a22, T{1});
    multiply_product(lhs_block(lhs_scratch), m_b11, product, multiply);
    copy_output(nullptr, m_c21, product);
    add_output(nullptr, m_c22, product, T{-1});

    // P3 = A11(B12 - B22); initialize C12 and add to C22.
    form_rhs(nullptr, rhs_scratch, m_b12, m_b22, T{-1});
    multiply_product(m_a11, rhs_block(rhs_scratch), product, multiply);
    copy_output(nullptr, m_c12, product);
    add_output(nullptr, m_c22, product, T{1});

    // P4 = A22(B21 - B11); add to C11 and C21.
    form_rhs(nullptr, rhs_scratch, m_b21, m_b11, T{-1});
    multiply_product(m_a22, rhs_block(rhs_scratch), product, multiply);
    add_output_pair(nullptr, m_c11, T{1}, m_c21, T{1}, product);

    // P5 = (A11 + A12)B22; subtract from C11 and add to C12.
    form_lhs(nullptr, lhs_scratch, m_a11, m_a12, T{1});
    multiply_product(lhs_block(lhs_scratch), m_b22, product, multiply);
    add_output_pair(nullptr, m_c11, T{-1}, m_c12, T{1}, product);

    // P6 = (A21 - A11)(B11 + B12); add to C22.
    form_lhs(nullptr, lhs_scratch, m_a21, m_a11, T{-1});
    form_rhs(nullptr, rhs_scratch, m_b11, m_b12, T{1});
    multiply_product(lhs_block(lhs_scratch), rhs_block(rhs_scratch), product, multiply);
    add_output(nullptr, m_c22, product, T{1});

    // P7 = (A12 - A22)(B21 + B22); add to C11.
    form_lhs(nullptr, lhs_scratch, m_a12, m_a22, T{-1});
    form_rhs(nullptr, rhs_scratch, m_b21, m_b22, T{1});
    multiply_product(lhs_block(lhs_scratch), rhs_block(rhs_scratch), product, multiply);
    add_output(nullptr, m_c11, product, T{1});
}

template <typename T>
template <typename Multiply>
void Step<T>::evaluate_leaf(Multiply const & multiply, TransformTeam * team) const
{
    size_t const scratch_size = leaf_workspace_size(lhs_size(), rhs_size(), product_size());
    T * const scratch = m_workspace.allocate(scratch_size);
    T * const lhs_scratch = scratch;
    T * const rhs_scratch = lhs_scratch + lhs_size();

    form_lhs(team, lhs_scratch, m_a11, m_a22, T{1});
    form_rhs(team, rhs_scratch, m_b11, m_b22, T{1});
    multiply_into(lhs_block(lhs_scratch), rhs_block(rhs_scratch), m_c11, T{1}, T{0}, multiply);
    copy_output(team, m_c22, m_c11);

    form_lhs(team, lhs_scratch, m_a21, m_a22, T{1});
    multiply_into(lhs_block(lhs_scratch), m_b11, m_c21, T{1}, T{0}, multiply);
    add_output(team, m_c22, m_c21, T{-1});

    form_rhs(team, rhs_scratch, m_b12, m_b22, T{-1});
    multiply_into(m_a11, rhs_block(rhs_scratch), m_c12, T{1}, T{0}, multiply);
    add_output(team, m_c22, m_c12, T{1});

    T * const p4_rhs = scratch;
    BlasOutputView<T> const p4 = product_block(p4_rhs + rhs_size());
    form_rhs(team, p4_rhs, m_b21, m_b11, T{-1});
    multiply_product(m_a22, rhs_block(p4_rhs), p4, multiply);
    add_output_pair(team, m_c11, T{1}, m_c21, T{1}, p4);

    T * const p5_lhs = scratch;
    BlasOutputView<T> const p5 = product_block(p5_lhs + lhs_size());
    form_lhs(team, p5_lhs, m_a11, m_a12, T{1});
    multiply_product(lhs_block(p5_lhs), m_b22, p5, multiply);
    add_output_pair(team, m_c11, T{-1}, m_c12, T{1}, p5);

    form_lhs(team, lhs_scratch, m_a21, m_a11, T{-1});
    form_rhs(team, rhs_scratch, m_b11, m_b12, T{1});
    multiply_into(lhs_block(lhs_scratch), rhs_block(rhs_scratch), m_c22, T{1}, T{1}, multiply);

    form_lhs(team, lhs_scratch, m_a12, m_a22, T{-1});
    form_rhs(team, rhs_scratch, m_b21, m_b22, T{1});
    multiply_into(lhs_block(lhs_scratch), rhs_block(rhs_scratch), m_c11, T{1}, T{1}, multiply);
}

template <typename T, typename Leaf>
class Kernel
{
public:
    Kernel(Workspace<T> & workspace, Leaf const & leaf)
        : m_workspace(workspace)
        , m_leaf(leaf)
    {
    }

    template <size_t Depth>
    void multiply(Gemm<T> const & gemm, TransformSchedule transform_schedule);

private:
    template <size_t Depth>
    void recurse(Gemm<T> const & gemm, TransformTeam * team);

    Workspace<T> & m_workspace;
    Leaf const & m_leaf;
}; /* end class Kernel */

template <typename T, typename Leaf>
template <size_t Depth>
void Kernel<T, Leaf>::multiply(
    Gemm<T> const & gemm,
    TransformSchedule transform_schedule)
{
    validate<Depth>(gemm);
    m_workspace.prepare(workspace_size<Depth>(gemm.rows, gemm.columns, gemm.inner_size));

    std::optional<TransformTeam> transform_team;
    if constexpr (Depth == 1)
    {
        if (transform_schedule == TransformSchedule::PreferParallel)
        {
            size_t const block_rows = static_cast<size_t>(gemm.rows / 2);
            size_t const block_columns = static_cast<size_t>(gemm.columns / 2);
            size_t const block_inner_size = static_cast<size_t>(gemm.inner_size / 2);
            size_t const maximum_elements = std::max({
                block_rows * block_inner_size,
                block_inner_size * block_columns,
                block_rows * block_columns,
            });
            if (maximum_elements >= DEPTH1_TRANSFORM_MIN_ELEMENTS)
            {
                try
                {
                    transform_team.emplace();
                }
                catch (std::system_error const &)
                {
                    // Retain serial execution when the OS cannot create workers.
                }
            }
        }
    }
    recurse<Depth>(gemm, transform_team ? &*transform_team : nullptr);
}

template <typename T, typename Leaf>
template <size_t Depth>
void Kernel<T, Leaf>::recurse(Gemm<T> const & gemm, TransformTeam * team)
{
    if constexpr (Depth == 0)
    {
        m_leaf(gemm);
    }
    else
    {
        size_t const mark = m_workspace.mark();
        Step<T> const step(gemm, m_workspace);
        if constexpr (Depth == 1)
        {
            step.evaluate_leaf(m_leaf, team);
        }
        else
        {
            auto const recurse_product = [this](Gemm<T> const & product)
            { this->template recurse<Depth - 1>(product, nullptr); };
            step.evaluate(recurse_product);
        }
        m_workspace.rewind(mark);
    }
}

template <size_t Depth, typename T, typename Leaf>
void multiply(
    Gemm<T> const & gemm,
    Workspace<T> & workspace,
    Leaf const & leaf,
    TransformSchedule transform_schedule)
{
    Kernel<T, Leaf> kernel(workspace, leaf);
    kernel.template multiply<Depth>(gemm, transform_schedule);
}

} /* end namespace strassen */

template <size_t Depth, typename T>
void gemm_strassen(
    strassen::Gemm<T> const & gemm,
    strassen::Workspace<T> & workspace,
    strassen::TransformSchedule transform_schedule)
{
    auto const leaf = [](strassen::Gemm<T> const & leaf_gemm)
    {
        gemm_blas(
            leaf_gemm.rows,
            leaf_gemm.columns,
            leaf_gemm.inner_size,
            leaf_gemm.lhs,
            leaf_gemm.rhs,
            leaf_gemm.output,
            leaf_gemm.alpha,
            leaf_gemm.beta);
    };
    strassen::multiply<Depth>(gemm, workspace, leaf, transform_schedule);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
