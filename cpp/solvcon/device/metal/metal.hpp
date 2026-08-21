#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Metal-backed buffer allocation and asynchronous matrix multiplication.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/device/BufferBackend.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace solvcon
{

class ConcreteBuffer;

namespace device
{

/// Read-only row-major matrix view over a Metal-backed ConcreteBuffer.
struct MetalMatrixView
{
    ConcreteBuffer const * m_buffer;
    size_t m_byte_offset;
    ssize_t m_leading_dimension;
}; /* end struct MetalMatrixView */

/// Writable row-major matrix view over a Metal-backed ConcreteBuffer.
struct MetalOutputView
{
    ConcreteBuffer * m_buffer;
    size_t m_byte_offset;
    ssize_t m_leading_dimension;
}; /* end struct MetalOutputView */

/// Native FP32 GEMM dimensions and buffer views.
struct MetalGemmOperation
{
    ssize_t m_rows;
    ssize_t m_columns;
    ssize_t m_inner_size;
    MetalMatrixView m_lhs;
    MetalMatrixView m_rhs;
    MetalOutputView m_output;
}; /* end struct MetalGemmOperation */

/// Process-wide diagnostic counters for the prototype Metal runtime.
struct MetalStatistics
{
    std::uint64_t m_allocated_buffers;
    std::uint64_t m_submitted_commands;
    std::uint64_t m_host_waits;
    std::uint64_t m_host_exports;
}; /* end struct MetalStatistics */

/// Own the process-wide Metal device and serial command queue.
class MetalManager : public BufferBackend
{
public:

    static MetalManager & instance();

    MetalManager(MetalManager const &) = delete;
    MetalManager(MetalManager &&) = delete;
    MetalManager & operator=(MetalManager const &) = delete;
    MetalManager & operator=(MetalManager &&) = delete;
    ~MetalManager();

    /// Return true when a unified-memory Metal device and queue are ready.
    bool started() const noexcept;

    BufferDevice device() const noexcept override { return BufferDevice::Metal; }
    bool built() const noexcept override { return true; }
    bool available() const noexcept override { return started(); }
    /// Allocate CPU-visible shared Metal storage.
    std::shared_ptr<ConcreteBuffer> allocate(size_t nbytes, size_t alignment) const override;
    /// Submit one FP32 GEMM without waiting for completion.
    void gemm_async(MetalGemmOperation const & operation);

    /// Return process-wide diagnostic counters.
    MetalStatistics statistics() const noexcept;
    /// Reset process-wide diagnostic counters.
    void reset_statistics() noexcept;

private:
    class Impl;

    MetalManager();

    std::unique_ptr<Impl> const m_impl;

}; /* end class MetalManager */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
