/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/device/metal/metal.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace solvcon
{

namespace device
{

namespace
{

struct MetalCounters
{
    std::atomic<std::uint64_t> m_allocated_buffers{0};
    std::atomic<std::uint64_t> m_submitted_commands{0};
    std::atomic<std::uint64_t> m_host_waits{0};
    std::atomic<std::uint64_t> m_host_exports{0};
}; /* end struct MetalCounters */

MetalCounters & metal_counters()
{
    static MetalCounters counters;
    return counters;
}

id<MTLDevice> select_unified_device()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil && device.hasUnifiedMemory)
    {
        return device;
    }
    for (id<MTLDevice> candidate in MTLCopyAllDevices())
    {
        if (candidate.hasUnifiedMemory)
        {
            return candidate;
        }
    }
    return nil;
}

std::string error_description(NSError * error)
{
    if (error == nil || error.localizedDescription == nil || error.localizedDescription.UTF8String == nullptr)
    {
        return "no driver error description";
    }
    return error.localizedDescription.UTF8String;
}

void check_command(id<MTLCommandBuffer> command, std::string_view operation)
{
    if (command.status == MTLCommandBufferStatusCompleted)
    {
        return;
    }
    throw std::runtime_error(
        std::format("Metal {} failed: {}", operation, error_description(command.error)));
}

class MetalTask
{
public:
    MetalTask(
        id<MTLCommandBuffer> command,
        NSArray<id<MTLBuffer>> * resources,
        std::vector<std::shared_ptr<MetalTask>> predecessors);

    bool ready() const;
    void wait() const;
    void attach_completion_handler(std::shared_ptr<MetalTask> const & self);

private:
    enum class State : std::uint8_t
    {
        Pending,
        Finalizing,
        Finalized,
    }; /* end enum class State */

    void finalize() const;
    void rethrow_error() const;

    mutable std::mutex m_mutex;
    mutable std::condition_variable m_condition;
    mutable State m_state = State::Pending;
    mutable std::exception_ptr m_error;
    mutable __strong id<MTLCommandBuffer> m_command;
    mutable __strong NSArray<id<MTLBuffer>> * m_resources;
    mutable std::vector<std::shared_ptr<MetalTask>> m_predecessors;
}; /* end class MetalTask */

class MetalBufferAccess final : public ConcreteBuffer::access_type
{
public:
    MetalBufferAccess(id<MTLBuffer> buffer, size_t resource_offset);

    BufferDevice device() const noexcept override { return BufferDevice::Metal; }

    void begin_host_access(BufferHostAccessMode) const override;
    void end_host_access(BufferHostAccessMode) const noexcept override;
    void export_host_access() const override;
    void wait() const override;
    bool ready() const override;

    bool host_exported() const noexcept override { return m_host_exported.load(); }

    id<MTLBuffer> buffer() const noexcept { return m_buffer; }
    size_t resource_offset() const noexcept { return m_resource_offset; }
    std::mutex & access_mutex() const noexcept { return m_access_mutex; }
    bool host_access_active_unlocked() const noexcept { return m_active_host_accesses != 0; }
    std::shared_ptr<MetalTask> const & last_use_unlocked() const noexcept { return m_last_use; }
    void set_last_use_unlocked(std::shared_ptr<MetalTask> const & task) const { m_last_use = task; }

private:
    std::shared_ptr<MetalTask> last_use() const;
    static void wait_for_task(std::shared_ptr<MetalTask> const & task);

    __strong id<MTLBuffer> m_buffer;
    size_t m_resource_offset;
    mutable std::mutex m_access_mutex;
    mutable std::shared_ptr<MetalTask> m_last_use;
    mutable size_t m_active_host_accesses = 0;
    mutable std::atomic<bool> m_host_exported{false};
}; /* end class MetalBufferAccess */

class MetalBufferRemover final : public ConcreteBuffer::remover_type
{
public:
    MetalBufferRemover(id<MTLBuffer> buffer, size_t resource_offset);

    void operator()(int8_t *, size_t) const override {}
    ConcreteBuffer::access_type const * access_state() const noexcept override { return &m_access; }

private:
    MetalBufferAccess m_access;
}; /* end class MetalBufferRemover */

MetalTask::MetalTask(
    id<MTLCommandBuffer> command,
    NSArray<id<MTLBuffer>> * resources,
    std::vector<std::shared_ptr<MetalTask>> predecessors)
    : m_command(command)
    , m_resources(resources)
    , m_predecessors(std::move(predecessors))
{
}

bool MetalTask::ready() const
{
    std::scoped_lock lock(m_mutex);
    if (m_state == State::Finalized)
    {
        return true;
    }
    MTLCommandBufferStatus const status = m_command.status;
    return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
}

void MetalTask::wait() const
{
    id<MTLCommandBuffer> command = nil;
    {
        std::scoped_lock lock(m_mutex);
        if (m_state != State::Finalized)
        {
            command = m_command;
        }
    }
    if (command != nil)
    {
        [command waitUntilCompleted];
    }
    finalize();
    rethrow_error();
}

void MetalTask::attach_completion_handler(std::shared_ptr<MetalTask> const & self)
{
    std::weak_ptr<MetalTask> weak_self(self);
    [m_command addCompletedHandler:^(id<MTLCommandBuffer>) {
      if (std::shared_ptr<MetalTask> task = weak_self.lock())
      {
          task->finalize();
      }
    }];
}

void MetalTask::finalize() const
{
    std::vector<std::shared_ptr<MetalTask>> predecessors;
    id<MTLCommandBuffer> command = nil;
    {
        std::unique_lock lock(m_mutex);
        if (m_state == State::Finalized)
        {
            return;
        }
        if (m_state == State::Finalizing)
        {
            m_condition.wait(lock, [this]()
                             { return m_state == State::Finalized; });
            return;
        }
        m_state = State::Finalizing;
        predecessors = std::move(m_predecessors);
        command = m_command;
    }

    std::exception_ptr error;
    try
    {
        check_command(command, "command");
    }
    catch (...)
    {
        error = std::current_exception();
    }
    for (std::shared_ptr<MetalTask> const & predecessor : predecessors)
    {
        try
        {
            predecessor->wait();
        }
        catch (...)
        {
            if (!error)
            {
                error = std::current_exception();
            }
        }
    }

    {
        std::scoped_lock lock(m_mutex);
        m_error = error;
        m_command = nil;
        m_resources = nil;
        m_predecessors.clear();
        m_state = State::Finalized;
    }
    m_condition.notify_all();
}

void MetalTask::rethrow_error() const
{
    std::exception_ptr error;
    {
        std::scoped_lock lock(m_mutex);
        error = m_error;
    }
    if (error)
    {
        std::rethrow_exception(error);
    }
}

MetalBufferAccess::MetalBufferAccess(id<MTLBuffer> buffer, size_t resource_offset)
    : m_buffer(buffer)
    , m_resource_offset(resource_offset)
{
}

MetalBufferRemover::MetalBufferRemover(id<MTLBuffer> buffer, size_t resource_offset)
    : m_access(buffer, resource_offset)
{
}

void MetalBufferAccess::begin_host_access(BufferHostAccessMode) const
{
    std::unique_lock lock(m_access_mutex);
    wait_for_task(m_last_use);
    m_last_use.reset();
    ++m_active_host_accesses;
}

void MetalBufferAccess::end_host_access(BufferHostAccessMode) const noexcept
{
    std::scoped_lock lock(m_access_mutex);
    --m_active_host_accesses;
}

void MetalBufferAccess::export_host_access() const
{
    if (m_host_exported.load(std::memory_order_acquire))
    {
        return;
    }

    std::unique_lock lock(m_access_mutex);
    if (m_host_exported.load(std::memory_order_relaxed))
    {
        return;
    }
    wait_for_task(m_last_use);
    m_last_use.reset();
    m_host_exported.store(true, std::memory_order_release);
    ++metal_counters().m_host_exports;
}

void MetalBufferAccess::wait() const
{
    std::unique_lock lock(m_access_mutex);
    wait_for_task(m_last_use);
    m_last_use.reset();
}

bool MetalBufferAccess::ready() const
{
    std::shared_ptr<MetalTask> const task = last_use();
    return !task || task->ready();
}

std::shared_ptr<MetalTask> MetalBufferAccess::last_use() const
{
    std::scoped_lock lock(m_access_mutex);
    return m_last_use;
}

void MetalBufferAccess::wait_for_task(std::shared_ptr<MetalTask> const & task)
{
    if (task)
    {
        ++metal_counters().m_host_waits;
        task->wait();
    }
}

MetalBufferAccess const & require_metal_buffer(ConcreteBuffer const & buffer)
{
    auto const * access = dynamic_cast<MetalBufferAccess const *>(buffer.access_state());
    if (access == nullptr)
    {
        throw std::invalid_argument("Metal operation requires Metal-backed buffers");
    }
    return *access;
}

size_t checked_dimension(ssize_t value, std::string_view name)
{
    if (value < 0)
    {
        throw std::invalid_argument(std::format("Metal GEMM: {} must be non-negative", name));
    }
    return static_cast<size_t>(value);
}

size_t checked_product(size_t lhs, size_t rhs, std::string_view name)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::length_error(std::format("Metal GEMM: {} size overflows", name));
    }
    return lhs * rhs;
}

size_t checked_sum(size_t lhs, size_t rhs, std::string_view name)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
    {
        throw std::length_error(std::format("Metal GEMM: {} size overflows", name));
    }
    return lhs + rhs;
}

void validate_matrix(
    size_t rows,
    size_t columns,
    ssize_t leading_dimension,
    size_t byte_offset,
    ConcreteBuffer const & owner,
    std::string_view name)
{
    if (leading_dimension < static_cast<ssize_t>(columns))
    {
        throw std::invalid_argument(std::format("Metal GEMM: invalid {} leading dimension", name));
    }
    if (rows == 0 || columns == 0)
    {
        return;
    }

    size_t const last_row = checked_product(rows - 1, static_cast<size_t>(leading_dimension), name);
    size_t const elements = checked_sum(last_row, columns, name);
    size_t const matrix_nbytes = checked_product(elements, sizeof(float), name);
    if (byte_offset > owner.nbytes() || matrix_nbytes > owner.nbytes() - byte_offset)
    {
        throw std::out_of_range(std::format("Metal GEMM: {} view exceeds its buffer", name));
    }
}

} /* end namespace */

class MetalManager::Impl
{
public:
    Impl();

    bool started() const noexcept { return m_device != nil && m_queue != nil; }
    std::shared_ptr<ConcreteBuffer> make_buffer(size_t nbytes, size_t alignment);
    void gemm_async(MetalGemmOperation const & operation);

private:
    static NSArray<id<MTLBuffer>> * retain_resources(
        MetalBufferAccess const & lhs,
        MetalBufferAccess const & rhs,
        MetalBufferAccess const & output);

    __strong id<MTLDevice> m_device;
    __strong id<MTLCommandQueue> m_queue;
    std::mutex m_submit_mutex;
}; /* end class MetalManager::Impl */

MetalManager::Impl::Impl()
    : m_device(select_unified_device())
{
    if (m_device == nil || !MPSSupportsMTLDevice(m_device))
    {
        m_device = nil;
        return;
    }
    m_queue = [m_device newCommandQueue];
    if (m_queue == nil)
    {
        m_device = nil;
        return;
    }
}

std::shared_ptr<ConcreteBuffer> MetalManager::Impl::make_buffer(size_t nbytes, size_t alignment)
{
    if (!started())
    {
        throw std::runtime_error("ConcreteBuffer: no unified-memory Metal device is available");
    }
    validate_alignment(alignment, "ConcreteBuffer::construct");
    if (nbytes != 0 && alignment != 0)
    {
        validate_size_alignment(nbytes, alignment, "ConcreteBuffer::construct");
    }

    if (nbytes == 0)
    {
        auto remover = std::make_unique<MetalBufferRemover>(nil, 0);
        return ConcreteBuffer::construct(0, static_cast<int8_t *>(nullptr), std::move(remover), alignment);
    }

    size_t const padding = alignment == 0 ? 0 : alignment - 1;
    size_t const capacity = checked_sum(nbytes, padding, "buffer");
    if (capacity > m_device.maxBufferLength)
    {
        throw std::length_error("ConcreteBuffer: requested Metal buffer exceeds maxBufferLength");
    }

    id<MTLBuffer> buffer = [m_device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
    if (buffer == nil || buffer.contents == nullptr)
    {
        throw std::bad_alloc();
    }

    auto const address = reinterpret_cast<std::uintptr_t>(buffer.contents);
    size_t const resource_offset = alignment == 0 ? 0 : (alignment - address % alignment) % alignment;
    auto * data = static_cast<int8_t *>(buffer.contents) + resource_offset;
    auto remover = std::make_unique<MetalBufferRemover>(buffer, resource_offset);
    ++metal_counters().m_allocated_buffers;
    return ConcreteBuffer::construct(nbytes, data, std::move(remover), alignment);
}

NSArray<id<MTLBuffer>> * MetalManager::Impl::retain_resources(
    MetalBufferAccess const & lhs,
    MetalBufferAccess const & rhs,
    MetalBufferAccess const & output)
{
    return @[ lhs.buffer(), rhs.buffer(), output.buffer() ];
}

void MetalManager::Impl::gemm_async(MetalGemmOperation const & operation)
{
    if (!started())
    {
        throw std::runtime_error("Metal GEMM: no unified-memory Metal device is available");
    }
    if (operation.m_lhs.m_buffer == nullptr ||
        operation.m_rhs.m_buffer == nullptr ||
        operation.m_output.m_buffer == nullptr)
    {
        throw std::invalid_argument("Metal GEMM: null ConcreteBuffer owner");
    }

    size_t const rows = checked_dimension(operation.m_rows, "rows");
    size_t const columns = checked_dimension(operation.m_columns, "columns");
    size_t const inner_size = checked_dimension(operation.m_inner_size, "inner_size");
    if (rows == 0 || columns == 0 || inner_size == 0)
    {
        throw std::invalid_argument("Metal GEMM: empty contractions must be handled by the caller");
    }

    validate_matrix(
        rows,
        inner_size,
        operation.m_lhs.m_leading_dimension,
        operation.m_lhs.m_byte_offset,
        *operation.m_lhs.m_buffer,
        "lhs");
    validate_matrix(
        inner_size,
        columns,
        operation.m_rhs.m_leading_dimension,
        operation.m_rhs.m_byte_offset,
        *operation.m_rhs.m_buffer,
        "rhs");
    validate_matrix(
        rows,
        columns,
        operation.m_output.m_leading_dimension,
        operation.m_output.m_byte_offset,
        *operation.m_output.m_buffer,
        "output");

    MetalBufferAccess const & lhs = require_metal_buffer(*operation.m_lhs.m_buffer);
    MetalBufferAccess const & rhs = require_metal_buffer(*operation.m_rhs.m_buffer);
    MetalBufferAccess const & output = require_metal_buffer(*operation.m_output.m_buffer);
    id<MTLBuffer> lhs_buffer = lhs.buffer();
    id<MTLBuffer> rhs_buffer = rhs.buffer();
    id<MTLBuffer> output_buffer = output.buffer();

    std::scoped_lock submit_lock(m_submit_mutex);
    std::vector<MetalBufferAccess const *> buffers{&lhs, &rhs, &output};
    std::ranges::sort(buffers, std::less<MetalBufferAccess const *>());
    buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());
    std::vector<std::unique_lock<std::mutex>> access_locks;
    access_locks.reserve(buffers.size());
    for (MetalBufferAccess const * buffer : buffers)
    {
        access_locks.emplace_back(buffer->access_mutex());
        if (buffer->host_exported())
        {
            throw std::runtime_error(
                "Metal GEMM: a host pointer or view has escaped; copy the array to a new Metal buffer first");
        }
        if (buffer->host_access_active_unlocked())
        {
            throw std::runtime_error(
                "Metal GEMM: scoped host access is active; release the host view first");
        }
    }

    std::vector<std::shared_ptr<MetalTask>> predecessors;
    for (MetalBufferAccess const * buffer : buffers)
    {
        std::shared_ptr<MetalTask> const & last_use = buffer->last_use_unlocked();
        if (last_use && std::ranges::find(predecessors, last_use) == predecessors.end())
        {
            predecessors.push_back(last_use);
        }
    }

    size_t const lhs_row_bytes = checked_product(
        static_cast<size_t>(operation.m_lhs.m_leading_dimension), sizeof(float), "lhs row");
    size_t const rhs_row_bytes = checked_product(
        static_cast<size_t>(operation.m_rhs.m_leading_dimension), sizeof(float), "rhs row");
    size_t const output_row_bytes = checked_product(
        static_cast<size_t>(operation.m_output.m_leading_dimension), sizeof(float), "output row");

    MPSMatrixDescriptor * lhs_descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                                 columns:inner_size
                                                                                rowBytes:lhs_row_bytes
                                                                                dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor * rhs_descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:inner_size
                                                                                 columns:columns
                                                                                rowBytes:rhs_row_bytes
                                                                                dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor * output_descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                                    columns:columns
                                                                                   rowBytes:output_row_bytes
                                                                                   dataType:MPSDataTypeFloat32];
    MPSMatrix * lhs_matrix = [[MPSMatrix alloc] initWithBuffer:lhs_buffer
                                                        offset:lhs.resource_offset() + operation.m_lhs.m_byte_offset
                                                    descriptor:lhs_descriptor];
    MPSMatrix * rhs_matrix = [[MPSMatrix alloc] initWithBuffer:rhs_buffer
                                                        offset:rhs.resource_offset() + operation.m_rhs.m_byte_offset
                                                    descriptor:rhs_descriptor];
    MPSMatrix * output_matrix = [[MPSMatrix alloc] initWithBuffer:output_buffer
                                                           offset:output.resource_offset() + operation.m_output.m_byte_offset
                                                       descriptor:output_descriptor];
    if (lhs_matrix == nil || rhs_matrix == nil || output_matrix == nil)
    {
        throw std::runtime_error("Metal GEMM: failed to create MPSMatrix views");
    }

    MPSMatrixMultiplication * kernel = [[MPSMatrixMultiplication alloc]
         initWithDevice:m_device
          transposeLeft:NO
         transposeRight:NO
             resultRows:rows
          resultColumns:columns
        interiorColumns:inner_size
                  alpha:1.0
                   beta:0.0];
    id<MTLCommandBuffer> command = [m_queue commandBuffer];
    if (kernel == nil || command == nil)
    {
        throw std::runtime_error("Metal GEMM: failed to create command resources");
    }

    [kernel encodeToCommandBuffer:command
                       leftMatrix:lhs_matrix
                      rightMatrix:rhs_matrix
                     resultMatrix:output_matrix];
    auto task = std::make_shared<MetalTask>(
        command,
        retain_resources(lhs, rhs, output),
        std::move(predecessors));
    for (MetalBufferAccess const * buffer : buffers)
    {
        buffer->set_last_use_unlocked(task);
    }
    task->attach_completion_handler(task);
    [command commit];
    ++metal_counters().m_submitted_commands;
}

MetalManager::MetalManager()
    : m_impl(std::make_unique<Impl>())
{
}

MetalManager::~MetalManager() = default;

MetalManager & MetalManager::instance()
{
    static MetalManager manager;
    return manager;
}

bool MetalManager::started() const noexcept
{
    return m_impl->started();
}

std::shared_ptr<ConcreteBuffer> MetalManager::allocate(size_t nbytes, size_t alignment) const
{
    @autoreleasepool
    {
        return m_impl->make_buffer(nbytes, alignment);
    }
}

void MetalManager::gemm_async(MetalGemmOperation const & operation)
{
    @autoreleasepool
    {
        m_impl->gemm_async(operation);
    }
}

MetalStatistics MetalManager::statistics() const noexcept
{
    MetalCounters & counters = metal_counters();
    return MetalStatistics{
        .m_allocated_buffers = counters.m_allocated_buffers.load(),
        .m_submitted_commands = counters.m_submitted_commands.load(),
        .m_host_waits = counters.m_host_waits.load(),
        .m_host_exports = counters.m_host_exports.load(),
    };
}

void MetalManager::reset_statistics() noexcept
{
    MetalCounters & counters = metal_counters();
    counters.m_allocated_buffers.store(0);
    counters.m_submitted_commands.store(0);
    counters.m_host_waits.store(0);
    counters.m_host_exports.store(0);
}

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
