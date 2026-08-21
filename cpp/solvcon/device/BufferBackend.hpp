#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Backend-neutral allocation contract for device buffers.
 *
 * @ingroup group_core
 */

#include <solvcon/device/BufferDevice.hpp>

#include <cstddef>
#include <memory>

namespace solvcon
{

class ConcreteBuffer;

namespace device
{

class BufferBackend
{
public:
    BufferBackend() = default;
    BufferBackend(BufferBackend const &) = delete;
    BufferBackend(BufferBackend &&) = delete;
    BufferBackend & operator=(BufferBackend const &) = delete;
    BufferBackend & operator=(BufferBackend &&) = delete;
    virtual ~BufferBackend() = default;

    /// Return the storage device implemented by this provider.
    virtual BufferDevice device() const noexcept = 0;
    /// Return true when the provider was compiled into this build.
    virtual bool built() const noexcept = 0;
    /// Return true when the provider can allocate on the current machine.
    virtual bool available() const noexcept = 0;
    /// Allocate one buffer using this provider.
    virtual std::shared_ptr<ConcreteBuffer> allocate(size_t nbytes, size_t alignment) const = 0;
}; /* end class BufferBackend */

/// Return the provider for a storage device, including an unavailable provider.
BufferBackend const & buffer_backend(BufferDevice device);

/// Allocate through the provider selected by device.
std::shared_ptr<ConcreteBuffer> allocate_buffer(size_t nbytes, size_t alignment, BufferDevice device);

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
