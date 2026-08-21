/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/device/BufferBackend.hpp>

namespace solvcon
{

std::shared_ptr<ConcreteBuffer> ConcreteBuffer::construct(
    size_t nbytes,
    size_t alignment,
    BufferDevice target_device)
{
    return device::allocate_buffer(nbytes, alignment, target_device);
}

std::shared_ptr<ConcreteBuffer> ConcreteBuffer::clone_to(BufferDevice target_device) const
{
    std::shared_ptr<ConcreteBuffer> ret = construct(nbytes(), m_alignment, target_device);
    ret->copy_from(*this);
    return ret;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
