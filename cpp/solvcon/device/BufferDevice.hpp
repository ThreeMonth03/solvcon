#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Device identity shared by buffer storage backends.
 *
 * @ingroup group_core
 */

#include <cstdint>
#include <optional>
#include <string_view>

namespace solvcon
{

/// Storage device selected for an owned ConcreteBuffer.
enum class BufferDevice : std::uint8_t
{
    Cpu,
    Metal,
}; /* end enum class BufferDevice */

constexpr std::string_view buffer_device_name(BufferDevice device) noexcept
{
    switch (device)
    {
    case BufferDevice::Cpu:
        return "cpu";
    case BufferDevice::Metal:
        return "metal";
    }
    return "unknown";
}

constexpr std::string_view buffer_device_label(BufferDevice device) noexcept
{
    switch (device)
    {
    case BufferDevice::Cpu:
        return "CPU";
    case BufferDevice::Metal:
        return "Metal";
    }
    return "unknown";
}

constexpr std::optional<BufferDevice> buffer_device_from_name(std::string_view name) noexcept
{
    if (name == "cpu")
    {
        return BufferDevice::Cpu;
    }
    if (name == "metal")
    {
        return BufferDevice::Metal;
    }
    return std::nullopt;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
