#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Reference-counted contiguous memory buffer.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/BufferBase.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace solvcon
{

template <typename T>
class SimpleArray;

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

namespace detail
{

// Take the remover and deleter classes outside ConcreteBuffer to work around
// https://bugzilla.redhat.com/show_bug.cgi?id=1569374

/**
 * The base class of memory deallocator for ConcreteBuffer.  When the object
 * exists in ConcreteBufferDataDeleter (the unique_ptr deleter), the deleter
 * calls it to release the memory of the ConcreteBuffer data buffer.
 */
struct ConcreteBufferRemover
{

    ConcreteBufferRemover() = default;
    ConcreteBufferRemover(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover(ConcreteBufferRemover &&) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover &&) = default;
    virtual ~ConcreteBufferRemover() = default;

    static void deallocate_memory(int8_t * p, size_t alignment)
    {
        if (alignment > 0) // NOLINT(bugprone-branch-clone)
        {
#ifdef _WIN32
            _aligned_free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#else
            std::free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
        }
        else
        {
            std::free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
        }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    virtual void operator()(int8_t * p, size_t alignment) const
    {
        deallocate_memory(p, alignment);
    }

    virtual BufferDevice device() const noexcept { return BufferDevice::Cpu; }
    virtual void begin_internal_host_access() const {}
    virtual void end_internal_host_access() const noexcept {}
    virtual void prepare_host_access() const {}
    virtual void wait() const {}
    virtual bool ready() const { return true; }
    virtual bool host_exported() const noexcept { return false; }

}; /* end struct ConcreteBufferRemover */

struct ConcreteBufferNoRemove : public ConcreteBufferRemover
{

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *, size_t) const override {}

}; /* end struct ConcreteBufferNoRemove */

struct ConcreteBufferDataDeleter
{

    using remover_type = ConcreteBufferRemover;

    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter const &) = delete;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter const &) = delete;

    ConcreteBufferDataDeleter() = default;
    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter &&) = default;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter &&) = default;
    ~ConcreteBufferDataDeleter() = default;
    explicit ConcreteBufferDataDeleter(std::unique_ptr<remover_type> && remover_in, size_t alignment_in = 0)
        : remover(std::move(remover_in))
        , alignment(alignment_in)
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t * p) const
    {
        if (!remover)
        {
            remover_type::deallocate_memory(p, alignment);
        }
        else
        {
            (*remover)(p, alignment);
        }
    }

    std::unique_ptr<remover_type> remover{nullptr};
    size_t alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.

}; /* end struct ConcreteBufferDataDeleter */

} /* end namespace detail */

/**
 * Untyped and unresizeable memory buffer for contiguous data storage.
 *
 * @ingroup group_core
 */
class ConcreteBuffer
    : public std::enable_shared_from_this<ConcreteBuffer>
    , public BufferBase<ConcreteBuffer, true>
{

private:

    struct ctor_passkey
    {
    }; /* end struct ctor_passkey */

    using data_deleter_type = detail::ConcreteBufferDataDeleter;

public:

    using remover_type = detail::ConcreteBufferRemover;
    using size_type = std::size_t;

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, alignment, ctor_passkey());
    }

    /// Allocate an owned buffer on the requested device.
    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, size_t alignment, BufferDevice device)
    {
        if (device == BufferDevice::Cpu)
        {
            return construct(nbytes, alignment);
        }
#ifdef SOLVCON_METAL
        return construct_metal(nbytes, alignment);
#else
        throw std::runtime_error("ConcreteBuffer: Metal support is not built");
#endif
    }

    /*
     * This factory method is dangerous since the data pointer passed in will
     * not be owned by the ConcreteBuffer created.  It is an error if the
     * number of bytes of the externally owned buffer doesn't match the value
     * passed in (but we cannot know here).
     */
    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, data, std::move(remover), alignment, ctor_passkey());
    }

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, void * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return construct(nbytes, static_cast<int8_t *>(data), std::move(remover), alignment);
    }

    /// Construct an empty ConcreteBuffer with no data and no alignment.
    static std::shared_ptr<ConcreteBuffer> construct() { return construct(0, 0); }

    std::shared_ptr<ConcreteBuffer> clone() const { return clone_to(device()); }

    /// Deep-copy the bytes into storage owned by the requested device.
    std::shared_ptr<ConcreteBuffer> clone_to(BufferDevice target_device) const
    {
        std::shared_ptr<ConcreteBuffer> ret = construct(nbytes(), m_alignment, target_device);
        ret->copy_from(*this);
        return ret;
    }

    /**
     * @param[in] nbytes
     *      Size of the memory buffer in bytes.
     * @param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    ConcreteBuffer(size_t nbytes, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer, true>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(allocate(nbytes, m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

    /**
     * @param[in] nbytes
     *      Size of the memory buffer in bytes.
     * @param[in] data
     *      Pointer to the memory buffer that is not supposed to be owned by
     *      this ConcreteBuffer.
     * @param[in] remover
     *      The memory deallocator for the unowned data buffer passed in.
     * @param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    // NOLINTNEXTLINE(readability-non-const-parameter)
    ConcreteBuffer(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer, true>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(data, data_deleter_type(std::move(remover), m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

    ~ConcreteBuffer() = default;

    ConcreteBuffer() = delete;
    ConcreteBuffer(ConcreteBuffer &&) = delete;
    ConcreteBuffer & operator=(ConcreteBuffer &&) = delete;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#endif
    // Avoid enabled_shared_from_this copy constructor
    // NOLINTNEXTLINE(bugprone-copy-constructor-init)
    ConcreteBuffer(ConcreteBuffer const & other)
        : BufferBase<ConcreteBuffer, true>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(other.m_nbytes)
        , m_alignment(other.m_alignment)
        , m_data(allocate(other.m_nbytes, other.m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
        if (size() != other.size())
        {
            throw std::out_of_range("Buffer size mismatch");
        }
        copy_from(other);
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    ConcreteBuffer & operator=(ConcreteBuffer const & other)
    {
        if (this != &other)
        {
            if (size() != other.size())
            {
                throw std::out_of_range("Buffer size mismatch");
            }
            copy_from(other);
        }
        return *this;
    }

    bool has_remover() const noexcept { return static_cast<bool>(m_data.get_deleter().remover); }
    remover_type const & get_remover() const { return *m_data.get_deleter().remover; }
    remover_type & get_remover() { return *m_data.get_deleter().remover; }

    size_type alignment() const noexcept { return m_alignment; }
    /// Return the device that owns this storage.
    BufferDevice device() const noexcept
    {
        return has_remover() ? m_data.get_deleter().remover->device() : BufferDevice::Cpu;
    }
    /// Wait for the last asynchronous use of this storage.
    void wait() const
    {
        if (has_remover())
        {
            m_data.get_deleter().remover->wait();
        }
    }
    /// Return true when the last asynchronous use has completed.
    bool ready() const
    {
        return !has_remover() || m_data.get_deleter().remover->ready();
    }
    /// Return true after an unrestricted host pointer has escaped.
    bool host_exported() const noexcept
    {
        return has_remover() && m_data.get_deleter().remover->host_exported();
    }

    void prepare_buffer_host_access() const
    {
        if (has_remover())
        {
            m_data.get_deleter().remover->prepare_host_access();
        }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    using unique_ptr_type = std::unique_ptr<int8_t, data_deleter_type>;

    static constexpr const char * name() { return "ConcreteBuffer"; }

private:
#ifdef SOLVCON_METAL
    static std::shared_ptr<ConcreteBuffer> construct_metal(size_t nbytes, size_t alignment);
#endif

    template <typename T>
    T const * data_unchecked() const noexcept
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T const *>(m_begin);
    }

    template <typename T>
    T * data_unchecked() noexcept
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T *>(m_begin);
    }

    template <typename T>
    friend class SimpleArray;

    class InternalHostAccessGuard
    {
    public:
        explicit InternalHostAccessGuard(remover_type const * remover);

        InternalHostAccessGuard(InternalHostAccessGuard const &) = delete;
        InternalHostAccessGuard & operator=(InternalHostAccessGuard const &) = delete;
        InternalHostAccessGuard(InternalHostAccessGuard &&) = delete;
        InternalHostAccessGuard & operator=(InternalHostAccessGuard &&) = delete;

        ~InternalHostAccessGuard();

    private:
        remover_type const * m_remover;
    }; /* end class InternalHostAccessGuard */

    remover_type const * remover() const noexcept { return has_remover() ? m_data.get_deleter().remover.get() : nullptr; }

    void copy_from(ConcreteBuffer const & other);

    static unique_ptr_type allocate(size_t nbytes, size_t alignment)
    {
        unique_ptr_type ret(nullptr, data_deleter_type());
        if (0 != nbytes)
        {
            void * ptr = nullptr;
            if (alignment > 0)
            {
                validate_size_alignment(nbytes, alignment, "ConcreteBuffer::allocate");
#ifdef _WIN32
                ptr = _aligned_malloc(nbytes, alignment); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#else
                ptr = std::aligned_alloc(alignment, nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
            }
            else
            {
                ptr = std::malloc(nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
            }
            if (!ptr)
            {
                throw std::bad_alloc();
            }
            ret = unique_ptr_type(static_cast<int8_t *>(ptr), data_deleter_type(nullptr, alignment));
        }
        return ret;
    }

    size_t m_nbytes;
    size_t m_alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.
    unique_ptr_type m_data;
}; /* end class ConcreteBuffer */

inline ConcreteBuffer::InternalHostAccessGuard::InternalHostAccessGuard(remover_type const * remover)
    : m_remover(remover)
{
    if (m_remover != nullptr)
    {
        m_remover->begin_internal_host_access();
    }
}

inline ConcreteBuffer::InternalHostAccessGuard::~InternalHostAccessGuard()
{
    if (m_remover != nullptr)
    {
        m_remover->end_internal_host_access();
    }
}

inline void ConcreteBuffer::copy_from(ConcreteBuffer const & other)
{
    std::array<remover_type const *, 2> removers{remover(), other.remover()};
    std::ranges::sort(removers, std::less<>());

    std::optional<InternalHostAccessGuard> first;
    std::optional<InternalHostAccessGuard> second;
    if (removers[0] != nullptr)
    {
        first.emplace(removers[0]);
    }
    if (removers[1] != nullptr && removers[1] != removers[0])
    {
        second.emplace(removers[1]);
    }
    std::copy_n(other.data_unchecked<int8_t>(), size(), data_unchecked<int8_t>());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
