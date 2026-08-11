/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Strassen.hpp>

namespace solvcon
{

namespace detail
{

namespace strassen
{

TransformTeam::TransformTeam()
{
    try
    {
        for (size_t lane = 0; lane + 1 < DEPTH1_TRANSFORM_LANE_COUNT; ++lane)
        {
            m_workers[lane] = std::jthread(&TransformTeam::worker_loop, this, lane);
        }
    }
    catch (...)
    {
        stop();
        throw;
    }
}

TransformTeam::~TransformTeam()
{
    stop();
}

void TransformTeam::run_lane(size_t lane) noexcept
{
    try
    {
        ssize_t const first = m_rows * static_cast<ssize_t>(lane) /
                              static_cast<ssize_t>(DEPTH1_TRANSFORM_LANE_COUNT);
        ssize_t const last = m_rows * static_cast<ssize_t>(lane + 1) /
                             static_cast<ssize_t>(DEPTH1_TRANSFORM_LANE_COUNT);
        m_invoke(m_function, first, last);
    }
    catch (...)
    {
        m_errors[lane] = std::current_exception();
    }
}

void TransformTeam::worker_loop(size_t lane)
{
    size_t generation = 0;
    while (true)
    {
        std::unique_lock lock(m_mutex);
        m_start.wait(lock, [this, generation]
                     { return m_stopping || generation != m_generation; });
        if (m_stopping)
        {
            return;
        }
        generation = m_generation;
        lock.unlock();

        run_lane(lane);

        lock.lock();
        ++m_completed_workers;
        if (m_completed_workers == DEPTH1_TRANSFORM_LANE_COUNT - 1)
        {
            m_done.notify_one();
        }
    }
}

void TransformTeam::stop() noexcept
{
    {
        std::lock_guard lock(m_mutex);
        m_stopping = true;
    }
    m_start.notify_all();
}

} /* end namespace strassen */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
