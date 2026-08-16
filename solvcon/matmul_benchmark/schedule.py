# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Deterministic balanced schedules for interleaved route panels."""


def balanced_orders(routes, panels):
    """Return deterministic route orders balanced across panel positions."""

    routes = tuple(routes)
    if not routes:
        return tuple()
    if len(routes) != len(set(routes)):
        raise ValueError('routes must be unique')
    if len(routes) == 1:
        return tuple(routes for _ in range(panels))

    indices = [0]
    for index in range(1, len(routes)):
        offset = (index + 1) // 2
        indices.append(offset if index % 2 else -offset)
    block = []
    for offset in range(len(routes)):
        order = tuple(
            routes[(index + offset) % len(routes)] for index in indices
        )
        block.append(order)
    if len(routes) % 2:
        reversed_block = [tuple(reversed(order)) for order in block]
        block.extend(reversed_block)
    return tuple(block[index % len(block)] for index in range(panels))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
