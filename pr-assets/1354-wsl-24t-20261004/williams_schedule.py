# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


def _williams_rows(names):
    """Build a Williams schedule for benchmark candidates.

    Each row contains every candidate once. For four candidates:

        A B D C
        B C A D
        C D B A
        D A C B

    Across any prefix of rows, occurrence counts at each position differ by
    at most one. A complete design balances immediate predecessors within
    rows. This reduces bias from cache, thermal, or frequency state left by
    the previous candidate. For two or more candidates, even counts use one
    row per candidate; odd counts require both orientations and twice as many
    rows.

    See https://doi.org/10.1071/CH9490149.
    """

    count = len(names)
    if count < 2:
        return (tuple(names),)

    pattern = []
    for position in range(count):
        if position % 2:
            candidate_index = (position + 1) // 2
        else:
            candidate_index = count - position // 2
        pattern.append(candidate_index % count)

    base_rows = []
    for offset in range(count):
        row = tuple((index + offset) % count for index in pattern)
        base_rows.append(row)

    if count % 2:
        # Both orientations balance predecessors for odd counts. This row
        # traversal also makes adjacent rows meet on the same candidate.
        forward_step = (count + 1) // 2
        reverse_step = count - forward_step
        rows = []
        for offset in range(count):
            row_index = (offset * forward_step) % count
            rows.append(base_rows[row_index])
        for offset in range(1, count + 1):
            row_index = (offset * reverse_step) % count
            rows.append(tuple(reversed(base_rows[row_index])))
    else:
        rows = base_rows

    named_rows = []
    for row in rows:
        named_rows.append(tuple(names[index] for index in row))
    return tuple(named_rows)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
