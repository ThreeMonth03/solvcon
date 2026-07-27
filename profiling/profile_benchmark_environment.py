# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import os


THREAD_COUNT = os.environ.get('SOLVCON_BENCHMARK_THREADS', '1')
THREAD_VARIABLES = (
    'OPENBLAS_NUM_THREADS',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)
for thread_variable in THREAD_VARIABLES:
    os.environ[thread_variable] = THREAD_COUNT


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
