# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import math

from .collect import (
    DEFAULT_RESULTS,
    REFINEMENT_GAP,
    collect_records,
)
from .policy import DEFAULT_INCLUDE, DEFAULT_REPORT, fit_policy


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def fraction(value):
    result = float(value)
    if not 0 < result < 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return result


def make_parser():
    parser = argparse.ArgumentParser(
        description="Measure and fit an experimental GEMM dispatch policy.")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser(
        "collect", help="collect forced-route timings as JSONL")
    collect.add_argument("--output", default=DEFAULT_RESULTS)
    collect.add_argument("--budget-seconds", type=positive_float,
                         default=300.0)
    collect.add_argument("--max-cases", type=positive_int, default=256)
    collect.add_argument("--max-dimension", type=positive_int, default=4096)
    collect.add_argument("--max-bytes", type=positive_int,
                         default=256 * 1024 * 1024)
    collect.add_argument("--dtypes", nargs="+",
                         choices=("float32", "float64"),
                         default=("float32",))
    collect.add_argument("--layouts", nargs="+",
                         choices=("cc", "cf", "fc", "ff"),
                         default=("cc",))
    collect.add_argument("--seed", type=int, default=1208)
    collect.add_argument("--warmups", type=positive_int, default=1)
    collect.add_argument("--min-samples", type=positive_int, default=3)
    collect.add_argument("--max-samples", type=positive_int, default=9)
    collect.add_argument("--tie-gap", type=fraction, default=0.05)
    collect.add_argument("--refinement-gap", type=fraction,
                         default=REFINEMENT_GAP)
    collect.add_argument("--loop-work-limit", type=positive_int,
                         default=32 * 1024 * 1024)
    collect.add_argument("--target-sample-ms", type=positive_float,
                         default=5.0)
    collect.add_argument("--max-inner-repetitions", type=positive_int,
                         default=10_000)
    collect.add_argument("--overwrite", action="store_true")

    fit = commands.add_parser(
        "fit", help="fit and assess a shallow dispatch tree")
    fit.add_argument("--input", default=DEFAULT_RESULTS)
    fit.add_argument("--report", default=DEFAULT_REPORT)
    fit.add_argument("--generated-include", default=DEFAULT_INCLUDE)
    fit.add_argument("--holdout-fraction", type=fraction, default=0.25,
                     help=argparse.SUPPRESS)
    fit.add_argument("--max-depth", type=positive_int, default=5)
    fit.add_argument("--min-samples-leaf", type=positive_int, default=6)
    fit.add_argument("--seed", type=int, default=1208)
    fit.add_argument("--min-holdout-samples", type=positive_int,
                     default=20)
    fit.add_argument("--min-oracle-headroom", type=positive_float,
                     default=1.05)
    fit.add_argument("--min-speedup", type=positive_float, default=1.03)
    fit.add_argument("--min-stump-speedup", type=positive_float,
                     default=1.005)
    fit.add_argument("--max-geomean-regret", type=positive_float,
                     default=1.03)
    fit.add_argument("--max-p95-regret", type=positive_float, default=1.05)
    fit.add_argument("--max-regression", type=positive_float, default=1.10)
    fit.add_argument("--min-captured-gap", type=fraction, default=0.70)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "collect":
        if args.max_samples < args.min_samples:
            raise ValueError("max samples must not be less than min samples")
        result = collect_records(args)
    else:
        result = fit_policy(args)
    print(json.dumps(result, indent=2, sort_keys=True))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
