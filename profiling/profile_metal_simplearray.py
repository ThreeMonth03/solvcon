# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

import solvcon as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare resident SimpleArray Metal and CPU matmul chains")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sides", nargs="+", type=int,
                        default=(512, 1024, 2048))
    parser.add_argument("--depths", nargs="+", type=int, default=(1, 4))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output")
    return parser.parse_args()


def system_output(command):
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return str(exc)
    return result.stdout.strip() or result.stderr.strip()


def workspace_sha256():
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], check=False,
        capture_output=True)
    untracked = system_output(
        ["git", "ls-files", "--others", "--exclude-standard"])
    paths = [Path(name) for name in untracked.splitlines()]
    if not result.stdout and not paths:
        return None
    digest = hashlib.sha256(result.stdout)
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def busy_processes():
    output = system_output(["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"])
    return output.splitlines()[:16]


def display_devices():
    output = system_output(
        ["system_profiler", "SPDisplaysDataType", "-json"])
    try:
        records = json.loads(output)["SPDisplaysDataType"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return output
    keys = (
        "_name",
        "sppci_model",
        "sppci_vendor",
        "sppci_bus",
        "sppci_cores",
        "spdisplays_mtlgpufamilysupport",
    )
    return [{key: record[key] for key in keys if key in record}
            for record in records]


def metadata(args):
    thread_variables = (
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    module_path = Path(sc.core._impl.__file__).resolve()
    return {
        "kind": "metadata",
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": system_output(["sysctl", "-n", "hw.model"]),
        "memory_bytes": system_output(["sysctl", "-n", "hw.memsize"]),
        "power": system_output(["pmset", "-g", "custom"]),
        "power_source": system_output(["pmset", "-g", "ps"]),
        "thermal": system_output(["pmset", "-g", "therm"]),
        "display_devices": display_devices(),
        "git_revision": system_output(["git", "rev-parse", "HEAD"]),
        "git_status": system_output(["git", "status", "--porcelain"]),
        "workspace_sha256": workspace_sha256(),
        "build_type_environment": os.environ.get(
            "CMAKE_BUILD_TYPE", "unknown"),
        "module_path": str(module_path),
        "module_sha256": file_sha256(module_path),
        "linked_libraries": system_output(["otool", "-L", module_path]),
        "cmake": system_output(["cmake", "--version"]),
        "compiler": system_output(["clang++", "--version"]),
        "sdk": system_output(["xcrun", "--show-sdk-version"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "busy_processes": busy_processes(),
        "thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
        "sides": args.sides,
        "depths": args.depths,
        "warmups": args.warmups,
        "samples": args.samples,
        "rounds": args.rounds,
    }


def completion_metadata():
    return {
        "kind": "completion",
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "power_source": system_output(["pmset", "-g", "ps"]),
        "thermal": system_output(["pmset", "-g", "therm"]),
        "busy_processes": busy_processes(),
    }


def make_array(values, device):
    return sc.SimpleArrayFloat32(array=values, device=device)


def run_cpu(lhs, weight, depth):
    state = lhs
    for _ in range(depth):
        state = state.matmul(weight)
    return state


def run_metal(lhs, weight, depth):
    state = lhs
    for _ in range(depth):
        state = state.matmul_metal(weight)
    state.wait()
    return state


def elapsed_ms(function):
    begin = time.perf_counter_ns()
    result = function()
    del result
    return (time.perf_counter_ns() - begin) * 1e-6


def check_result(cpu_lhs, cpu_weight, metal_lhs, metal_weight, depth):
    reference = run_cpu(cpu_lhs, cpu_weight, depth).ndarray.copy()
    metal_result = run_metal(
        metal_lhs, metal_weight, depth).cpu().ndarray.copy()
    error = np.asarray(metal_result, dtype="float64") - reference
    denominator = np.linalg.norm(reference.ravel().astype("float64"))
    relative_l2 = np.linalg.norm(error.ravel()) / denominator
    if not np.isfinite(relative_l2) or relative_l2 > 2e-3:
        raise RuntimeError(
            f"Metal result failed validation: relative_l2={relative_l2}")
    return float(np.max(np.abs(error))), float(relative_l2)


def profile_case(side, depth, args):
    rng = np.random.default_rng(20260816 + side + depth)
    scale = np.float32(0.25 / np.sqrt(side))
    lhs_values = rng.standard_normal((side, side), dtype="float32")
    weight_values = (
        rng.standard_normal((side, side), dtype="float32") * scale)
    cpu_lhs = make_array(lhs_values, "cpu")
    cpu_weight = make_array(weight_values, "cpu")
    metal_lhs = make_array(lhs_values, "metal")
    metal_weight = make_array(weight_values, "metal")

    max_abs, relative_l2 = check_result(
        cpu_lhs, cpu_weight, metal_lhs, metal_weight, depth)
    methods = {
        "cpu": lambda: run_cpu(cpu_lhs, cpu_weight, depth),
        "metal": lambda: run_metal(metal_lhs, metal_weight, depth),
    }
    timings = {name: [] for name in methods}
    for round_index in range(args.rounds):
        order = ("cpu", "metal")
        if round_index % 2:
            order = tuple(reversed(order))
        for name in order:
            for _ in range(args.warmups):
                methods[name]()
            for _ in range(args.samples):
                timings[name].append(elapsed_ms(methods[name]))

    cpu_median = statistics.median(timings["cpu"])
    metal_median = statistics.median(timings["metal"])
    return {
        "kind": "case",
        "side": side,
        "depth": depth,
        "cpu_ms": timings["cpu"],
        "metal_ms": timings["metal"],
        "cpu_median_ms": cpu_median,
        "metal_median_ms": metal_median,
        "metal_over_cpu": metal_median / cpu_median,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
    }


def write_record(output, record):
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()


def main():
    args = parse_args()
    if not args.run:
        print("Metal SimpleArray profile is opt-in; pass --run and --output")
        return
    if args.output is None:
        raise ValueError("--output is required with --run")
    if args.rounds < 2 or args.rounds % 2:
        raise ValueError("--rounds must be a positive even number")
    if not sc.METAL_BUILT or not sc.metal_running():
        raise RuntimeError("Metal support is unavailable")

    with open(args.output, "x", encoding="utf-8") as output:
        write_record(output, metadata(args))
        for side in args.sides:
            for depth in args.depths:
                record = profile_case(side, depth, args)
                write_record(output, record)
                print(
                    f"S={side} depth={depth}: "
                    f"CPU={record['cpu_median_ms']:.3f} ms, "
                    f"Metal={record['metal_median_ms']:.3f} ms, "
                    f"ratio={record['metal_over_cpu']:.3f}")
        write_record(output, completion_metadata())


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
