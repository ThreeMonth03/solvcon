# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
PACKAGE_TUNING_SOURCES = tuple(sorted((
    "contrib/matmul_dispatch/__init__.py",
    "contrib/matmul_dispatch/cli.py",
    "contrib/matmul_dispatch/codegen.py",
    "contrib/matmul_dispatch/collect.py",
    "contrib/matmul_dispatch/data.py",
    "contrib/matmul_dispatch/environment.py",
    "contrib/matmul_dispatch/measurement.py",
    "contrib/matmul_dispatch/model.py",
    "contrib/matmul_dispatch/policy.py",
    "contrib/matmul_dispatch/sampling.py",
)))
TUNING_SOURCES = (
    "contrib/tune_matmul_dispatch.py",
    *PACKAGE_TUNING_SOURCES,
    "cpp/solvcon/buffer/matmul.cpp",
    "cpp/solvcon/buffer/matmul.hpp",
    "cpp/solvcon/buffer/pymod/wrap_SimpleArray.cpp",
    "cpp/solvcon/buffer/pymod/wrap_SimpleArray.hpp",
)


def _repository_root():
    return Path(__file__).resolve().parents[2]


def _git_value(*arguments):
    repository = _repository_root()
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_diff_hash():
    repository = _repository_root()
    result = subprocess.run(
        ("git", "-C", str(repository), "diff", "--binary", "HEAD"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return "unknown"
    return hashlib.sha256(result.stdout).hexdigest()


def _source_hashes():
    repository = _repository_root()
    result = {}
    for name in TUNING_SOURCES:
        path = repository / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _cpu_name():
    name = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return name or "unknown"
    with cpuinfo.open(encoding="utf-8") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            if separator and key.strip() in ("model name", "hardware"):
                model = value.strip()
                if model:
                    return model
    return name or "unknown"


def make_environment(backend=None):
    import numpy as np

    try:
        sklearn_version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        sklearn_version = "not-installed"
    try:
        import threadpoolctl
        threadpools = threadpoolctl.threadpool_info()
    except ImportError:
        threadpools = []
    blas = np.show_config(mode="dicts").get(
        "Build Dependencies", {}).get("blas", {})
    environment = {
        "backend": backend,
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        },
        "cpu": _cpu_name(),
        "git_diff_sha256": _git_diff_hash(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "thread_environment": {
            name: os.environ.get(name)
            for name in THREAD_VARIABLES
        },
        "threadpools": threadpools,
        "tuning_source_sha256": _source_hashes(),
    }
    payload = json.dumps(environment, sort_keys=True).encode("utf-8")
    environment["fingerprint"] = hashlib.sha256(payload).hexdigest()
    return environment


def environment_with_backend(environment, backend):
    result = dict(environment)
    result["backend"] = backend
    result.pop("fingerprint", None)
    payload = json.dumps(result, sort_keys=True).encode("utf-8")
    result["fingerprint"] = hashlib.sha256(payload).hexdigest()
    return result


def validate_tuning_sources(records):
    current = _source_hashes()
    for record in records:
        environment = record.get("environment")
        recorded = (environment.get("tuning_source_sha256")
                    if isinstance(environment, Mapping) else None)
        if recorded != current:
            sample_id = record.get("sample_id", "unknown sample")
            raise ValueError(
                f"{sample_id}: tuning sources do not match current "
                "checkout")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
