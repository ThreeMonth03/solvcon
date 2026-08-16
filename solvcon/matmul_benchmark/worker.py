# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Headless JSON-lines worker for the Qt benchmark controller."""

import argparse
import ctypes
import dataclasses
import json
import os
import pathlib
import signal
import sys

from . import artifact
from . import collection
from . import collector
from . import schema


PR_SET_PDEATHSIG = 1


def _emit(event):
    print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def _read_request(path):
    if path:
        with pathlib.Path(path).open(encoding='utf8') as stream:
            return json.load(stream)
    return json.load(sys.stdin)


def _install_parent_death_signal():
    if sys.platform != 'linux':
        return True
    parent_pid = os.getppid()
    if parent_pid == 1:
        return False
    try:
        library = ctypes.CDLL(None, use_errno=True)
        result = library.prctl(
            PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (AttributeError, OSError):
        return True
    if result != 0:
        return True
    return os.getppid() == parent_pid


def build_parser():
    parser = argparse.ArgumentParser(
        description='collect an isolated matmul benchmark artifact')
    parser.add_argument(
        '--request', help='request JSON path; default reads stdin')
    parser.add_argument(
        '--output', help='override the request artifact output path')
    parser.add_argument(
        '--json-lines', action='store_true',
        help='emit the stable JSON-lines worker protocol')
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        if (arguments.json_lines
                and not _install_parent_death_signal()):
            raise RuntimeError(
                'benchmark controller exited before worker startup')
        payload = _read_request(arguments.request)
        if payload.get('schema_kind') == collection.PLAN_KIND:
            request = collection.CollectionPlan.from_dict(payload)
        else:
            request = schema.BenchmarkRequest.from_dict(payload)
        output = arguments.output or request.output_path
        if output is None:
            request_id = (request.plan_id
                          if isinstance(request, collection.CollectionPlan)
                          else request.request_id)
            output = f'matmul-benchmark-{request_id}.json'
        checkpoint_path = None
        if isinstance(request, collection.CollectionPlan):
            if request.target_duration is not None:
                request = dataclasses.replace(
                    request, output_path=output)
                checkpoint_path = collector.duration_checkpoint_path(
                    output, request.sha256())
            result = collector.collect_plan(
                request, progress=_emit,
                checkpoint_path=checkpoint_path)
            identifier = result['collection_id']
        else:
            result = collector.collect(request, progress=_emit)
            identifier = result['artifact_id']
        artifact_path = collector._run_activity(
            lambda: artifact.write_artifact(result, output),
            'artifact_write', 'artifact', 1, _emit, None)
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError:
                pass
        _emit({
            'type': 'result',
            'artifact_id': identifier,
            'artifact_path': str(artifact_path),
        })
        return 0
    except Exception as exc:
        _emit({
            'type': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        })
        return 1


if __name__ == '__main__':
    sys.exit(main())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
