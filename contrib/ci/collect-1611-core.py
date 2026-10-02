#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile


def backtrace(core):
    report = core.with_name(core.name + '.backtrace.txt')
    command = [
        'gdb', '--batch', '--nx',
        '-iex', 'set debuginfod enabled off',
        '-ex', 'set pagination off',
        '-ex', 'thread apply all bt 50',
        '-ex', 'info registers',
        '-ex', 'info sharedlibrary',
        '-ex', 'info proc mappings',
        sys.executable, str(core),
    ]
    with report.open('w') as output:
        result = subprocess.run(command, stdout=output,
                                stderr=subprocess.STDOUT, timeout=120)
    result.check_returncode()
    return report


def preflight(directory):
    if (os.environ.get('GITHUB_ACTIONS') != 'true'
            or os.environ.get('GITHUB_REPOSITORY')
            != 'ThreeMonth03/solvcon'):
        raise RuntimeError('The abort preflight requires the fork CI runner')
    before = set(directory.glob('core.*'))
    result = subprocess.run([sys.executable, '-c', 'import os; os.abort()'])
    if result.returncode != -signal.SIGABRT:
        raise RuntimeError(f'Unexpected abort status: {result.returncode}')
    cores = set(directory.glob('core.*')) - before
    if len(cores) != 1:
        raise RuntimeError(f'Expected one preflight core, found {len(cores)}')
    preflight_dir = directory / 'preflight'
    preflight_dir.mkdir()
    core = cores.pop()
    core = core.rename(preflight_dir / core.name)
    report = backtrace(core)
    if 'SIGABRT' not in report.read_text():
        raise RuntimeError('GDB did not identify the preflight abort')
    core.unlink()
    print('Core capture and postmortem GDB preflight passed', flush=True)


def collect(directory):
    cores = sorted(directory.glob('core.*'))
    cores = [core for core in cores if not core.name.endswith('.txt')]
    if not cores:
        print('No test core dumps found', flush=True)
        return
    paths = {Path(sys.executable)}
    paths.update(Path.cwd().glob('_solvcon*.so'))
    paths.update(Path.cwd().glob('solvcon/_solvcon*.so'))
    failures = []
    for core in cores:
        try:
            report = backtrace(core)
        except (subprocess.SubprocessError, OSError) as error:
            failures.append(f'{core.name}: {error}')
            continue
        for line in report.read_text().splitlines():
            fields = line.split()
            if (len(fields) >= 2 and fields[0].startswith('0x')
                    and fields[1].startswith('0x')
                    and fields[-1].startswith('/')):
                paths.add(Path(fields[-1]))
    files = sorted(path for path in paths if path.is_file())
    with (directory / 'binary-paths.txt').open('w') as output:
        for path in files:
            output.write(f'{path}\n')
    with tarfile.open(directory / 'matching-binaries.tar.gz', 'w:gz',
                      dereference=True, compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=str(path).lstrip('/'), recursive=False)
    if failures:
        raise RuntimeError('\n'.join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'collect'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.mode == 'preflight':
        preflight(args.directory)
    else:
        collect(args.directory)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
