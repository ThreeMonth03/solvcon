import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


DEST = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()
    families = ('single', 'batch', 'broadcast')
    if not args.small:
        families += ('large',)
    provenance = json.loads((DEST / 'provenance.json').read_text())
    root = DEST.parents[2]
    extension = root / 'solvcon/_solvcon.cpython-314-x86_64-linux-gnu.so'
    assert hashlib.sha256(extension.read_bytes()).hexdigest() == (
        provenance['extension_sha256'])
    for name in ('profile_pr1354.py', 'williams_schedule.py'):
        assert hashlib.sha256((DEST / name).read_bytes()).hexdigest() == (
            provenance[name]), name
    records = []
    paths = sorted(DEST.glob('*-r[012].jsonl'))
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows[0]['family'] not in families:
            continue
        assert len(rows) == 20, path
        keys = [(row['round'], row['method']) for row in rows]
        expected = [(i, name) for i in range(10)
                    for name in ('numpy', 'solvcon')]
        assert set(keys) == set(expected) and len(set(keys)) == 20
        assert len({row['inner_loops'] for row in rows}) == 1
        for row in rows:
            assert row['blas']['threads'] == 24
            assert row['affinity'] == list(range(24))
            assert row['numpy_version'] == '2.5.1'
            assert row['sha'] == provenance['commit']
            assert row['rounds'] == 10 and row['warmup_rounds'] == 2
            assert row['samples_per_round'] == 1
            assert row['method_order'] == row['williams_rows'][
                row['round'] % 2]
            assert row['sample_ns'] > 0
            assert math.isclose(row['sample_ns'],
                                row['batch_elapsed_ns'] / row['inner_loops'])
            assert math.isclose(
                row['cpu_wall_ratio'],
                row['cpu_elapsed_ns'] / row['batch_elapsed_ns'])
            validation = row['validation']
            assert validation['relative_frobenius_error'] <= (
                validation['relative_frobenius_tolerance'])
            s = row['side']
            family = row['family']
            if family in ('single', 'large'):
                shapes = [(s, s)] * 3
                strides = ([(2 * s, 2), (s, 1), (s, 1)]
                           if family == 'single' else [(s, 1)] * 3)
            else:
                count = 1 if family == 'broadcast' else 10
                shapes = [(count, s, s), (10, s, s), (10, s, s)]
                strides = [(2 * s * s, 2 * s, 2)] * 2 + [(s * s, s, 1)]
            arrays = [row['lhs'], row['rhs'], validation['output']]
            for array, shape, stride in zip(arrays, shapes, strides):
                assert array['shape'] == list(shape)
                assert array['element_strides'] == list(stride)
                assert array['dtype'] == row['dtype']
        counts = collections.Counter(
            row['method_order'][0] for row in rows
            if row['method'] == 'numpy')
        assert counts == {'numpy': 5, 'solvcon': 5}
        records.extend(rows)
    cases = collections.defaultdict(list)
    for row in records:
        cases[row['case']].append(row)
    expected_cases = 48 if args.small else 54
    assert len(cases) == expected_cases
    for case, rows in cases.items():
        assert len(rows) == 60, case
        assert {row['replicate'] for row in rows} == {0, 1, 2}
        assert len({row['seed'] for row in rows}) == 1
    assert len({row['blas']['path'] for row in records}) == 1
    name = 'audit-small.txt' if args.small else 'audit.txt'
    text = (f'PASS: {len(cases)} cases, {len(cases) * 3} independent '
            f'process runs, {len(records)} timed blocks. '
            'All outputs, shapes, strides, shared repetitions, 24-thread '
            'runtime limits, CPU affinities, Williams orders, timing '
            'arithmetic, and source hashes verified.\n')
    (DEST / name).write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
