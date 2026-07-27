# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib
import statistics

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402


REFERENCE_COLOR = '#777777'
SOLVCON_COLOR = '#0072B2'


def load_results(path):
    return json.loads(path.read_text(encoding='utf-8'))['results']


def split_layout(row):
    parts = row['layout'].split('/')
    fields = {}
    for part in parts[1:]:
        key, value = part.split('=', 1)
        fields[key] = int(value)
    return parts[0], fields


def ratio(row):
    return row['numpy_over_planned']


def median(values):
    return statistics.median(values)


def finish_figure(figure, output):
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)


def add_reference(axis):
    axis.axhline(
        1.0,
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle='--',
    )
    axis.grid(True, axis='y', alpha=0.2)


def render_core(rows, output):
    dtypes = ('float32', 'float64')
    topologies = ('2d-2d', 'nd-nd')
    layouts = (
        'c',
        'f',
        'negative-core',
        'step2-core',
    )
    labels = {
        'c': 'C',
        'f': 'F',
        'negative-core': 'negative core',
        'step2-core': 'step-2 core',
    }
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row_number, dtype in enumerate(dtypes):
        for column, topology in enumerate(topologies):
            axis = axes[row_number][column]
            for layout in layouts:
                selected = [
                    row for row in rows
                    if row['dtype'] == dtype
                    and row['topology'] == topology
                    and row['layout'] == layout
                ]
                selected.sort(key=lambda row: row['lhs_shape'][-1])
                axis.plot(
                    [row['lhs_shape'][-1] for row in selected],
                    [ratio(row) for row in selected],
                    marker='o',
                    markersize=3,
                    label=labels[layout],
                )
            add_reference(axis)
            axis.set_xscale('log', base=2)
            axis.set_title(f'{dtype}, {topology}')
            axis.set_xlabel('matrix side')
            axis.set_ylabel('NumPy / planned')
    axes[0][0].legend(fontsize=8)
    finish_figure(figure, output)


def render_broadcast(rows, output):
    dtypes = ('float32', 'float64')
    layouts = ('c', 'negative-core', 'step2-core')
    topologies = sorted({row['topology'] for row in rows})
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for row_number, dtype in enumerate(dtypes):
        for column, layout in enumerate(layouts):
            axis = axes[row_number][column]
            for topology in topologies:
                selected = []
                for row in rows:
                    row_layout, fields = split_layout(row)
                    if (
                        row['dtype'] == dtype
                        and row['topology'] == topology
                        and row_layout == layout
                    ):
                        selected.append((fields['batch'], ratio(row)))
                selected.sort()
                axis.plot(
                    [item[0] for item in selected],
                    [item[1] for item in selected],
                    marker='o',
                    markersize=3,
                    label=topology,
                )
            add_reference(axis)
            axis.set_xscale('log', base=2)
            axis.set_title(f'{dtype}, {layout}')
            axis.set_xlabel('batch')
            axis.set_ylabel('NumPy / planned')
    axes[0][0].legend(fontsize=7)
    finish_figure(figure, output)


def render_pack(rows, output):
    layouts = sorted({split_layout(row)[0] for row in rows})
    topologies = ('1d-nd', 'nd-1d')
    figure, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    positions = list(range(len(layouts)))
    for column, topology in enumerate(topologies):
        axis = axes[column]
        medians = []
        minima = []
        for layout in layouts:
            values = [
                ratio(row) for row in rows
                if row['topology'] == topology
                and split_layout(row)[0] == layout
            ]
            medians.append(median(values))
            minima.append(min(values))
        axis.scatter(
            medians,
            positions,
            color=SOLVCON_COLOR,
            label='median',
        )
        axis.scatter(
            minima,
            positions,
            color='#D55E00',
            marker='x',
            label='minimum',
        )
        axis.axvline(
            1.0,
            color=REFERENCE_COLOR,
            linewidth=1,
            linestyle='--',
        )
        axis.set_xscale('log', base=2)
        axis.grid(True, axis='x', alpha=0.2)
        axis.set_title(topology)
        axis.set_xlabel('NumPy / planned')
        axis.set_yticks(positions, layouts)
    axes[0].legend(fontsize=8)
    finish_figure(figure, output)


def render_vector_threshold(rows, output):
    topologies = ('1d-nd', 'nd-1d')
    batches = (1, 4, 16, 64)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for column, topology in enumerate(topologies):
        axis = axes[column]
        for batch in batches:
            selected = []
            for row in rows:
                _, fields = split_layout(row)
                if (
                    row['topology'] == topology
                    and fields['batch'] == batch
                ):
                    selected.append((fields['side'], ratio(row)))
            selected.sort()
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                marker='o',
                markersize=3,
                label=f'batch={batch}',
            )
        add_reference(axis)
        axis.set_xscale('log', base=2)
        axis.set_title(topology)
        axis.set_xlabel('matrix side')
        axis.set_ylabel('NumPy / planned')
    axes[0].legend(fontsize=8)
    finish_figure(figure, output)


def render_rectangular(rows, output):
    topologies = ('1d-nd', 'nd-1d')
    layouts = (
        'negative-vector',
        'negative-step2-vector',
        'zero-vector',
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for row_number, topology in enumerate(topologies):
        for column, layout in enumerate(layouts):
            axis = axes[row_number][column]
            groups = {}
            for row in rows:
                row_layout, fields = split_layout(row)
                if (
                    row['topology'] == topology
                    and row_layout == layout
                ):
                    key = (fields['k'], fields['o'])
                    groups.setdefault(key, []).append(ratio(row))
            pairs = sorted(groups)
            values = [median(groups[pair]) for pair in pairs]
            labels = [f'{pair[0]}x{pair[1]}' for pair in pairs]
            axis.plot(
                range(len(pairs)),
                values,
                marker='o',
                color=SOLVCON_COLOR,
            )
            add_reference(axis)
            axis.set_title(f'{topology}, {layout}')
            axis.set_xticks(
                range(len(pairs)),
                labels,
                rotation=45,
                ha='right',
                fontsize=7,
            )
            axis.set_xlabel('inner x output extent')
            axis.set_ylabel('median NumPy / planned')
    finish_figure(figure, output)


def render_reuse(rows, output):
    layouts = ('dense-reuse', 'negative-inner-reuse')
    methods = ('numpy', 'planned', 'numpy_prepacked',
               'planned_prepacked')
    labels = ('NumPy', 'planned', 'NumPy prepacked',
              'planned prepacked')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for column, layout in enumerate(layouts):
        axis = axes[column]
        row = next(row for row in rows if row['layout'] == layout)
        values = [
            row['timings'][method]['median_seconds'] * 1000
            for method in methods
        ]
        axis.bar(
            range(len(methods)),
            values,
            color=('#999999', SOLVCON_COLOR, '#BBBBBB', '#56B4E9'),
        )
        axis.set_xticks(
            range(len(methods)),
            labels,
            rotation=25,
            ha='right',
        )
        axis.set_title(layout.replace('-reuse', ''))
        axis.set_ylabel('median milliseconds')
        axis.grid(True, axis='y', alpha=0.2)
    finish_figure(figure, output)


def group_summary(name, rows):
    values = [ratio(row) for row in rows]
    return {
        'name': name,
        'cases': len(values),
        'median': median(values),
        'wins': sum(value >= 1.0 for value in values),
        'minimum': min(values),
        'maximum': max(values),
    }


def write_summary(groups, metadata, output):
    lines = [
        '# Apple M1 planned matmul benchmark summary',
        '',
        (
            f"Measured revision `{metadata['revision']}` with Python "
            f"{metadata['python_version']} and NumPy "
            f"{metadata['numpy_version']}."
        ),
        '',
        '| Group | Cases | Median NumPy / planned | >= 1 | Min | Max |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for summary in groups:
        lines.append(
            f"| {summary['name']} | {summary['cases']} | "
            f"{summary['median']:.3f}x | "
            f"{summary['wins']}/{summary['cases']} | "
            f"{summary['minimum']:.3f}x | "
            f"{summary['maximum']:.3f}x |"
        )
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render planned matmul benchmark reports.')
    parser.add_argument('--core', type=pathlib.Path, required=True)
    parser.add_argument('--broadcast', type=pathlib.Path, required=True)
    parser.add_argument('--pack', type=pathlib.Path, required=True)
    parser.add_argument('--threshold', type=pathlib.Path, required=True)
    parser.add_argument('--rectangular', type=pathlib.Path, required=True)
    parser.add_argument('--reuse', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        'core': load_results(args.core),
        'broadcast scaling': load_results(args.broadcast),
        'pack crossover': load_results(args.pack),
        'vector threshold': load_results(args.threshold),
        'vector rectangular': load_results(args.rectangular),
    }
    reuse = load_results(args.reuse)
    render_core(datasets['core'], args.output_dir / 'core.png')
    render_broadcast(
        datasets['broadcast scaling'],
        args.output_dir / 'broadcast-scaling.png',
    )
    render_pack(
        datasets['pack crossover'],
        args.output_dir / 'pack-crossover.png',
    )
    render_vector_threshold(
        datasets['vector threshold'],
        args.output_dir / 'vector-threshold.png',
    )
    render_rectangular(
        datasets['vector rectangular'],
        args.output_dir / 'vector-rectangular.png',
    )
    render_reuse(reuse, args.output_dir / 'broadcast-reuse.png')
    metadata = json.loads(
        args.core.read_text(encoding='utf-8'))['metadata']
    summaries = [
        group_summary(name, rows)
        for name, rows in datasets.items()
    ]
    write_summary(
        summaries,
        metadata,
        args.output_dir / 'summary.md',
    )


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
