# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


DEST = Path(__file__).resolve().parent
FAMILIES = ('single', 'batch', 'broadcast', 'large')
TITLES = {
    'single': 'Single GEMM',
    'batch': 'Batched GEMM',
    'broadcast': 'Broadcast GEMM',
    'large': 'Large GEMM / Winograd',
}
LAYOUTS = {
    'single': [
        ('lhs', '(S, S)', '(2S, 2)'),
        ('rhs', '(S, S)', '(S, 1)'),
        ('output', '(S, S)', '(S, 1)'),
    ],
    'batch': [
        ('lhs', '(10, S, S)', '(2S^2, 2S, 2)'),
        ('rhs', '(10, S, S)', '(2S^2, 2S, 2)'),
        ('output', '(10, S, S)', '(S^2, S, 1)'),
    ],
    'broadcast': [
        ('lhs', '(1, S, S)', '(2S^2, 2S, 2)'),
        ('rhs', '(10, S, S)', '(2S^2, 2S, 2)'),
        ('output', '(10, S, S)', '(S^2, S, 1)'),
    ],
    'large': [
        ('lhs', '(S, S)', '(S, 1)'),
        ('rhs', '(S, S)', '(S, 1)'),
        ('output', '(S, S)', '(S, 1)'),
    ],
}
COLORS = {'numpy': '#64748B', 'solvcon': '#0072B2'}


def collect(families=FAMILIES):
    cases = {}
    files = sorted(DEST.glob('*-r[012].jsonl'))
    sources = files or [DEST / 'raw.jsonl']
    for path in sources:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['family'] not in families:
                continue
            cases.setdefault(row['case'], []).append(row)
    summary = []
    raw = []
    for case, rows in sorted(cases.items()):
        assert len(rows) == 60, (case, len(rows))
        rows.sort(key=lambda row: (
            row['replicate'], row['round'],
            row['method_order'].index(row['method'])))
        raw.extend(rows)
        metadata = rows[0]
        times = {
            name: np.array([row['sample_ns'] for row in rows
                            if row['method'] == name], dtype='float64')
            for name in COLORS
        }
        ratios = []
        replicates = []
        for replicate in range(3):
            subset = [row for row in rows if row['replicate'] == replicate]
            assert len(subset) == 20
            for round_index in range(10):
                pair = {row['method']: row['sample_ns'] for row in subset
                        if row['round'] == round_index}
                assert set(pair) == set(COLORS)
                ratios.append(pair['numpy'] / pair['solvcon'])
            medians = {name: float(np.median([
                row['sample_ns'] for row in subset if row['method'] == name]))
                for name in COLORS}
            replicates.append({
                'replicate': replicate,
                'numpy_median_ms': medians['numpy'] / 1e6,
                'solvcon_median_ms': medians['solvcon'] / 1e6,
                'speedup': medians['numpy'] / medians['solvcon'],
            })
        medians = {name: float(np.median(values))
                   for name, values in times.items()}
        process_times = {
            name: [item[f'{name}_median_ms'] for item in replicates]
            for name in COLORS
        }
        centers = {name: float(np.exp(np.mean(np.log(values))))
                   for name, values in process_times.items()}
        process_ratios = [item['speedup'] for item in replicates]
        row = {
            'family': metadata['family'],
            'dtype': metadata['dtype'],
            'side': metadata['side'],
            'threads': metadata['blas']['threads'],
            'numpy_time_ms': centers['numpy'],
            'solvcon_time_ms': centers['solvcon'],
            'speedup': centers['numpy'] / centers['solvcon'],
            'numpy_pooled_median_ms': medians['numpy'] / 1e6,
            'solvcon_pooled_median_ms': medians['solvcon'] / 1e6,
            'process_ratio_min': min(process_ratios),
            'process_ratio_max': max(process_ratios),
            'round_ratio_p10': float(np.quantile(ratios, 0.1)),
            'round_ratio_p90': float(np.quantile(ratios, 0.9)),
            'samples_per_method': len(times['numpy']),
            'process_replicates': replicates,
            'round_speedups': [float(value) for value in ratios],
            'lhs': metadata['lhs'],
            'rhs': metadata['rhs'],
            'output': metadata['validation']['output'],
            'relative_frobenius_error': max(
                item['validation']['relative_frobenius_error']
                for item in rows),
            'peak_rss_gib': max(item['peak_rss_kib'] for item in rows)
            / (1024 ** 2),
        }
        for name, values in times.items():
            row[f'{name}_process_min_ms'] = min(process_times[name])
            row[f'{name}_process_max_ms'] = max(process_times[name])
            for quantile in (10, 90):
                row[f'{name}_p{quantile}_ms'] = float(np.quantile(
                    values, quantile / 100)) / 1e6
            row[f'{name}_cpu_wall_ratio'] = float(np.median([
                item['cpu_wall_ratio'] for item in rows
                if item['method'] == name]))
        summary.append(row)
    return summary, raw


def plot_family(family, summary):
    selected = [row for row in summary if row['family'] == family]
    sides = sorted({row['side'] for row in selected})
    fig = plt.figure(figsize=(12, 8.5), facecolor='white')
    fig.text(0.075, 0.943, TITLES[family], fontsize=23, weight='bold')
    table_axis = fig.add_axes([0.075, 0.747, 0.86, 0.153])
    table_axis.axis('off')
    table = table_axis.table(
        cellText=LAYOUTS[family],
        colLabels=['Operand', 'Shape', 'Element strides'],
        colWidths=[0.16, 0.40, 0.44], cellLoc='left', colLoc='left',
        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#E2E8F0')
        cell.set_linewidth(0.6)
        cell.set_facecolor('#F1F5F9' if row == 0 else '#FFFFFF')
        if row == 0:
            cell.get_text().set_weight('bold')
        elif column:
            cell.get_text().set_fontfamily('monospace')
    x = np.log2(np.array(sides, dtype='float64'))
    unit = 's' if family == 'large' else 'ms'
    divisor = 1000 if family == 'large' else 1
    for column, dtype in enumerate(('float32', 'float64')):
        left = 0.09 + 0.47 * column
        ax = fig.add_axes([left, 0.36, 0.385, 0.31])
        ratio_ax = fig.add_axes([left, 0.095, 0.385, 0.17], sharex=ax)
        rows = sorted((row for row in selected if row['dtype'] == dtype),
                      key=lambda row: row['side'])
        assert [row['side'] for row in rows] == sides
        ax.set_title('FP32' if dtype == 'float32' else 'FP64',
                     fontsize=15, weight='bold', pad=12)
        for name, marker in (('numpy', 's'), ('solvcon', 'o')):
            y = [row[f'{name}_time_ms'] / divisor for row in rows]
            ax.plot(x, y, marker=marker, color=COLORS[name],
                    linewidth=2.1, markersize=5,
                    label='NumPy 2.5.1' if name == 'numpy' else 'solvcon')
            ax.fill_between(
                x, [row[f'{name}_process_min_ms'] / divisor for row in rows],
                [row[f'{name}_process_max_ms'] / divisor for row in rows],
                color=COLORS[name], alpha=0.10, linewidth=0)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, pos: f'{value:g}'))
        if family == 'large':
            ax.yaxis.set_minor_formatter(FuncFormatter(
                lambda value, pos: f'{value:g}'))
        ax.set_ylabel(f'Time ({unit}, lower is faster)', fontsize=10)
        ax.tick_params(axis='x', labelbottom=False)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ratio = [row['speedup'] for row in rows]
        ratio_ax.plot(x, ratio, color=COLORS['solvcon'], marker='o',
                      linewidth=2, markersize=5)
        ratio_ax.fill_between(
            x, [row['process_ratio_min'] for row in rows],
            [row['process_ratio_max'] for row in rows],
            color=COLORS['solvcon'], alpha=0.12, linewidth=0)
        ratio_ax.axhline(1, color='#64748B', linestyle='--', linewidth=1)
        ratio_ax.set_ylabel('Speedup over NumPy', fontsize=10)
        ratio_ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, pos: f'{value:g}x'))
        ratio_ax.set_xticks(x, [f'{side:,}' for side in sides])
        ratio_ax.set_xlabel('Matrix side S', labelpad=10)
        ratio_ax.annotate(f'{ratio[-1]:.2f}x', (x[-1], ratio[-1]),
                          xytext=(-6, 9), textcoords='offset points',
                          ha='right', color=COLORS['solvcon'],
                          fontsize=10, weight='bold')
        for axis in (ax, ratio_ax):
            axis.set_xlim(x[0] - 0.18, x[-1] + 0.25)
            axis.tick_params(labelsize=9)
            axis.grid(axis='y', color='#E2E8F0', linewidth=0.7)
            axis.spines[['top', 'right']].set_visible(False)
        if family == 'large':
            for axis in (ax, ratio_ax):
                axis.axvline(14, color='#94A3B8', linestyle=':', linewidth=1)
            ax.text(14.025, 0.04, 'Winograd',
                    transform=ax.get_xaxis_transform(), fontsize=9,
                    color='#475569')
    fig.savefig(DEST / f'pr1354-{family}.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('all', 'small'), default='all')
    parser.add_argument('--no-checksums', action='store_true')
    args = parser.parse_args()
    families = FAMILIES if args.suite == 'all' else FAMILIES[:3]
    suffix = '' if args.suite == 'all' else '-small'
    summary, raw = collect(families)
    assert len(summary) == (54 if args.suite == 'all' else 48), len(summary)
    (DEST / f'summary{suffix}.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    columns = [key for key in summary[0] if key not in (
        'round_speedups', 'process_replicates', 'lhs', 'rhs', 'output')]
    with (DEST / f'summary{suffix}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    (DEST / f'raw{suffix}.jsonl').write_text(''.join(
        json.dumps(row, sort_keys=True) + '\n' for row in raw))
    for family in families:
        plot_family(family, summary)
        values = [row['speedup'] for row in summary
                  if row['family'] == family]
        print(family, 'points', len(values), 'faster',
              sum(value > 1 for value in values), 'range',
              min(values), max(values), 'geomean',
              float(np.exp(np.mean(np.log(values)))))
    if args.suite == 'small' or args.no_checksums:
        return
    files = sorted(DEST.glob('pr1354-*.png')) + [
        DEST / name for name in ('raw.jsonl', 'summary.json', 'summary.csv',
                                 'profile_pr1354.py', 'plot_pr1354.py',
                                 'williams_schedule.py', 'audit_results.py',
                                 'manifest.json',
                                 'provenance.json', 'README.md', 'audit.txt')]
    (DEST / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name
        + '\n' for path in files if path.exists()))


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
