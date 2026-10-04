# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
        ('lhs', '(S, S)', '(2S, 2)', 'step-two'),
        ('rhs', '(S, S)', '(S, 1)', 'contiguous'),
        ('output', '(S, S)', '(S, 1)', 'contiguous'),
    ],
    'batch': [
        ('lhs', '(10, S, S)', '(2S^2, 2S, 2)', 'step-two'),
        ('rhs', '(10, S, S)', '(2S^2, 2S, 2)', 'step-two'),
        ('output', '(10, S, S)', '(S^2, S, 1)', 'contiguous'),
    ],
    'broadcast': [
        ('lhs', '(1, S, S)', '(2S^2, 2S, 2)', 'step-two; reused 10x'),
        ('rhs', '(10, S, S)', '(2S^2, 2S, 2)', 'step-two'),
        ('output', '(10, S, S)', '(S^2, S, 1)', 'contiguous'),
    ],
    'large': [
        ('lhs', '(S, S)', '(S, 1)', 'contiguous'),
        ('rhs', '(S, S)', '(S, 1)', 'contiguous'),
        ('output', '(S, S)', '(S, 1)', 'contiguous'),
    ],
}
COLORS = {'float32': '#0072B2', 'float64': '#D55E00'}


def collect():
    summary = []
    raw = []
    cases = {}
    for family in FAMILIES:
        for path in sorted(DEST.glob(f'{family}-*.jsonl')):
            cases[path.stem] = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
    if not cases:
        for line in (DEST / 'raw.jsonl').read_text().splitlines():
            row = json.loads(line)
            cases.setdefault(row['case'], []).append(row)
    for family in FAMILIES:
        for case, rows in sorted(cases.items()):
            if rows[0]['family'] != family:
                continue
            assert len(rows) == 12, case
            raw.extend(rows)
            metadata = rows[0]
            times = {
                name: np.array([
                    value for row in rows if row['method'] == name
                    for value in row['samples_ns']
                ], dtype='float64')
                for name in ('numpy', 'solvcon')
            }
            ratios = []
            for round_index in range(6):
                pair = {
                    row['method']: np.median(row['samples_ns'])
                    for row in rows if row['round'] == round_index
                }
                assert len(pair) == 2
                ratios.append(pair['numpy'] / pair['solvcon'])
            medians = {
                name: float(np.median(values))
                for name, values in times.items()
            }
            summary.append({
                'family': family,
                'dtype': metadata['dtype'],
                'side': metadata['side'],
                'threads': metadata['blas']['threads'],
                'numpy_median_ms': medians['numpy'] / 1e6,
                'solvcon_median_ms': medians['solvcon'] / 1e6,
                'speedup': medians['numpy'] / medians['solvcon'],
                'round_ratio_p10': float(np.quantile(ratios, 0.1)),
                'round_ratio_p90': float(np.quantile(ratios, 0.9)),
                'samples_per_method': len(times['numpy']),
                'round_speedups': [float(value) for value in ratios],
                'lhs': metadata['lhs'],
                'rhs': metadata['rhs'],
                'output': metadata['validation']['output'],
                'relative_frobenius_error': metadata['validation'][
                    'relative_frobenius_error'],
                'peak_rss_gib': max(row['peak_rss_kib'] for row in rows)
                / (1024 ** 2),
                'major_faults': max(row['major_faults'] for row in rows),
            })
    return summary, raw


def plot_family(family, summary):
    selected = [row for row in summary if row['family'] == family]
    sides = sorted({row['side'] for row in selected})
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    fig.text(0.075, 0.94, TITLES[family], fontsize=23, weight='bold')
    fig.text(0.075, 0.899, 'Public matmul vs NumPy 2.5.1',
             fontsize=13, color='#475569')
    table_axis = fig.add_axes([0.065, 0.699, 0.88, 0.175])
    table_axis.axis('off')
    table = table_axis.table(
        cellText=LAYOUTS[family],
        colLabels=['Operand', 'Shape', 'Element strides', 'Layout'],
        colWidths=[0.12, 0.23, 0.27, 0.38],
        cellLoc='left', colLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#E2E8F0')
        cell.set_linewidth(0.6)
        cell.set_facecolor('#F1F5F9' if row == 0 else '#FFFFFF')
        if row == 0:
            cell.get_text().set_weight('bold')
        elif column in (1, 2):
            cell.get_text().set_fontfamily('monospace')
    ax = fig.add_axes([0.105, 0.18, 0.81, 0.47])
    x = np.log2(np.array(sides, dtype='float64'))
    final_values = {row['dtype']: row['speedup'] for row in selected
                    if row['side'] == sides[-1]}
    higher_dtype = max(final_values, key=final_values.get)
    for dtype, marker in (('float32', 'o'), ('float64', 's')):
        rows = sorted((row for row in selected if row['dtype'] == dtype),
                      key=lambda row: row['side'])
        assert [row['side'] for row in rows] == sides
        y = [row['speedup'] for row in rows]
        ax.plot(x, y, marker=marker, color=COLORS[dtype],
                linewidth=2.3, markersize=6.5,
                label='FP32' if dtype == 'float32' else 'FP64')
        ax.fill_between(
            x, [row['round_ratio_p10'] for row in rows],
            [row['round_ratio_p90'] for row in rows],
            color=COLORS[dtype], alpha=0.10, linewidth=0)
        ax.annotate(f'{y[-1]:.2f}x', (x[-1], y[-1]),
                    xytext=(8, 6 if dtype == higher_dtype else -16),
                    textcoords='offset points', color=COLORS[dtype],
                    fontsize=10, weight='bold')
    ax.axhline(1, color='#64748B', linestyle='--', linewidth=1.1)
    ax.set_xticks(x, [f'{side:,}' for side in sides])
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.3)
    ax.set_xlabel('Matrix side S (log2 scale)', labelpad=10)
    ax.set_ylabel('Speedup over NumPy (higher is faster)', labelpad=10)
    ax.grid(axis='y', color='#E2E8F0', linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='best', frameon=False, ncols=2)
    if family == 'large':
        threshold = np.log2(16384)
        ax.axvline(threshold, color='#94A3B8', linestyle=':', linewidth=1)
        ax.text(threshold + 0.04, 0.98, 'Winograd from S = 16,384',
                transform=ax.get_xaxis_transform(), va='top',
                fontsize=9, color='#475569')
    threads = 24 if family == 'large' else 1
    thread_word = 'thread' if threads == 1 else 'threads'
    samples = 6 if family == 'large' else 90
    fig.text(0.075, 0.092,
             f'WSL2 | Intel i7-13700K | OpenBLAS 0.3.26 | '
             f'{threads} BLAS {thread_word} for both libraries',
             fontsize=9, color='#475569')
    fig.text(0.075, 0.067,
             f'Ratio of medians; {samples} samples/method. '
             'Shading: p10-p90 of six round speedups.',
             fontsize=9, color='#475569')
    fig.text(0.075, 0.042,
             'Includes output allocation/release and internal packing/'
             'scratch. Strides are in elements. PR #1354: a1494b45.',
             fontsize=9, color='#475569')
    fig.savefig(DEST / f'pr1354-{family}.png', dpi=180)
    plt.close(fig)


def main():
    summary, raw = collect()
    assert len(summary) == 54, len(summary)
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    columns = [key for key in summary[0] if key not in (
        'round_speedups', 'lhs', 'rhs', 'output')]
    with (DEST / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    (DEST / 'raw.jsonl').write_text(''.join(
        json.dumps(row, sort_keys=True) + '\n' for row in raw))
    for family in FAMILIES:
        plot_family(family, summary)
        values = [row['speedup'] for row in summary
                  if row['family'] == family]
        print(family, 'points', len(values), 'faster',
              sum(value > 1 for value in values), 'range',
              min(values), max(values), 'geomean',
              float(np.exp(np.mean(np.log(values)))))
    files = sorted(DEST.glob('pr1354-*.png')) + [
        DEST / name for name in ('raw.jsonl', 'summary.json', 'summary.csv',
                                 'profile_pr1354.py', 'plot_pr1354.py',
                                 'manifest.json', 'provenance.json',
                                 'README.md', 'audit.txt')]
    (DEST / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name
        + '\n' for path in files if path.exists()))


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
