"""Offline matplotlib figures and self-contained HTML; used on WSL only.

Charts consume measured data. Missing data is reported rather than imputed.
"""
from __future__ import annotations
import base64
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'axes.titleweight': 'bold'})


def save_figure(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def html_table(rows, columns=None):
    if not rows:
        return '<p>기록이 없습니다.</p>'
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    def fmt(v):
        if v is None: return '—'
        if isinstance(v, float): return f'{v:.5g}'
        if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
        return str(v)
    return '<div class="table"><table><thead><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(fmt(row.get(k))) + '</td>' for k in columns) + '</tr>' for row in rows) + '</tbody></table></div>'


def embedded_image(path, alt=''):
    data = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return '<img alt="' + html.escape(alt, quote=True) + '" src="data:image/png;base64,' + data + '">'


def write_page(path, title, sections, figures=()):
    """sections: (heading, trusted generated HTML); figures appended afterward."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f'<section><h2>{html.escape(heading)}</h2>{body}</section>' for heading, body in sections]
    parts.extend('<section>' + embedded_image(p, Path(p).stem) + '</section>' for p in figures)
    page = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#153041;background:#f2f6f8;margin:0}main{max-width:1160px;margin:auto;padding:30px 24px}h1{font-size:32px;line-height:1.25}h2{font-size:23px;color:#123d55}section{background:#fff;border:1px solid #dde6ec;border-radius:14px;padding:24px;margin:22px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{border-bottom:1px solid #d7e3eb;padding:8px;text-align:left;vertical-align:top}th{background:#e9f1f5;white-space:nowrap}.table{overflow:auto}code{background:#eef3f6;padding:2px 5px}p{margin-top:0}ul{padding-left:22px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}.card{border:1px solid #d7e3eb;padding:12px;border-radius:8px}.note{background:#fff5db;padding:12px;border-radius:8px}</style></head><body><main><h1>' + html.escape(title) + '</h1><p>이 보고서는 현재 실행에서 읽은 실제 데이터와 측정값으로 생성했습니다. 그림은 HTML 내부에 포함되어 인터넷 없이 열립니다.</p>' + ''.join(parts) + '</main></body></html>'
    path.write_text(page, encoding='utf-8')
    print('ANALYSIS:', path)
    return path


def confusion_counts(truth, prediction, classes):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError('Confusion matrix needs matched, nonempty 1D labels')
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Class labels must be integer indices')
    if min(truth.min(), prediction.min()) < 0 or max(truth.max(), prediction.max()) >= classes:
        raise ValueError('Class index outside configured labels')
    return np.bincount(classes * truth + prediction, minlength=classes * classes).reshape(classes, classes)


def confusion_figure(matrix, names, title, path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 7.8))
    im = ax.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(im, ax=ax, label='Number of examples', shrink=.8)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
           yticklabels=names, xlabel='Predicted class', ylabel='True class', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    limit = matrix.max() * .5
    for row in range(len(names)):
        for col in range(len(names)):
            ax.text(col, row, str(int(matrix[row, col])), ha='center', va='center',
                    fontsize=8, color='white' if matrix[row, col] > limit else '#173148')
    fig.tight_layout()
    return save_figure(fig, path)


def benchmark_groups(rows):
    """Only directly comparable execution/data groups share a chart."""
    grouped = defaultdict(list)
    fields = ('tag', 'backend', 'hostname', 'machine', 'batch', 'threads', 'input_data_sha256')
    for row in rows:
        if row.get('schema') != 'edge_benchmark_v1':
            continue
        if any(row.get(key) is None for key in ('accuracy', 'median_ms', 'p95_ms', 'bytes')):
            continue
        grouped[tuple(row.get(k) for k in fields)].append(row)
    return grouped


def benchmark_figures(rows, out_dir):
    paths = []; out_dir = Path(out_dir)
    for number, (key, records) in enumerate(benchmark_groups(rows).items(), start=1):
        tag, backend, host, machine, batch, threads, _ = key
        names = [f"{r['variant']}\n{r['precision']}" for r in records]
        x = np.arange(len(records))
        fig, axes = plt.subplots(1, 3, figsize=(max(12, len(records)*2), 4.5))
        axes[0].bar(x, [100*r['accuracy'] for r in records], color='#1f7a8c')
        axes[0].set(ylabel='Test accuracy (%)', ylim=(0, 105))
        for i, r in enumerate(records): axes[0].text(i,100*r['accuracy']+.8,f"{100*r['accuracy']:.1f}",ha='center',fontsize=8)
        axes[1].bar(x-.18, [r['median_ms'] for r in records], width=.36, label='Median', color='#265c8b')
        axes[1].bar(x+.18, [r['p95_ms'] for r in records], width=.36, label='P95', color='#e3a03e')
        axes[1].set(ylabel=f'Latency per batch of {batch} (ms)'); axes[1].legend()
        axes[2].bar(x, [r['bytes']/1024 for r in records], color='#7972b4')
        axes[2].set(ylabel='Model file size (KiB)')
        for ax in axes:
            ax.set_xticks(x, names, fontsize=8)
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        fig.suptitle(f'{tag} | {backend} | host={host} | {machine} | batch={batch} | threads={threads}', fontsize=12)
        fig.tight_layout()
        paths.append(save_figure(fig, out_dir/f'benchmark_{number:02d}.png'))
    return paths
