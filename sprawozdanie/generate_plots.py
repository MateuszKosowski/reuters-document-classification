"""
Generowanie wykresów do sprawozdania z klasyfikacji k-NN na zbiorze Reuters-21578.
Uruchomienie: python generate_plots.py
Wymagane: matplotlib, numpy (pip install matplotlib numpy)
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'experiment-results')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUT, exist_ok=True)

# ── Styl globalny ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

COLORS = {
    'Manhattan': '#2563EB',
    'Euclidean': '#DC2626',
    'Chebyshev': '#16A34A',
}
CLASS_COLORS = {
    'usa':          '#2563EB',
    'uk':           '#DC2626',
    'japan':        '#F59E0B',
    'west-germany': '#16A34A',
    'canada':       '#9333EA',
    'france':       '#EC4899',
}

def load_csv(fname):
    path = os.path.join(RESULTS, fname)
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=';'))


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 1 – wpływ K i metryki
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stage1_k_metric():
    rows = [r for r in load_csv('01-k-metric-search.csv') if r['RowType'] == 'MACRO']

    metrics = ['Manhattan', 'Euclidean', 'Chebyshev']
    by_metric = defaultdict(list)
    for r in rows:
        by_metric[r['Metric']].append(r)
    for m in metrics:
        by_metric[m].sort(key=lambda r: int(r['K']))

    ks = [int(r['K']) for r in by_metric['Manhattan']]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    # ── Lewy: F1 makro vs K ───────────────────────────────────────────────────
    ax = axes[0]
    for m in metrics:
        f1s = [float(r['F1']) for r in by_metric[m]]
        ax.plot(ks, f1s, marker='o', markersize=3, color=COLORS[m], label=m, linewidth=1.6)
    ax.set_xlabel('Wartość k')
    ax.set_ylabel('F1 makro')
    ax.set_title('F1 makro w zależności od k i metryki')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(ks[::2])

    # ── Prawy: Accuracy vs K ──────────────────────────────────────────────────
    ax = axes[1]
    for m in metrics:
        accs = [float(r['Accuracy']) for r in by_metric[m]]
        ax.plot(ks, accs, marker='o', markersize=3, color=COLORS[m], label=m, linewidth=1.6)
    ax.set_xlabel('Wartość k')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy w zależności od k i metryki')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(ks[::2])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage1_k_metric.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage1_k_metric.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage1_k_metric')


def plot_stage1_precision_recall():
    """Precision i Recall makro vs K – pokazuje 'nożyce' między metrykami."""
    rows = [r for r in load_csv('01-k-metric-search.csv') if r['RowType'] == 'MACRO']

    metrics = ['Manhattan', 'Euclidean', 'Chebyshev']
    by_metric = defaultdict(list)
    for r in rows:
        by_metric[r['Metric']].append(r)
    for m in metrics:
        by_metric[m].sort(key=lambda r: int(r['K']))

    ks = [int(r['K']) for r in by_metric['Manhattan']]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, m in enumerate(metrics):
        ax = axes[i]
        prec = [float(r['Precision']) for r in by_metric[m]]
        rec  = [float(r['Recall'])    for r in by_metric[m]]
        f1   = [float(r['F1'])        for r in by_metric[m]]
        ax.plot(ks, prec, marker='s', markersize=2.5, color='#2563EB', label='Precision', linewidth=1.5)
        ax.plot(ks, rec,  marker='^', markersize=2.5, color='#DC2626', label='Recall',    linewidth=1.5)
        ax.plot(ks, f1,   marker='o', markersize=2.5, color='#16A34A', label='F1',        linewidth=1.5, linestyle='--')
        ax.set_title(f'Metryka: {m}')
        ax.set_xlabel('Wartość k')
        if i == 0:
            ax.set_ylabel('Wartość miary (makro)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(ks[::5])
    fig.suptitle('Precision, Recall i F1 makro w zależności od k', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage1_prec_rec_f1.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage1_prec_rec_f1.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage1_prec_rec_f1')


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 2 – wpływ podziału zbioru
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stage2_split():
    rows = [r for r in load_csv('02-split-search.csv') if r['RowType'] == 'MACRO']
    rows.sort(key=lambda r: float(r['TrainRatio']))

    train_ratios = [float(r['TrainRatio']) for r in rows]
    f1s  = [float(r['F1'])       for r in rows]
    accs = [float(r['Accuracy']) for r in rows]
    prec = [float(r['Precision']) for r in rows]
    rec  = [float(r['Recall'])    for r in rows]

    x = np.arange(len(rows))
    width = 0.2
    labels = [f"{int(float(r['TrainRatio'])*100)}/{int(float(r['TestRatio'])*100)}" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5*width, accs, width, label='Accuracy',  color='#2563EB', alpha=0.85)
    ax.bar(x - 0.5*width, prec, width, label='Precision', color='#DC2626', alpha=0.85)
    ax.bar(x + 0.5*width, rec,  width, label='Recall',    color='#16A34A', alpha=0.85)
    ax.bar(x + 1.5*width, f1s,  width, label='F1 makro',  color='#F59E0B', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Podział zbiór uczący / testowy [%]')
    ax.set_ylabel('Wartość miary')
    ax.set_title('Wpływ podziału zbioru na miary jakości klasyfikacji\n(metryka Manhattan, k=4, wszystkie cechy)')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    # Dodaj wartości F1 nad słupkami
    for xi, f1 in zip(x, f1s):
        ax.text(xi + 1.5*width, f1 + 0.015, f'{f1:.3f}', ha='center', va='bottom', fontsize=7.5, color='#B45309')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage2_split.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage2_split.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage2_split')


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 3 – wpływ liczby i wyboru cech
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stage3_feature_count():
    rows = [r for r in load_csv('03-feature-search.csv') if r['RowType'] == 'MACRO']

    by_count = defaultdict(list)
    for r in rows:
        by_count[int(r['FeatureCount'])].append(r)

    counts = sorted(by_count.keys())
    avg_f1   = [np.mean([float(r['F1'])       for r in by_count[c]]) for c in counts]
    max_f1   = [np.max( [float(r['F1'])       for r in by_count[c]]) for c in counts]
    min_f1   = [np.min( [float(r['F1'])       for r in by_count[c]]) for c in counts]
    avg_acc  = [np.mean([float(r['Accuracy']) for r in by_count[c]]) for c in counts]
    all_f1   = [[float(r['F1']) for r in by_count[c]] for c in counts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Lewy: średnia F1 z zakresem ───────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(counts, min_f1, max_f1, alpha=0.2, color='#2563EB', label='Min–Max F1')
    ax.plot(counts, avg_f1,  marker='o', color='#2563EB', linewidth=2, label='Średnia F1', zorder=3)
    ax.plot(counts, max_f1,  marker='^', color='#2563EB', linewidth=1, linestyle='--', alpha=0.7, label='Max F1')
    ax2 = ax.twinx()
    ax2.plot(counts, avg_acc, marker='s', color='#DC2626', linewidth=1.8, linestyle=':', label='Średnia Accuracy')
    ax2.set_ylabel('Średnia Accuracy', color='#DC2626')
    ax2.tick_params(axis='y', colors='#DC2626')
    ax.set_xlabel('Liczba cech')
    ax.set_ylabel('F1 makro')
    ax.set_title('Wpływ liczby cech na F1 makro i Accuracy\n(Manhattan, k=4, podział 60/40)')
    ax.set_xticks(counts)
    ax.legend(loc='lower right')
    ax2.legend(loc='center right')
    ax.grid(True, linestyle='--', alpha=0.4)

    # ── Prawy: boxplot F1 per liczba cech ─────────────────────────────────────
    ax = axes[1]
    bp = ax.boxplot(all_f1, positions=counts, widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    boxprops=dict(facecolor='#2563EB', alpha=0.7),
                    whiskerprops=dict(color='#2563EB'),
                    capprops=dict(color='#2563EB'),
                    flierprops=dict(marker='o', color='#2563EB', alpha=0.3, markersize=3))
    ax.set_xlabel('Liczba cech')
    ax.set_ylabel('F1 makro')
    ax.set_title('Rozkład F1 makro dla podzbiorów cech\n(Manhattan, k=4, podział 60/40)')
    ax.set_xticks(counts)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage3_feature_count.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage3_feature_count.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage3_feature_count')


def plot_stage3_feature_frequency():
    """Jak często każda cecha pojawia się w top-N konfiguracjach."""
    rows = [r for r in load_csv('03-feature-search.csv') if r['RowType'] == 'MACRO']
    rows.sort(key=lambda r: float(r['SelectionScore']), reverse=True)

    FEATURE_LABELS = {
        'longestWord':            'Najdłuższe słowo',
        'mostFrequentWord':       'Najczęstsze słowo',
        'averageWordLength':      'Śr. dł. słowa',
        'vocabularyRichness':     'Bogactwo słownictwa',
        'averageSentenceLength':  'Śr. dł. zdania',
        'uppercaseLetterRatio':   'Udział wielkich liter',
        'financialSignDensity':   'Zn. finansowe',
        'fleschReadingEaseIndex': 'Indeks Flescha',
        'vowelToConsonantRatio':  'Stosunek sam./spół.',
        'sumOfAllNumericValues':  'Suma wartości liczb.',
    }

    top_ns = [20, 50, 100, len(rows)]
    fig, axes = plt.subplots(1, len(top_ns), figsize=(14, 5), sharey=True)

    for ax, n in zip(axes, top_ns):
        subset = rows[:n]
        counter = defaultdict(int)
        for r in subset:
            for feat in r['Features'].split(','):
                counter[feat.strip()] += 1
        all_feats = list(FEATURE_LABELS.keys())
        counts = [counter.get(f, 0) for f in all_feats]
        labels = [FEATURE_LABELS[f] for f in all_feats]
        colors_bar = ['#2563EB' if c == max(counts) else '#93C5FD' for c in counts]
        ax.barh(labels, counts, color=colors_bar, edgecolor='white')
        ax.set_xlabel('Liczba wystąpień')
        ax.set_title(f'Top {n}' if n < len(rows) else f'Wszystkie ({len(rows)})')
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        for i, v in enumerate(counts):
            ax.text(v + 0.3, i, str(v), va='center', fontsize=8)

    axes[0].set_ylabel('Cecha')
    fig.suptitle('Częstość występowania cech w najlepszych konfiguracjach\n(Stage 3, Manhattan, k=4, podział 60/40)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage3_feature_frequency.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage3_feature_frequency.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage3_feature_frequency')


# ═══════════════════════════════════════════════════════════════════════════════
# NAJLEPSZA KONFIGURACJA – metryki per klasa
# ═══════════════════════════════════════════════════════════════════════════════

def plot_best_per_class():
    rows = load_csv('04-best-final-summary.csv')
    class_rows = [r for r in rows if r['RowType'] == 'CLASS']
    macro_row  = next(r for r in rows if r['RowType'] == 'MACRO')

    classes = [r['Class'] for r in class_rows]
    prec    = [float(r['Precision']) for r in class_rows]
    rec     = [float(r['Recall'])    for r in class_rows]
    f1      = [float(r['F1'])        for r in class_rows]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, prec, width, label='Precision', color='#2563EB', alpha=0.85)
    b2 = ax.bar(x,         rec,  width, label='Recall',    color='#DC2626', alpha=0.85)
    b3 = ax.bar(x + width, f1,   width, label='F1',        color='#16A34A', alpha=0.85)

    # Etykiety wartości nad słupkami F1
    for xi, v in zip(x, f1):
        ax.text(xi + width, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#14532D')

    # Linia makro F1
    ax.axhline(float(macro_row['F1']), color='#16A34A', linestyle='--', linewidth=1.5,
               label=f"Makro F1 = {float(macro_row['F1']):.3f}")
    ax.axhline(float(macro_row['Accuracy']), color='#7C3AED', linestyle=':', linewidth=1.5,
               label=f"Accuracy = {float(macro_row['Accuracy']):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_ylabel('Wartość miary')
    ax.set_ylim(0, 1.05)
    ax.set_title('Precision, Recall i F1 dla każdej klasy – najlepsza konfiguracja\n'
                 '(Manhattan, k=4, podział 60/40, 9 cech:\n'
                 'wszystkie oprócz uppercaseLetterRatio)')
    ax.legend(loc='upper left', fontsize=8.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'best_per_class.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'best_per_class.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: best_per_class')


def plot_best_class_imbalance():
    """Ilustracja nierównowagi klas (Recall jako proxy liczebności)."""
    rows = load_csv('04-best-final-summary.csv')
    class_rows = [r for r in rows if r['RowType'] == 'CLASS']
    classes = [r['Class'] for r in class_rows]
    f1   = [float(r['F1'])        for r in class_rows]
    prec = [float(r['Precision']) for r in class_rows]
    rec  = [float(r['Recall'])    for r in class_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cls, p, r, f) in enumerate(zip(classes, prec, rec, f1)):
        c = CLASS_COLORS.get(cls, '#666')
        ax.scatter(r, p, s=300 * f + 40, color=c, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
        ax.annotate(cls, (r, p), textcoords='offset points', xytext=(8, 4), fontsize=9)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall dla każdej klasy\n(rozmiar punktu proporcjonalny do F1)')
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.4, label='Precision = Recall')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'best_precision_recall_scatter.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'best_precision_recall_scatter.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: best_precision_recall_scatter')


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – porównanie metryk przy najlepszym K każdej metryki
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stage1_metric_comparison():
    rows = [r for r in load_csv('01-k-metric-search.csv') if r['RowType'] == 'MACRO']
    by_metric = defaultdict(list)
    for r in rows:
        by_metric[r['Metric']].append(r)

    metrics = ['Manhattan', 'Euclidean', 'Chebyshev']
    best_per_metric = {}
    for m in metrics:
        best = max(by_metric[m], key=lambda r: float(r['SelectionScore']))
        best_per_metric[m] = best

    meas = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(meas))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        vals = [float(best_per_metric[m][key]) for key in meas]
        bars = ax.bar(x + (i-1)*width, vals, width, label=f"{m} (k={int(best_per_metric[m]['K'])})",
                      color=COLORS[m], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(meas)
    ax.set_ylabel('Wartość miary')
    ax.set_ylim(0, 1.0)
    ax.set_title('Porównanie metryk przy optymalnym k dla każdej metryki\n(podział 50/50, wszystkie cechy)')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage1_metric_comparison.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage1_metric_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage1_metric_comparison')


def plot_stage3_ablation():
    """Ablacja: usunięcie jednej cechy z pełnego zestawu 10 cech – wpływ na Accuracy i F1."""
    rows = [r for r in load_csv('03-feature-search.csv') if r['RowType'] == 'MACRO']
    s2   = [r for r in load_csv('02-split-search.csv')   if r['RowType'] == 'MACRO']

    # baseline: 10 cech, podział 60/40 (optimum ze Stage 2)
    base = next(r for r in s2 if r['TrainRatio'] == '0.60')
    base_acc = float(base['Accuracy'])
    base_f1  = float(base['F1'])

    LABELS = {
        'longestWord':            'Najdłuższe słowo',
        'mostFrequentWord':       'Najczęstsze słowo',
        'averageWordLength':      'Śr. dł. słowa',
        'vocabularyRichness':     'Bogactwo słownictwa',
        'averageSentenceLength':  'Śr. dł. zdania',
        'uppercaseLetterRatio':   'Udział wielkich liter',
        'financialSignDensity':   'Zn. finansowe',
        'fleschReadingEaseIndex': 'Indeks Flescha',
        'vowelToConsonantRatio':  'Stosunek sam./spół.',
        'sumOfAllNumericValues':  'Suma wartości liczb.',
    }
    all10 = set(LABELS.keys())
    nine_feat = [r for r in rows if int(r['FeatureCount']) == 9]

    results = []
    for removed in all10:
        subset = all10 - {removed}
        for r in nine_feat:
            if set(r['Features'].split(',')) == subset:
                results.append((LABELS[removed],
                                 float(r['Accuracy']) - base_acc,
                                 float(r['F1'])  - base_f1))
                break

    # sort by delta accuracy
    results.sort(key=lambda x: x[0])  # alphabetical for consistency

    # ── single-feature accuracy ───────────────────────────────────────────────
    singles = [r for r in rows if int(r['FeatureCount']) == 1]
    single_data = {}
    for r in singles:
        feat = r['Features'].strip()
        single_data[LABELS[feat]] = (float(r['Accuracy']), float(r['F1']))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Lewy: pojedyncze cechy
    ax = axes[0]
    labels_s = [d[0] for d in sorted(single_data.items(), key=lambda x: x[1][0])]
    accs_s   = [single_data[l][0] for l in labels_s]
    f1s_s    = [single_data[l][1] for l in labels_s]
    y = np.arange(len(labels_s))
    ax.barh(y - 0.2, accs_s, 0.38, label='Accuracy', color='#2563EB', alpha=0.85)
    ax.barh(y + 0.2, f1s_s,  0.38, label='F1 makro', color='#16A34A', alpha=0.85)
    ax.axvline(base_acc, color='#2563EB', linestyle='--', linewidth=1.2, alpha=0.6, label=f'Acc (10 cech) = {base_acc:.3f}')
    ax.set_yticks(y)
    ax.set_yticklabels(labels_s, fontsize=8.5)
    ax.set_xlabel('Wartość miary')
    ax.set_title('Accuracy i F1 makro\ndla każdej cechy osobno')
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    # Prawy: ablacja (delta od baseline 10 cech)
    ax = axes[1]
    labels_a = [r[0] for r in results]
    delta_acc = [r[1] for r in results]
    delta_f1  = [r[2] for r in results]
    # sort by delta_acc for readability
    paired = sorted(zip(labels_a, delta_acc, delta_f1), key=lambda x: x[1])
    labels_a  = [p[0] for p in paired]
    delta_acc = [p[1] for p in paired]
    delta_f1  = [p[2] for p in paired]
    y = np.arange(len(labels_a))
    colors_acc = ['#DC2626' if d < 0 else '#16A34A' for d in delta_acc]
    ax.barh(y - 0.2, delta_acc, 0.38, color=colors_acc, alpha=0.85, label='ΔAccuracy')
    colors_f1 = ['#F97316' if d < 0 else '#6366F1' for d in delta_f1]
    ax.barh(y + 0.2, delta_f1,  0.38, color=colors_f1,  alpha=0.6,  label='ΔF1 makro')
    ax.axvline(0, color='black', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_a, fontsize=8.5)
    ax.set_xlabel('Zmiana miary względem baseline (10 cech)')
    ax.set_title('Ablacja: wpływ usunięcia jednej cechy\nna Accuracy i F1 makro')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC2626', alpha=0.85, label='ΔAcc < 0 (cecha pomaga)'),
        Patch(facecolor='#16A34A', alpha=0.85, label='ΔAcc > 0 (cecha szkodzi)'),
        Patch(facecolor='#F97316', alpha=0.6,  label='ΔF1 < 0'),
        Patch(facecolor='#6366F1', alpha=0.6,  label='ΔF1 > 0'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc='lower right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    fig.suptitle('Analiza wpływu poszczególnych cech na Accuracy i F1 makro\n'
                 '(Manhattan, k=4, podział 60/40)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stage3_ablation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'stage3_ablation.png'), bbox_inches='tight')
    plt.close(fig)
    print('Zapisano: stage3_ablation')


if __name__ == '__main__':
    print(f'Zapisywanie wykresów do: {OUT}')
    plot_stage1_k_metric()
    plot_stage1_precision_recall()
    plot_stage1_metric_comparison()
    plot_stage2_split()
    plot_stage3_feature_count()
    plot_stage3_feature_frequency()
    plot_stage3_ablation()
    plot_best_per_class()
    plot_best_class_imbalance()
    print('Gotowe.')
