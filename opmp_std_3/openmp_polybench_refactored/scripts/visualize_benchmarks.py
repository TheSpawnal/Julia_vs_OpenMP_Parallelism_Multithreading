#!/usr/bin/env python3
"""
OpenMP PolyBench Visualization Suite
NEON CYBERPUNK EDITION

Designed for OpenMP benchmark analysis and Julia vs OpenMP comparison.
Works with CSV files from both implementations.

Features:
- Strategy performance dashboard
- Thread scaling with Amdahl's Law overlays
- Strategy comparison heatmaps
- Multi-benchmark comparison
- Julia vs OpenMP comparison (when both present)

Usage:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv -o ./plots
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
    python3 visualize_benchmarks.py julia/*.csv openmp/*.csv --compare

Author: SpawnAl / Falkor collaboration
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime

# =============================================================================
# NEON CYBERPUNK COLOR PALETTE
# =============================================================================
COLORS = {
    'neon_cyan': '#00FFFF',
    'neon_magenta': '#FF00FF',
    'neon_green': '#39FF14',
    'neon_orange': '#FF6600',
    'neon_yellow': '#FFFF00',
    'neon_pink': '#FF1493',
    'neon_blue': '#00BFFF',
    'neon_purple': '#BF00FF',
    'electric_lime': '#CCFF00',
    'hot_coral': '#FF4040',
    'plasma_violet': '#9D00FF',
    'cyber_teal': '#00CED1',
    'laser_gold': '#FFD700',
    'dark_bg': '#0D1117',
    'panel_bg': '#161B22',
    'grid': '#30363D',
    'text': '#E6EDF3',
    'text_dim': '#8B949E',
}

STRATEGY_COLORS = {
    'sequential': COLORS['text_dim'],
    'seq': COLORS['text_dim'],
    'threads_static': COLORS['neon_cyan'],
    'threads': COLORS['neon_cyan'],
    'static': COLORS['neon_cyan'],
    'threads_dynamic': COLORS['neon_magenta'],
    'dynamic': COLORS['neon_magenta'],
    'tiled': COLORS['neon_green'],
    'blocked': COLORS['neon_green'],
    'blas': COLORS['laser_gold'],
    'tasks': COLORS['neon_purple'],
    'wavefront': COLORS['neon_orange'],
    'simd': COLORS['neon_blue'],
    'collapsed': COLORS['neon_pink'],
    'colmajor': COLORS['electric_lime'],
}

BENCHMARK_COLORS = {
    '2mm': COLORS['neon_cyan'],
    '3mm': COLORS['neon_magenta'],
    'cholesky': COLORS['neon_green'],
    'correlation': COLORS['laser_gold'],
    'nussinov': COLORS['neon_orange'],
    'heat3d': COLORS['neon_purple'],
    'jacobi2d': COLORS['neon_pink'],
}

LANGUAGE_COLORS = {
    'julia': COLORS['neon_green'],
    'openmp': COLORS['neon_orange'],
    'c': COLORS['neon_blue'],
}


def get_strategy_color(strategy):
    return STRATEGY_COLORS.get(strategy.lower(), COLORS['neon_blue'])


def get_benchmark_color(benchmark):
    return BENCHMARK_COLORS.get(benchmark.lower(), COLORS['neon_cyan'])


def get_language_color(lang):
    return LANGUAGE_COLORS.get(lang.lower(), COLORS['text'])


# =============================================================================
# MATPLOTLIB STYLE SETUP
# =============================================================================
def setup_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['dark_bg'],
        'figure.edgecolor': COLORS['grid'],
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.facecolor': COLORS['dark_bg'],
        'axes.facecolor': COLORS['panel_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'font.family': 'monospace',
        'font.size': 10,
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'legend.frameon': True,
        'legend.facecolor': COLORS['panel_bg'],
        'legend.edgecolor': COLORS['grid'],
        'legend.fontsize': 9,
        'legend.labelcolor': COLORS['text'],
    })


# =============================================================================
# DATA LOADING
# =============================================================================
def detect_language(filepath, df):
    """Detect if CSV is from Julia or OpenMP based on filename or content."""
    fname = Path(filepath).name.lower()
    if 'julia' in fname:
        return 'julia'
    if 'openmp' in fname or 'omp' in fname:
        return 'openmp'
    # Check content patterns
    if 'blas' in df['strategy'].str.lower().values:
        return 'julia'  # BLAS strategy is Julia-specific
    return 'openmp'


def load_csv(filepath):
    """Load a single CSV file with language detection."""
    try:
        df = pd.read_csv(filepath)
        df['source_file'] = Path(filepath).name
        df['language'] = detect_language(filepath, df)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_multiple_csvs(filepaths):
    """Load and concatenate multiple CSV files."""
    dfs = []
    for fp in filepaths:
        df = load_csv(fp)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    # Normalize efficiency column name
    if 'efficiency' in combined.columns and 'efficiency_pct' not in combined.columns:
        combined['efficiency_pct'] = combined['efficiency']
    return combined


# =============================================================================
# FILENAME GENERATION
# =============================================================================
def generate_filename(plot_type, df, suffix=None):
    """Generate descriptive filename from data."""
    parts = [plot_type]
    if 'benchmark' in df.columns:
        benchmarks = df['benchmark'].unique()
        if len(benchmarks) == 1:
            parts.append(str(benchmarks[0]))
        else:
            parts.append(f"{len(benchmarks)}bench")
    if 'dataset' in df.columns:
        datasets = df['dataset'].unique()
        if len(datasets) == 1:
            parts.append(str(datasets[0]))
    if 'threads' in df.columns:
        threads = sorted(df['threads'].unique())
        if len(threads) == 1:
            parts.append(f"{threads[0]}T")
        else:
            parts.append(f"{min(threads)}-{max(threads)}T")
    if 'language' in df.columns:
        langs = df['language'].unique()
        if len(langs) > 1:
            parts.append("vs".join(sorted(langs)))
        elif len(langs) == 1:
            parts.append(langs[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + ".png"


# =============================================================================
# AMDAHL'S LAW
# =============================================================================
def amdahl_speedup(threads, parallel_fraction):
    """S = 1 / ((1-f) + f/p)"""
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / np.array(threads))


def amdahl_efficiency(threads, parallel_fraction):
    """E = S/p * 100"""
    s = amdahl_speedup(threads, parallel_fraction)
    return s / np.array(threads) * 100


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_summary_dashboard(df, output_path, title_prefix=""):
    """4-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{title_prefix}OpenMP Benchmark Performance Summary',
                 fontsize=16, fontweight='bold', color=COLORS['neon_cyan'])

    strategies = df['strategy'].unique()
    colors = [get_strategy_color(s) for s in strategies]
    y_pos = np.arange(len(strategies))

    # Subtitle with metadata
    subtitle_parts = []
    if 'benchmark' in df.columns:
        subtitle_parts.append(f"Benchmark: {', '.join(df['benchmark'].unique())}")
    if 'dataset' in df.columns:
        subtitle_parts.append(f"Dataset: {', '.join(df['dataset'].unique())}")
    if 'threads' in df.columns:
        subtitle_parts.append(f"Threads: {sorted(df['threads'].unique())}")
    if subtitle_parts:
        fig.text(0.5, 0.95, " | ".join(subtitle_parts), ha='center',
                 fontsize=10, color=COLORS['text_dim'])

    # Panel 1: Execution Time
    ax1 = fig.add_subplot(2, 2, 1)
    times = df.groupby('strategy')['median_ms'].mean().reindex(strategies)
    bars1 = ax1.barh(y_pos, times, color=colors, edgecolor=COLORS['neon_cyan'], linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('Time (ms)', color=COLORS['text'])
    ax1.set_title('Execution Time (lower is better)', color=COLORS['neon_cyan'])
    ax1.invert_yaxis()
    for bar, t in zip(bars1, times):
        if not np.isnan(t):
            ax1.text(bar.get_width() + times.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{t:.2f}', va='center', fontsize=9, color=COLORS['text'])

    # Panel 2: Speedup
    if 'speedup' in df.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        speedups = df.groupby('strategy')['speedup'].mean().reindex(strategies)
        bar_colors = [COLORS['neon_green'] if s > 1 else COLORS['hot_coral'] for s in speedups]
        bars2 = ax2.barh(y_pos, speedups, color=bar_colors, edgecolor=COLORS['neon_green'], linewidth=1)
        ax2.axvline(x=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(strategies)
        ax2.set_xlabel('Speedup (vs sequential)', color=COLORS['text'])
        ax2.set_title('Speedup (higher is better)', color=COLORS['neon_green'])
        ax2.invert_yaxis()
        for bar, s in zip(bars2, speedups):
            if not np.isnan(s):
                ax2.text(bar.get_width() + speedups.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                         f'{s:.2f}x', va='center', fontsize=9, color=COLORS['text'])

    # Panel 3: GFLOP/s
    if 'gflops' in df.columns:
        ax3 = fig.add_subplot(2, 2, 3)
        gflops = df.groupby('strategy')['gflops'].mean().reindex(strategies)
        bars3 = ax3.barh(y_pos, gflops, color=colors, edgecolor=COLORS['laser_gold'], linewidth=1)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(strategies)
        ax3.set_xlabel('GFLOP/s', color=COLORS['text'])
        ax3.set_title('Throughput (higher is better)', color=COLORS['laser_gold'])
        ax3.invert_yaxis()
        for bar, g in zip(bars3, gflops):
            if not np.isnan(g):
                ax3.text(bar.get_width() + gflops.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                         f'{g:.1f}', va='center', fontsize=9, color=COLORS['text'])

    # Panel 4: Efficiency
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    if eff_col in df.columns:
        ax4 = fig.add_subplot(2, 2, 4)
        df_eff = df.copy()
        if df_eff[eff_col].dtype == object:
            df_eff = df_eff[df_eff[eff_col] != '']
            df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')
        eff = df_eff.groupby('strategy')[eff_col].mean().reindex(strategies)
        bar_colors = [COLORS['neon_purple'] if e > 50 else COLORS['text_dim'] for e in eff.fillna(0)]
        bars4 = ax4.barh(y_pos, eff.fillna(0), color=bar_colors, edgecolor=COLORS['neon_purple'], linewidth=1)
        ax4.axvline(x=100.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7, linewidth=2)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(strategies)
        ax4.set_xlabel('Efficiency (%)', color=COLORS['text'])
        ax4.set_title('Parallel Efficiency (parallel strategies only)', color=COLORS['neon_purple'])
        ax4.invert_yaxis()
        for bar, e in zip(bars4, eff):
            if not np.isnan(e):
                ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                         f'{e:.1f}%', va='center', fontsize=9, color=COLORS['text'])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_thread_scaling_chart(df, output_path, title_prefix=""):
    """Thread scaling with Amdahl's Law overlay."""
    if 'threads' not in df.columns:
        print("No 'threads' column for scaling chart")
        return
    if df['threads'].nunique() < 2:
        print("Need multiple thread counts for scaling chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title_prefix}Thread Scaling Analysis',
                 fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])

    # Subtitle
    subtitle_parts = []
    if 'benchmark' in df.columns:
        subtitle_parts.append(f"Benchmark: {', '.join(df['benchmark'].unique())}")
    if 'dataset' in df.columns:
        subtitle_parts.append(f"Dataset: {', '.join(df['dataset'].unique())}")
    if subtitle_parts:
        fig.text(0.5, 0.94, " | ".join(subtitle_parts), ha='center',
                 fontsize=10, color=COLORS['text_dim'])

    threads = sorted(df['threads'].unique())
    thread_range = np.linspace(1, max(threads), 100)

    # Left: Speedup
    ax1 = axes[0]
    parallel_strategies = df[df['is_parallel'] == True]['strategy'].unique() if 'is_parallel' in df.columns else \
        [s for s in df['strategy'].unique() if s.lower() not in ['sequential', 'seq', 'simd', 'blas']]

    for strategy in parallel_strategies:
        df_s = df[df['strategy'] == strategy]
        speedups = df_s.groupby('threads')['speedup'].mean()
        ax1.plot(speedups.index, speedups.values, 'o-', color=get_strategy_color(strategy),
                 label=strategy, linewidth=2, markersize=8)

    # Amdahl overlays
    for f, style, label in [(0.95, '--', 'Amdahl f=95%'), (0.90, ':', 'Amdahl f=90%'),
                            (0.80, '-.', 'Amdahl f=80%')]:
        ax1.plot(thread_range, amdahl_speedup(thread_range, f), style,
                 color=COLORS['text_dim'], alpha=0.5, label=label)

    # Ideal scaling
    ax1.plot(threads, threads, '--', color=COLORS['neon_yellow'], alpha=0.7, label='Ideal (linear)')

    ax1.set_xlabel('Threads', color=COLORS['text'])
    ax1.set_ylabel('Speedup', color=COLORS['text'])
    ax1.set_title('Speedup vs Threads', color=COLORS['neon_green'])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, max(threads) + 1)
    ax1.set_ylim(0, max(threads) + 1)

    # Right: Efficiency
    ax2 = axes[1]
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    if eff_col in df.columns:
        df_eff = df.copy()
        if df_eff[eff_col].dtype == object:
            df_eff = df_eff[df_eff[eff_col] != '']
            df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')

        for strategy in parallel_strategies:
            df_s = df_eff[df_eff['strategy'] == strategy]
            eff = df_s.groupby('threads')[eff_col].mean()
            ax2.plot(eff.index, eff.values, 'o-', color=get_strategy_color(strategy),
                     label=strategy, linewidth=2, markersize=8)

        # Amdahl efficiency overlays
        for f, style in [(0.95, '--'), (0.90, ':'), (0.80, '-.')]:
            ax2.plot(thread_range, amdahl_efficiency(thread_range, f), style,
                     color=COLORS['text_dim'], alpha=0.5)

        ax2.axhline(y=100, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7)
        ax2.set_xlabel('Threads', color=COLORS['text'])
        ax2.set_ylabel('Efficiency (%)', color=COLORS['text'])
        ax2.set_title('Parallel Efficiency vs Threads', color=COLORS['neon_purple'])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_xlim(0, max(threads) + 1)
        ax2.set_ylim(0, 120)

    # Explanation
    fig.text(0.5, 0.02,
             "Amdahl's Law: S = 1/((1-f) + f/p) where f = parallel fraction, p = threads. "
             "Efficiency naturally decreases as threads increase unless f = 100%.",
             ha='center', fontsize=9, color=COLORS['text_dim'], style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_strategy_heatmap(df, output_path, title_prefix=""):
    """Heatmap: Strategies x Metrics."""
    if 'strategy' not in df.columns:
        print("No 'strategy' column for heatmap")
        return

    strategies = df['strategy'].unique()

    # Collect metrics per strategy
    data = []
    for strategy in strategies:
        df_s = df[df['strategy'] == strategy]
        row = {
            'strategy': strategy,
            'min_ms': df_s['min_ms'].mean() if 'min_ms' in df_s.columns else np.nan,
            'speedup': df_s['speedup'].mean() if 'speedup' in df_s.columns else np.nan,
            'gflops': df_s['gflops'].mean() if 'gflops' in df_s.columns else np.nan,
        }
        eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
        if eff_col in df.columns:
            df_eff = df_s.copy()
            if df_eff[eff_col].dtype == object:
                df_eff = df_eff[df_eff[eff_col] != '']
                df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')
            row['efficiency'] = df_eff[eff_col].mean() if not df_eff.empty else np.nan
        data.append(row)

    heatmap_df = pd.DataFrame(data).set_index('strategy')

    # Normalize for color mapping
    heatmap_norm = heatmap_df.copy()
    for col in heatmap_norm.columns:
        col_min = heatmap_norm[col].min()
        col_max = heatmap_norm[col].max()
        if col_max > col_min:
            if 'ms' in col.lower() or 'time' in col.lower():
                heatmap_norm[col] = 1 - (heatmap_norm[col] - col_min) / (col_max - col_min)
            else:
                heatmap_norm[col] = (heatmap_norm[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(12, 8))

    neon_cmap = LinearSegmentedColormap.from_list('neon',
        [COLORS['dark_bg'], COLORS['neon_purple'], COLORS['neon_cyan'], COLORS['neon_green']])

    im = ax.imshow(heatmap_norm.values, cmap=neon_cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    col_labels = ['Time (ms)\n(lower=better)', 'Speedup\n(higher=better)',
                  'GFLOP/s\n(higher=better)', 'Efficiency %\n(parallel only)']
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels(col_labels[:len(heatmap_df.columns)])
    ax.set_yticklabels(heatmap_df.index)

    # Annotate with actual values
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.values[i, j]
            if not np.isnan(val):
                if j == 0:
                    text = f'{val:.2f}'
                elif j == 1:
                    text = f'{val:.2f}x'
                elif j == 2:
                    text = f'{val:.1f}'
                else:
                    text = f'{val:.1f}%'
                bg_val = heatmap_norm.values[i, j]
                text_color = COLORS['dark_bg'] if bg_val > 0.5 else COLORS['text']
                ax.text(j, i, text, ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

    title = f'{title_prefix}Strategy Performance Heatmap'
    if 'benchmark' in df.columns:
        title += f" - {', '.join(df['benchmark'].unique())}"
    if 'dataset' in df.columns:
        title += f" [{', '.join(df['dataset'].unique())}]"
    ax.set_title(title, color=COLORS['neon_cyan'], fontsize=14, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (normalized, higher=better)', color=COLORS['text'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_benchmark_comparison(df, output_path, title_prefix=""):
    """Compare multiple benchmarks."""
    if 'benchmark' not in df.columns:
        print("No 'benchmark' column for comparison")
        return

    benchmarks = df['benchmark'].unique()
    if len(benchmarks) < 2:
        print("Need multiple benchmarks for comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    subtitle_parts = []
    if 'dataset' in df.columns:
        subtitle_parts.append(f"Dataset: {', '.join(df['dataset'].unique())}")
    if 'threads' in df.columns:
        subtitle_parts.append(f"Threads: {sorted(df['threads'].unique())}")

    fig.suptitle(f'{title_prefix}Multi-Benchmark Comparison',
                 fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])
    if subtitle_parts:
        fig.text(0.5, 0.94, " | ".join(subtitle_parts), ha='center',
                 fontsize=10, color=COLORS['text_dim'])

    x = np.arange(len(benchmarks))
    width = 0.6

    # Best speedup per benchmark
    ax1 = axes[0, 0]
    best_speedup = df.groupby('benchmark')['speedup'].max().reindex(benchmarks)
    colors = [get_benchmark_color(b) for b in benchmarks]
    bars = ax1.bar(x, best_speedup.values, color=colors, edgecolor='white', linewidth=1, width=width)
    ax1.axhline(y=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7)
    ax1.set_ylabel('Best Speedup')
    ax1.set_title('Best Speedup by Benchmark', color=COLORS['neon_green'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
    for bar, s in zip(bars, best_speedup.values):
        if not np.isnan(s):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{s:.2f}x', ha='center', fontsize=9, color=COLORS['text'])

    # Best GFLOP/s per benchmark
    ax2 = axes[0, 1]
    if 'gflops' in df.columns:
        best_gflops = df.groupby('benchmark')['gflops'].max().reindex(benchmarks)
        bars = ax2.bar(x, best_gflops.values, color=colors, edgecolor='white', linewidth=1, width=width)
        ax2.set_ylabel('Best GFLOP/s')
        ax2.set_title('Peak Throughput by Benchmark', color=COLORS['laser_gold'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
        for bar, g in zip(bars, best_gflops.values):
            if not np.isnan(g):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{g:.1f}', ha='center', fontsize=9, color=COLORS['text'])

    # Sequential time comparison
    ax3 = axes[1, 0]
    seq_time = df[df['strategy'].str.lower().isin(['sequential', 'seq'])].groupby('benchmark')['min_ms'].mean()
    seq_time = seq_time.reindex(benchmarks)
    bars = ax3.bar(x, seq_time.values, color=colors, edgecolor='white', linewidth=1, width=width)
    ax3.set_ylabel('Sequential Time (ms)')
    ax3.set_title('Sequential Baseline Time', color=COLORS['neon_orange'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks, rotation=45, ha='right')

    # Best strategy per benchmark
    ax4 = axes[1, 1]
    best_strategies = []
    for b in benchmarks:
        df_b = df[df['benchmark'] == b]
        if not df_b.empty and 'speedup' in df_b.columns:
            best_idx = df_b['speedup'].idxmax()
            best_strategies.append(df_b.loc[best_idx, 'strategy'])
        else:
            best_strategies.append('N/A')
    
    y_pos = np.arange(len(benchmarks))
    ax4.barh(y_pos, [1] * len(benchmarks), color=colors, edgecolor='white', linewidth=1)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(benchmarks)
    ax4.set_xlim(0, 1.5)
    ax4.set_xticks([])
    ax4.set_title('Best Strategy per Benchmark', color=COLORS['neon_purple'])
    for i, (bench, strat) in enumerate(zip(benchmarks, best_strategies)):
        ax4.text(0.5, i, strat, ha='center', va='center', fontsize=11,
                 color=COLORS['dark_bg'], fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_julia_vs_openmp_comparison(df, output_path, title_prefix=""):
    """Compare Julia and OpenMP results."""
    if 'language' not in df.columns:
        print("No 'language' column for comparison")
        return

    languages = df['language'].unique()
    if len(languages) < 2:
        print("Need multiple languages for comparison")
        return

    benchmarks = df['benchmark'].unique() if 'benchmark' in df.columns else ['benchmark']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix}Julia vs OpenMP Comparison',
                 fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])

    x = np.arange(len(benchmarks))
    width = 0.35

    # Speedup comparison
    ax1 = axes[0, 0]
    for i, lang in enumerate(languages):
        df_lang = df[df['language'] == lang]
        speedups = [df_lang[df_lang['benchmark'] == b]['speedup'].max()
                    if not df_lang[df_lang['benchmark'] == b].empty else 0
                    for b in benchmarks]
        offset = (i - len(languages) / 2 + 0.5) * width
        ax1.bar(x + offset, speedups, width, label=lang.capitalize(),
                color=get_language_color(lang), edgecolor='white', linewidth=1)
    ax1.set_ylabel('Best Speedup')
    ax1.set_title('Speedup Comparison', color=COLORS['neon_green'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.5)

    # Time comparison
    ax2 = axes[0, 1]
    for i, lang in enumerate(languages):
        df_lang = df[df['language'] == lang]
        times = [df_lang[df_lang['benchmark'] == b]['min_ms'].min()
                 if not df_lang[df_lang['benchmark'] == b].empty else 0
                 for b in benchmarks]
        offset = (i - len(languages) / 2 + 0.5) * width
        ax2.bar(x + offset, times, width, label=lang.capitalize(),
                color=get_language_color(lang), edgecolor='white', linewidth=1)
    ax2.set_ylabel('Min Time (ms)')
    ax2.set_title('Execution Time (lower is better)', color=COLORS['neon_orange'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax2.legend()

    # GFLOP/s comparison
    if 'gflops' in df.columns:
        ax3 = axes[1, 0]
        for i, lang in enumerate(languages):
            df_lang = df[df['language'] == lang]
            gflops = [df_lang[df_lang['benchmark'] == b]['gflops'].max()
                      if not df_lang[df_lang['benchmark'] == b].empty else 0
                      for b in benchmarks]
            offset = (i - len(languages) / 2 + 0.5) * width
            ax3.bar(x + offset, gflops, width, label=lang.capitalize(),
                    color=get_language_color(lang), edgecolor='white', linewidth=1)
        ax3.set_ylabel('GFLOP/s')
        ax3.set_title('Throughput (higher is better)', color=COLORS['laser_gold'])
        ax3.set_xticks(x)
        ax3.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax3.legend()

    # Winner summary
    ax4 = axes[1, 1]
    winners = []
    for b in benchmarks:
        best_time = float('inf')
        winner = 'tie'
        for lang in languages:
            df_bl = df[(df['benchmark'] == b) & (df['language'] == lang)]
            if not df_bl.empty:
                t = df_bl['min_ms'].min()
                if t < best_time:
                    best_time = t
                    winner = lang
        winners.append(winner)

    julia_wins = sum(1 for w in winners if w == 'julia')
    openmp_wins = sum(1 for w in winners if w == 'openmp')
    
    labels = ['Julia', 'OpenMP']
    wins = [julia_wins, openmp_wins]
    colors = [LANGUAGE_COLORS['julia'], LANGUAGE_COLORS['openmp']]
    
    wedges, texts, autotexts = ax4.pie(wins, labels=labels, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'color': COLORS['text']})
    ax4.set_title(f'Wins by Language ({len(benchmarks)} benchmarks)', color=COLORS['neon_purple'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='OpenMP PolyBench Visualization - Neon Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv -o ./plots
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
    python3 visualize_benchmarks.py julia/*.csv openmp/*.csv --compare
        """
    )

    parser.add_argument('files', nargs='*', help='CSV files to visualize')
    parser.add_argument('-o', '--output-dir', default='./benchmark_plots', help='Output directory')
    parser.add_argument('-t', '--title', default='', help='Title prefix for all plots')
    parser.add_argument('--scaling', action='store_true', help='Focus on thread scaling plots')
    parser.add_argument('--compare', action='store_true', help='Julia vs OpenMP comparison mode')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap only')

    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        sys.exit(1)

    setup_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_multiple_csvs(args.files)
    if df is None or df.empty:
        print("No valid data loaded")
        sys.exit(1)

    print(f"Loaded {len(df)} records from {len(args.files)} file(s)")
    print(f"Columns: {', '.join(df.columns)}")
    if 'benchmark' in df.columns:
        print(f"Benchmarks: {', '.join(df['benchmark'].unique())}")
    if 'dataset' in df.columns:
        print(f"Datasets: {', '.join(df['dataset'].unique())}")
    if 'threads' in df.columns:
        print(f"Thread counts: {sorted(df['threads'].unique())}")
    if 'language' in df.columns:
        print(f"Languages: {', '.join(df['language'].unique())}")

    prefix = f"{args.title} " if args.title else ""

    # Generate plots
    summary_path = output_dir / generate_filename("summary", df)
    create_summary_dashboard(df, summary_path, prefix)

    heatmap_path = output_dir / generate_filename("heatmap", df)
    create_strategy_heatmap(df, heatmap_path, prefix)

    if args.scaling or ('threads' in df.columns and df['threads'].nunique() > 1):
        scaling_path = output_dir / generate_filename("scaling", df)
        create_thread_scaling_chart(df, scaling_path, prefix)

    if 'benchmark' in df.columns and df['benchmark'].nunique() > 1:
        compare_path = output_dir / generate_filename("comparison", df)
        create_benchmark_comparison(df, compare_path, prefix)

    if 'language' in df.columns and df['language'].nunique() > 1:
        lang_path = output_dir / generate_filename("julia_vs_openmp", df)
        create_julia_vs_openmp_comparison(df, lang_path, prefix)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
