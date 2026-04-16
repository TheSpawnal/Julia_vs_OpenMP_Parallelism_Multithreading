#!/usr/bin/env python3
"""
Julia PolyBench Visualization Suite v2.0
NEON CYBERPUNK EDITION - Flashy Fluorescent Colors

Designed for Julia vs OpenMP performance comparison research.
Works EXCLUSIVELY with CSV data.

Features:
- Explicit file naming: benchmark_dataset_threads_node_timestamp
- Clear legends with full context
- Thread scaling with Amdahl's Law overlays and explanation
- Strategy comparison heatmaps
- Multi-benchmark dashboard
- OpenMP comparison ready (language column support)

Usage:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv -o ./plots --title "DAS-5 16-core"
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
    python3 visualize_benchmarks.py --compare julia_results/*.csv openmp_results/*.csv

Author: SpawnAl / Falkor collaboration
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
import re

# =============================================================================
# NEON CYBERPUNK COLOR PALETTE - FLASHY FLUORESCENT
# =============================================================================
COLORS = {
    # Neon primary - electric fluorescent
    'neon_cyan': '#00FFFF',       # Electric cyan
    'neon_magenta': '#FF00FF',    # Hot magenta
    'neon_green': '#39FF14',      # Radioactive green
    'neon_orange': '#FF6600',     # Blazing orange
    'neon_yellow': '#FFFF00',     # Electric yellow
    'neon_pink': '#FF1493',       # Deep pink
    'neon_blue': '#00BFFF',       # Deep sky blue
    'neon_purple': '#BF00FF',     # Electric purple
    
    # Secondary neon
    'electric_lime': '#CCFF00',   # Lime
    'hot_coral': '#FF4040',       # Hot coral
    'plasma_violet': '#9D00FF',   # Plasma violet
    'cyber_teal': '#00CED1',      # Dark turquoise
    'fire_red': '#FF3030',        # Fiery red
    'laser_gold': '#FFD700',      # Gold
    
    # Backgrounds
    'dark_bg': '#0D1117',         # GitHub dark
    'panel_bg': '#161B22',        # Panel background
    'grid': '#30363D',            # Grid lines
    'text': '#E6EDF3',            # Light text
    'text_dim': '#8B949E',        # Dimmed text
}

# Strategy color mapping - high contrast neon
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
    'redblack': COLORS['hot_coral'],
    'colmajor': COLORS['electric_lime'],
    # OpenMP strategies
    'omp_static': COLORS['neon_pink'],
    'omp_dynamic': COLORS['plasma_violet'],
    'omp_guided': COLORS['cyber_teal'],
}

# Benchmark colors
BENCHMARK_COLORS = {
    '2mm': COLORS['neon_cyan'],
    '3mm': COLORS['neon_magenta'],
    'cholesky': COLORS['neon_green'],
    'correlation': COLORS['laser_gold'],
    'jacobi2d': COLORS['neon_purple'],
    'nussinov': COLORS['neon_orange'],
    'gemm': COLORS['neon_pink'],
    'syrk': COLORS['cyber_teal'],
}

# Language colors for Julia vs OpenMP
LANGUAGE_COLORS = {
    'julia': COLORS['neon_green'],
    'openmp': COLORS['neon_orange'],
    'c': COLORS['neon_blue'],
}

def get_strategy_color(strategy):
    return STRATEGY_COLORS.get(strategy.lower(), COLORS['text_dim'])

def get_benchmark_color(benchmark):
    return BENCHMARK_COLORS.get(benchmark.lower(), COLORS['neon_blue'])

def get_language_color(lang):
    return LANGUAGE_COLORS.get(lang.lower(), COLORS['text'])

# =============================================================================
# MATPLOTLIB DARK NEON STYLE
# =============================================================================
def setup_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['dark_bg'],
        'figure.edgecolor': COLORS['grid'],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.facecolor': COLORS['dark_bg'],
        'savefig.edgecolor': COLORS['grid'],
        
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
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        
        'legend.frameon': True,
        'legend.facecolor': COLORS['panel_bg'],
        'legend.edgecolor': COLORS['grid'],
        'legend.fontsize': 9,
        'legend.labelcolor': COLORS['text'],
        
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        
        'text.color': COLORS['text'],
    })

# =============================================================================
# DATA LOADING
# =============================================================================
def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    # Normalize column names
    column_map = {
        'min(ms)': 'min_ms',
        'median(ms)': 'median_ms', 
        'mean(ms)': 'mean_ms',
        'std(ms)': 'std_ms',
        'gflop/s': 'gflops',
        'eff(%)': 'efficiency_pct',
        'efficiency': 'efficiency_pct',
    }
    df.columns = [column_map.get(c.lower(), c.lower().replace(' ', '_')) for c in df.columns]
    
    # Extract metadata from filename if not in columns
    fname = Path(filepath).stem
    
    if 'benchmark' not in df.columns:
        match = re.match(r'(\d*mm|cholesky|correlation|jacobi2d|nussinov|gemm|syrk)', fname.lower())
        if match:
            df['benchmark'] = match.group(1)
    
    if 'dataset' not in df.columns:
        for size in ['EXTRALARGE', 'LARGE', 'MEDIUM', 'SMALL', 'MINI']:
            if size in fname.upper():
                df['dataset'] = size
                break
    
    # Extract thread count from filename if not present
    if 'threads' not in df.columns:
        match = re.search(r'(\d+)T', fname)
        if match:
            df['threads'] = int(match.group(1))
    
    # Extract hostname from filename
    if 'hostname' not in df.columns:
        match = re.search(r'_(\w+node\d+|\w+)_\d{8}', fname)
        if match:
            df['hostname'] = match.group(1)
    
    df['source_file'] = fname
    
    return df

def load_multiple_csvs(filepaths):
    dfs = []
    for fp in filepaths:
        df = load_csv(fp)
        if df is not None and not df.empty:
            dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

# =============================================================================
# FILE NAMING
# =============================================================================
def generate_filename(prefix, df, suffix=""):
    """Generate explicit filename with benchmark, dataset, threads, node info"""
    parts = [prefix]
    
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
    
    if 'hostname' in df.columns:
        hosts = df['hostname'].unique()
        if len(hosts) == 1:
            parts.append(str(hosts[0]))
    
    if 'language' in df.columns:
        langs = df['language'].unique()
        if len(langs) > 1:
            parts.append("vs".join(sorted(langs)))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    if suffix:
        parts.append(suffix)
    
    return "_".join(parts) + ".png"

# =============================================================================
# AMDAHL'S LAW
# =============================================================================
def amdahl_speedup(threads, parallel_fraction):
    """Theoretical maximum speedup: S = 1 / ((1-f) + f/p)"""
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / np.array(threads))

def amdahl_efficiency(threads, parallel_fraction):
    """Theoretical efficiency: E = S/p"""
    s = amdahl_speedup(threads, parallel_fraction)
    return s / np.array(threads) * 100

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_summary_dashboard(df, output_path, title_prefix=""):
    """4-panel summary dashboard with explicit legends"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{title_prefix}Benchmark Performance Summary', 
                 fontsize=16, fontweight='bold', color=COLORS['neon_cyan'])
    
    strategies = df['strategy'].unique()
    colors = [get_strategy_color(s) for s in strategies]
    y_pos = np.arange(len(strategies))
    
    # Build subtitle with metadata
    subtitle_parts = []
    if 'benchmark' in df.columns:
        subtitle_parts.append(f"Benchmark: {', '.join(df['benchmark'].unique())}")
    if 'dataset' in df.columns:
        subtitle_parts.append(f"Dataset: {', '.join(df['dataset'].unique())}")
    if 'threads' in df.columns:
        subtitle_parts.append(f"Threads: {sorted(df['threads'].unique())}")
    if 'hostname' in df.columns:
        subtitle_parts.append(f"Node: {', '.join(df['hostname'].unique())}")
    
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
        ax1.text(bar.get_width() + times.max()*0.02, bar.get_y() + bar.get_height()/2,
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
            ax2.text(bar.get_width() + speedups.max()*0.02, bar.get_y() + bar.get_height()/2,
                    f'{s:.2f}x', va='center', fontsize=9, color=COLORS['text'])
    
    # Panel 3: GFLOP/s
    if 'gflops' in df.columns:
        ax3 = fig.add_subplot(2, 2, 3)
        gflops = df.groupby('strategy')['gflops'].mean().reindex(strategies)
        bars3 = ax3.barh(y_pos, gflops, color=colors, edgecolor=COLORS['laser_gold'], linewidth=1)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(strategies)
        ax3.set_xlabel('GFLOP/s', color=COLORS['text'])
        ax3.set_title('Computational Throughput (higher is better)', color=COLORS['laser_gold'])
        ax3.invert_yaxis()
        for bar, g in zip(bars3, gflops):
            ax3.text(bar.get_width() + gflops.max()*0.02, bar.get_y() + bar.get_height()/2,
                    f'{g:.1f}', va='center', fontsize=9, color=COLORS['text'])
    
    # Panel 4: Parallel Efficiency (parallel strategies only)
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    if eff_col in df.columns:
        ax4 = fig.add_subplot(2, 2, 4)
        df_eff = df.copy()
        if df_eff[eff_col].dtype == object:
            df_eff = df_eff[df_eff[eff_col] != '']
            df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')
        df_eff = df_eff.dropna(subset=[eff_col])
        
        if not df_eff.empty:
            eff_by_strat = df_eff.groupby('strategy')[eff_col].mean()
            parallel_strats = [s for s in strategies if s in eff_by_strat.index]
            
            if parallel_strats:
                eff_vals = [eff_by_strat.get(s, 0) for s in parallel_strats]
                eff_colors = [get_strategy_color(s) for s in parallel_strats]
                y_pos_eff = np.arange(len(parallel_strats))
                
                bars4 = ax4.barh(y_pos_eff, eff_vals, color=eff_colors, 
                                edgecolor=COLORS['neon_purple'], linewidth=1)
                ax4.axvline(x=100, color=COLORS['neon_yellow'], linestyle='--', 
                           alpha=0.7, linewidth=2, label='100% (ideal)')
                ax4.set_yticks(y_pos_eff)
                ax4.set_yticklabels(parallel_strats)
                ax4.set_xlabel('Efficiency (%)', color=COLORS['text'])
                ax4.set_title('Parallel Efficiency (parallel strategies only)', color=COLORS['neon_purple'])
                ax4.invert_yaxis()
                
                for bar, e in zip(bars4, eff_vals):
                    ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                            f'{e:.1f}%', va='center', fontsize=9, color=COLORS['text'])
                
                # Add Amdahl note
                ax4.text(0.98, 0.02, "Note: Efficiency decreases with threads\ndue to Amdahl's Law (expected)",
                        transform=ax4.transAxes, fontsize=8, color=COLORS['text_dim'],
                        ha='right', va='bottom', style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_thread_scaling_chart(df, output_path, title_prefix=""):
    """Thread scaling with Amdahl's Law overlay and explanation"""
    if 'threads' not in df.columns:
        print("Skipping scaling chart: no 'threads' column")
        return
    
    thread_counts = sorted(df['threads'].unique())
    if len(thread_counts) < 2:
        print("Skipping scaling chart: need multiple thread counts")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Build title with metadata
    title_parts = [title_prefix, "Thread Scaling Analysis"]
    if 'benchmark' in df.columns:
        title_parts.insert(1, f"[{', '.join(df['benchmark'].unique())}]")
    if 'dataset' in df.columns:
        title_parts.insert(-1, f"- {', '.join(df['dataset'].unique())}")
    
    fig.suptitle(" ".join(title_parts), fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])
    
    parallel_strategies = [s for s in df['strategy'].unique() 
                          if any(x in s.lower() for x in ['thread', 'tiled', 'task', 'wavefront', 'omp'])]
    
    max_threads = max(thread_counts)
    threads_fine = np.linspace(1, max_threads, 100)
    
    # Plot 1: Speedup vs Threads
    ax1 = axes[0]
    for strategy in parallel_strategies:
        df_s = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_s) > 1 and 'speedup' in df_s.columns:
            ax1.plot(df_s['threads'], df_s['speedup'],
                    marker='o', color=get_strategy_color(strategy),
                    label=strategy, linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    # Ideal scaling
    ax1.plot([1, max_threads], [1, max_threads], '--', color=COLORS['neon_yellow'], 
            alpha=0.8, label='Ideal (linear)', linewidth=2)
    
    # Amdahl curves
    for p, label in [(0.90, '90%'), (0.95, '95%'), (0.99, '99%')]:
        ax1.plot(threads_fine, amdahl_speedup(threads_fine, p), ':', 
                color=COLORS['text_dim'], alpha=0.5, linewidth=1.5)
        ax1.text(max_threads * 1.02, amdahl_speedup(max_threads, p), 
                f'f={label}', fontsize=8, color=COLORS['text_dim'], va='center')
    
    ax1.set_xlabel('Number of Threads', color=COLORS['text'])
    ax1.set_ylabel('Speedup (S = T1/Tp)', color=COLORS['text'])
    ax1.set_title('Strong Scaling - Speedup', color=COLORS['neon_green'])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, max_threads * 1.15)
    ax1.set_ylim(0, max_threads * 1.1)
    
    # Plot 2: Efficiency vs Threads
    ax2 = axes[1]
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    
    if eff_col in df.columns:
        for strategy in parallel_strategies:
            df_s = df[df['strategy'] == strategy].sort_values('threads')
            df_s = df_s[df_s[eff_col].notna()]
            if df_s[eff_col].dtype == object:
                df_s = df_s[df_s[eff_col] != ''].copy()
                df_s[eff_col] = pd.to_numeric(df_s[eff_col], errors='coerce')
            df_s = df_s.dropna(subset=[eff_col])
            
            if len(df_s) > 1:
                ax2.plot(df_s['threads'], df_s[eff_col],
                        marker='s', color=get_strategy_color(strategy),
                        label=strategy, linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1)
        
        # Amdahl efficiency curves
        for p, label in [(0.90, '90%'), (0.95, '95%'), (0.99, '99%')]:
            ax2.plot(threads_fine, amdahl_efficiency(threads_fine, p), ':', 
                    color=COLORS['text_dim'], alpha=0.5, linewidth=1.5)
        
        ax2.axhline(y=100, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Number of Threads', color=COLORS['text'])
        ax2.set_ylabel('Efficiency (%) = (S/p) * 100', color=COLORS['text'])
        ax2.set_title('Parallel Efficiency (decreases with threads - expected)', color=COLORS['neon_purple'])
        ax2.set_xlim(0, max_threads * 1.1)
        ax2.set_ylim(0, 110)
    
    # Plot 3: Execution Time
    ax3 = axes[2]
    for strategy in parallel_strategies:
        df_s = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_s) > 1 and 'min_ms' in df_s.columns:
            ax3.plot(df_s['threads'], df_s['min_ms'],
                    marker='^', color=get_strategy_color(strategy),
                    label=strategy, linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    ax3.set_xlabel('Number of Threads', color=COLORS['text'])
    ax3.set_ylabel('Execution Time (ms)', color=COLORS['text'])
    ax3.set_title('Execution Time vs Threads', color=COLORS['neon_orange'])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(0, max_threads * 1.1)
    
    # Add Amdahl explanation
    fig.text(0.5, 0.02, 
             "Amdahl's Law: S_max = 1/((1-f) + f/p) where f = parallel fraction, p = threads. "
             "Efficiency E = S/p naturally decreases as p increases unless f = 100%.",
             ha='center', fontsize=9, color=COLORS['text_dim'], style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_strategy_heatmap(df, output_path, title_prefix=""):
    """Heatmap: Strategies x Metrics with clear labeling"""
    
    # Prepare data - strategy as rows, metrics as columns
    if 'strategy' not in df.columns:
        print("Skipping heatmap: no 'strategy' column")
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
        # Efficiency
        eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
        if eff_col in df.columns:
            df_eff = df_s.copy()
            if df_eff[eff_col].dtype == object:
                df_eff = df_eff[df_eff[eff_col] != '']
                df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')
            row['efficiency'] = df_eff[eff_col].mean() if not df_eff.empty else np.nan
        data.append(row)
    
    heatmap_df = pd.DataFrame(data).set_index('strategy')
    
    # Normalize each column 0-1 for color mapping
    heatmap_norm = heatmap_df.copy()
    for col in heatmap_norm.columns:
        col_min = heatmap_norm[col].min()
        col_max = heatmap_norm[col].max()
        if col_max > col_min:
            # For time, lower is better (invert)
            if 'ms' in col or 'time' in col.lower():
                heatmap_norm[col] = 1 - (heatmap_norm[col] - col_min) / (col_max - col_min)
            else:
                heatmap_norm[col] = (heatmap_norm[col] - col_min) / (col_max - col_min)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Custom neon colormap
    neon_cmap = LinearSegmentedColormap.from_list('neon',
        [COLORS['dark_bg'], COLORS['neon_purple'], COLORS['neon_cyan'], COLORS['neon_green']])
    
    im = ax.imshow(heatmap_norm.values, cmap=neon_cmap, aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels(['Time (ms)\n(lower=better)', 'Speedup\n(higher=better)', 
                        'GFLOP/s\n(higher=better)', 'Efficiency %\n(parallel only)'][:len(heatmap_df.columns)])
    ax.set_yticklabels(heatmap_df.index)
    
    # Annotate with actual values
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.values[i, j]
            if not np.isnan(val):
                # Format based on metric
                if j == 0:  # time
                    text = f'{val:.2f}'
                elif j == 1:  # speedup
                    text = f'{val:.2f}x'
                elif j == 2:  # gflops
                    text = f'{val:.1f}'
                else:  # efficiency
                    text = f'{val:.1f}%'
                
                # Text color based on background
                bg_val = heatmap_norm.values[i, j]
                text_color = COLORS['dark_bg'] if bg_val > 0.5 else COLORS['text']
                ax.text(j, i, text, ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
    
    # Title with metadata
    title = f'{title_prefix}Strategy Performance Heatmap'
    if 'benchmark' in df.columns:
        title += f" - {', '.join(df['benchmark'].unique())}"
    if 'dataset' in df.columns:
        title += f" [{', '.join(df['dataset'].unique())}]"
    ax.set_title(title, color=COLORS['neon_cyan'], fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (normalized 0-1, higher=better)', color=COLORS['text'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_benchmark_comparison(df, output_path, title_prefix=""):
    """Compare multiple benchmarks with clear identification"""
    if 'benchmark' not in df.columns:
        print("Skipping comparison: no 'benchmark' column")
        return
    
    benchmarks = df['benchmark'].unique()
    if len(benchmarks) < 2:
        print("Skipping comparison: need multiple benchmarks")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Build subtitle
    subtitle_parts = []
    if 'dataset' in df.columns:
        subtitle_parts.append(f"Dataset: {', '.join(df['dataset'].unique())}")
    if 'threads' in df.columns:
        subtitle_parts.append(f"Threads: {sorted(df['threads'].unique())}")
    if 'hostname' in df.columns:
        subtitle_parts.append(f"Node: {', '.join(df['hostname'].unique())}")
    
    fig.suptitle(f'{title_prefix}Multi-Benchmark Comparison', 
                 fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])
    if subtitle_parts:
        fig.text(0.5, 0.94, " | ".join(subtitle_parts), ha='center', 
                fontsize=10, color=COLORS['text_dim'])
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    # 1. Best speedup per benchmark
    ax1 = axes[0, 0]
    best_speedup = df.groupby('benchmark')['speedup'].max()
    colors = [get_benchmark_color(b) for b in best_speedup.index]
    bars = ax1.bar(x, best_speedup.values, color=colors, edgecolor='white', linewidth=1)
    ax1.axhline(y=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.7)
    ax1.set_ylabel('Best Speedup')
    ax1.set_title('Best Speedup by Benchmark', color=COLORS['neon_green'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    for bar, s in zip(bars, best_speedup.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{s:.2f}x', ha='center', fontsize=9, color=COLORS['text'])
    
    # 2. Best GFLOP/s per benchmark
    if 'gflops' in df.columns:
        ax2 = axes[0, 1]
        best_gflops = df.groupby('benchmark')['gflops'].max()
        colors = [get_benchmark_color(b) for b in best_gflops.index]
        bars = ax2.bar(x, best_gflops.values, color=colors, edgecolor='white', linewidth=1)
        ax2.set_ylabel('Best GFLOP/s')
        ax2.set_title('Peak Throughput by Benchmark', color=COLORS['laser_gold'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(benchmarks)
        for bar, g in zip(bars, best_gflops.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + best_gflops.max()*0.02,
                    f'{g:.1f}', ha='center', fontsize=9, color=COLORS['text'])
    
    # 3. Fastest strategy per benchmark
    ax3 = axes[1, 0]
    fastest = df.loc[df.groupby('benchmark')['min_ms'].idxmin()]
    strategies = fastest['strategy'].values
    colors = [get_strategy_color(s) for s in strategies]
    bars = ax3.bar(x, fastest['min_ms'].values, color=colors, edgecolor='white', linewidth=1)
    ax3.set_ylabel('Min Time (ms)')
    ax3.set_title('Fastest Execution by Benchmark', color=COLORS['neon_orange'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks)
    for bar, strat, t in zip(bars, strategies, fastest['min_ms'].values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + fastest['min_ms'].max()*0.02,
                f'{strat}\n{t:.1f}ms', ha='center', fontsize=8, color=COLORS['text'])
    
    # 4. Strategy breakdown per benchmark
    ax4 = axes[1, 1]
    strategies_all = df['strategy'].unique()
    x_offset = np.linspace(-0.3, 0.3, len(strategies_all))
    
    for i, strategy in enumerate(strategies_all):
        speedups = [df[(df['benchmark'] == b) & (df['strategy'] == strategy)]['speedup'].mean() 
                    if not df[(df['benchmark'] == b) & (df['strategy'] == strategy)].empty else 0 
                    for b in benchmarks]
        ax4.bar(x + x_offset[i], speedups, width=0.6/len(strategies_all), 
               color=get_strategy_color(strategy), label=strategy, edgecolor='white', linewidth=0.5)
    
    ax4.set_ylabel('Speedup')
    ax4.set_title('All Strategies by Benchmark', color=COLORS['neon_purple'])
    ax4.set_xticks(x)
    ax4.set_xticklabels(benchmarks)
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.axhline(y=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_julia_vs_openmp_comparison(df, output_path, title_prefix=""):
    """Side-by-side Julia vs OpenMP comparison (if language column exists)"""
    if 'language' not in df.columns:
        print("Skipping Julia vs OpenMP: no 'language' column")
        return
    
    languages = df['language'].unique()
    if len(languages) < 2:
        print("Skipping language comparison: need multiple languages")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    fig.suptitle(f'{title_prefix}Julia vs OpenMP Performance Comparison', 
                 fontsize=14, fontweight='bold', color=COLORS['neon_cyan'])
    
    benchmarks = df['benchmark'].unique() if 'benchmark' in df.columns else ['benchmark']
    x = np.arange(len(benchmarks))
    width = 0.35
    
    # 1. Speedup comparison
    ax1 = axes[0]
    for i, lang in enumerate(languages):
        df_lang = df[df['language'] == lang]
        speedups = [df_lang[df_lang['benchmark'] == b]['speedup'].max() 
                   if not df_lang[df_lang['benchmark'] == b].empty else 0 
                   for b in benchmarks]
        offset = (i - len(languages)/2 + 0.5) * width
        ax1.bar(x + offset, speedups, width, label=lang.capitalize(), 
               color=get_language_color(lang), edgecolor='white', linewidth=1)
    
    ax1.set_ylabel('Best Speedup')
    ax1.set_title('Speedup Comparison', color=COLORS['neon_green'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    ax1.legend()
    ax1.axhline(y=1.0, color=COLORS['neon_yellow'], linestyle='--', alpha=0.5)
    
    # 2. Execution time comparison
    ax2 = axes[1]
    for i, lang in enumerate(languages):
        df_lang = df[df['language'] == lang]
        times = [df_lang[df_lang['benchmark'] == b]['min_ms'].min() 
                if not df_lang[df_lang['benchmark'] == b].empty else 0 
                for b in benchmarks]
        offset = (i - len(languages)/2 + 0.5) * width
        ax2.bar(x + offset, times, width, label=lang.capitalize(),
               color=get_language_color(lang), edgecolor='white', linewidth=1)
    
    ax2.set_ylabel('Min Time (ms)')
    ax2.set_title('Execution Time (lower is better)', color=COLORS['neon_orange'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks)
    ax2.legend()
    
    # 3. GFLOP/s comparison
    if 'gflops' in df.columns:
        ax3 = axes[2]
        for i, lang in enumerate(languages):
            df_lang = df[df['language'] == lang]
            gflops = [df_lang[df_lang['benchmark'] == b]['gflops'].max() 
                     if not df_lang[df_lang['benchmark'] == b].empty else 0 
                     for b in benchmarks]
            offset = (i - len(languages)/2 + 0.5) * width
            ax3.bar(x + offset, gflops, width, label=lang.capitalize(),
                   color=get_language_color(lang), edgecolor='white', linewidth=1)
        
        ax3.set_ylabel('GFLOP/s')
        ax3.set_title('Throughput (higher is better)', color=COLORS['laser_gold'])
        ax3.set_xticks(x)
        ax3.set_xticklabels(benchmarks)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['dark_bg'])
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Julia PolyBench Visualization - Neon Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv -o ./plots --title "DAS-5 VU68"
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
    python3 visualize_benchmarks.py julia_results/*.csv openmp_results/*.csv --compare
        """
    )
    
    parser.add_argument('files', nargs='*', help='CSV files to visualize')
    parser.add_argument('-o', '--output-dir', default='./benchmark_plots', help='Output directory')
    parser.add_argument('-t', '--title', default='', help='Title prefix for all plots')
    parser.add_argument('--scaling', action='store_true', help='Focus on thread scaling plots')
    parser.add_argument('--compare', action='store_true', help='Multi-language comparison mode')
    parser.add_argument('--heatmap', action='store_true', help='Generate detailed heatmaps')
    
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
    
    prefix = f"{args.title} " if args.title else ""
    
    # Generate plots with explicit naming
    summary_path = output_dir / generate_filename("summary", df)
    create_summary_dashboard(df, summary_path, prefix)
    
    heatmap_path = output_dir / generate_filename("heatmap", df)
    create_strategy_heatmap(df, heatmap_path, prefix)
    
    if args.scaling or 'threads' in df.columns:
        if df['threads'].nunique() > 1:
            scaling_path = output_dir / generate_filename("scaling", df)
            create_thread_scaling_chart(df, scaling_path, prefix)
    
    if 'benchmark' in df.columns and df['benchmark'].nunique() > 1:
        compare_path = output_dir / generate_filename("comparison", df)
        create_benchmark_comparison(df, compare_path, prefix)
    
    if 'language' in df.columns and df['language'].nunique() > 1:
        lang_path = output_dir / generate_filename("julia_vs_openmp", df)
        create_julia_vs_openmp_comparison(df, lang_path, prefix)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nFile naming convention: {type}_{benchmark}_{dataset}_{threads}_{node}_{timestamp}.png")


if __name__ == '__main__':
    main()
