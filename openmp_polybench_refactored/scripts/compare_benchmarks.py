#!/usr/bin/env python3
"""
Julia vs OpenMP Benchmark Comparison Tool

Compares benchmark results from Julia and OpenMP implementations,
generating unified visualizations and performance reports.

CSV Format (both languages):
benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations

Usage:
    python3 compare_benchmarks.py julia_results/*.csv openmp_results/*.csv
    python3 compare_benchmarks.py --merge results/
    python3 compare_benchmarks.py julia.csv openmp.csv --output report.html
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

COLORS = {
    'julia': '#9558B2',
    'openmp': '#389826',
    'sequential': '#888888',
    'threads_static': '#1f77b4',
    'threads_dynamic': '#ff7f0e',
    'tiled': '#2ca02c',
    'tasks': '#d62728',
    'simd': '#9467bd',
    'blas': '#8c564b',
    'wavefront': '#e377c2',
    'collapsed': '#7f7f7f'
}

def detect_language(filepath):
    """Detect language from filepath or CSV content."""
    fname = Path(filepath).stem.lower()
    if 'julia' in fname:
        return 'julia'
    if 'openmp' in fname or 'omp' in fname:
        return 'openmp'
    
    try:
        df = pd.read_csv(filepath, nrows=1)
        if 'language' in df.columns:
            return df['language'].iloc[0]
    except:
        pass
    
    return 'openmp'

def load_csv(filepath):
    """Load and normalize a benchmark CSV file."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
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
    
    if 'language' not in df.columns:
        df['language'] = detect_language(filepath)
    
    df['source_file'] = Path(filepath).stem
    
    return df

def load_multiple_csvs(filepaths):
    """Load and merge multiple CSV files."""
    dfs = []
    for fp in filepaths:
        df = load_csv(fp)
        if df is not None and not df.empty:
            dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def compute_comparison_metrics(df):
    """Compute comparison metrics between Julia and OpenMP."""
    results = []
    
    benchmarks = df['benchmark'].unique() if 'benchmark' in df.columns else ['benchmark']
    datasets = df['dataset'].unique() if 'dataset' in df.columns else ['LARGE']
    strategies = df['strategy'].unique() if 'strategy' in df.columns else ['sequential']
    
    for benchmark in benchmarks:
        for dataset in datasets:
            for strategy in strategies:
                mask = (df['benchmark'] == benchmark) & \
                       (df['dataset'] == dataset) & \
                       (df['strategy'] == strategy)
                
                julia_data = df[mask & (df['language'] == 'julia')]
                openmp_data = df[mask & (df['language'] == 'openmp')]
                
                if julia_data.empty or openmp_data.empty:
                    continue
                
                julia_time = julia_data['min_ms'].min()
                openmp_time = openmp_data['min_ms'].min()
                
                relative_perf = julia_time / openmp_time if openmp_time > 0 else float('inf')
                
                results.append({
                    'benchmark': benchmark,
                    'dataset': dataset,
                    'strategy': strategy,
                    'julia_min_ms': julia_time,
                    'openmp_min_ms': openmp_time,
                    'julia_gflops': julia_data['gflops'].max(),
                    'openmp_gflops': openmp_data['gflops'].max(),
                    'relative_perf': relative_perf,
                    'winner': 'Julia' if julia_time < openmp_time else 'OpenMP'
                })
    
    return pd.DataFrame(results)

def print_comparison_table(df):
    """Print a formatted comparison table."""
    comparison = compute_comparison_metrics(df)
    
    if comparison.empty:
        print("No comparable data found between Julia and OpenMP.")
        return
    
    print("\n" + "=" * 100)
    print("JULIA VS OPENMP PERFORMANCE COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Benchmark':<15} {'Dataset':<10} {'Strategy':<18} {'Julia(ms)':<12} {'OpenMP(ms)':<12} {'Ratio':<8} {'Winner':<8}")
    print("-" * 100)
    
    for _, row in comparison.iterrows():
        ratio_str = f"{row['relative_perf']:.2f}x"
        print(f"{row['benchmark']:<15} {row['dataset']:<10} {row['strategy']:<18} "
              f"{row['julia_min_ms']:<12.3f} {row['openmp_min_ms']:<12.3f} "
              f"{ratio_str:<8} {row['winner']:<8}")
    
    print("\n" + "-" * 100)
    
    julia_wins = (comparison['winner'] == 'Julia').sum()
    openmp_wins = (comparison['winner'] == 'OpenMP').sum()
    avg_ratio = comparison['relative_perf'].mean()
    
    print(f"Summary: Julia wins {julia_wins}, OpenMP wins {openmp_wins}")
    print(f"Average Julia/OpenMP ratio: {avg_ratio:.2f}x")
    if avg_ratio < 1.0:
        print(f"Julia is on average {1/avg_ratio:.2f}x faster than OpenMP")
    else:
        print(f"OpenMP is on average {avg_ratio:.2f}x faster than Julia")
    print()

def print_strategy_summary(df):
    """Print summary by strategy."""
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for lang in df['language'].unique():
        df_lang = df[df['language'] == lang]
        print(f"\n{lang.upper()}:")
        print(f"{'Strategy':<20} {'Min(ms)':<12} {'Median(ms)':<12} {'GFLOP/s':<12} {'Speedup':<10}")
        print("-" * 70)
        
        for strategy in df_lang['strategy'].unique():
            df_strat = df_lang[df_lang['strategy'] == strategy]
            print(f"{strategy:<20} {df_strat['min_ms'].min():<12.3f} "
                  f"{df_strat['median_ms'].mean():<12.3f} "
                  f"{df_strat['gflops'].max():<12.2f} "
                  f"{df_strat['speedup'].max():<10.2f}")

def export_merged_csv(df, output_path):
    """Export merged results to a single CSV."""
    df.to_csv(output_path, index=False)
    print(f"Merged CSV exported: {output_path}")

def generate_markdown_report(df, output_path):
    """Generate a Markdown report."""
    comparison = compute_comparison_metrics(df)
    
    with open(output_path, 'w') as f:
        f.write("# Julia vs OpenMP Benchmark Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        if not comparison.empty:
            julia_wins = (comparison['winner'] == 'Julia').sum()
            openmp_wins = (comparison['winner'] == 'OpenMP').sum()
            avg_ratio = comparison['relative_perf'].mean()
            
            f.write(f"- Julia wins: {julia_wins}\n")
            f.write(f"- OpenMP wins: {openmp_wins}\n")
            f.write(f"- Average Julia/OpenMP ratio: {avg_ratio:.2f}x\n\n")
        
        f.write("## Detailed Comparison\n\n")
        f.write("| Benchmark | Dataset | Strategy | Julia (ms) | OpenMP (ms) | Ratio | Winner |\n")
        f.write("|-----------|---------|----------|------------|-------------|-------|--------|\n")
        
        for _, row in comparison.iterrows():
            f.write(f"| {row['benchmark']} | {row['dataset']} | {row['strategy']} | "
                   f"{row['julia_min_ms']:.3f} | {row['openmp_min_ms']:.3f} | "
                   f"{row['relative_perf']:.2f}x | {row['winner']} |\n")
        
        f.write("\n## Raw Results\n\n")
        f.write("| Language | Benchmark | Dataset | Strategy | Threads | Min (ms) | GFLOP/s | Speedup | Verified |\n")
        f.write("|----------|-----------|---------|----------|---------|----------|---------|---------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row.get('language', 'N/A')} | {row.get('benchmark', 'N/A')} | "
                   f"{row.get('dataset', 'N/A')} | {row.get('strategy', 'N/A')} | "
                   f"{row.get('threads', 'N/A')} | {row.get('min_ms', 0):.3f} | "
                   f"{row.get('gflops', 0):.2f} | {row.get('speedup', 0):.2f} | "
                   f"{row.get('verified', 'N/A')} |\n")
    
    print(f"Markdown report: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare Julia and OpenMP benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 compare_benchmarks.py julia_results/*.csv openmp_results/*.csv
    python3 compare_benchmarks.py results/ --output comparison.csv
    python3 compare_benchmarks.py *.csv --report report.md
        """
    )
    
    parser.add_argument('files', nargs='*', help='CSV files or directories')
    parser.add_argument('-o', '--output', help='Output merged CSV file')
    parser.add_argument('-r', '--report', help='Generate Markdown report')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    
    args = parser.parse_args()
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    csv_files = []
    for path in args.files:
        p = Path(path)
        if p.is_dir():
            csv_files.extend(p.glob('**/*.csv'))
        elif p.suffix == '.csv':
            csv_files.append(p)
    
    if not csv_files:
        print("No CSV files found")
        sys.exit(1)
    
    print(f"Loading {len(csv_files)} CSV file(s)...")
    df = load_multiple_csvs([str(f) for f in csv_files])
    
    if df is None or df.empty:
        print("No valid data loaded")
        sys.exit(1)
    
    print(f"Loaded {len(df)} records")
    print(f"Languages: {', '.join(df['language'].unique())}")
    print(f"Benchmarks: {', '.join(df['benchmark'].unique())}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    
    if args.summary:
        print_strategy_summary(df)
    else:
        print_comparison_table(df)
        print_strategy_summary(df)
    
    if args.output:
        export_merged_csv(df, args.output)
    
    if args.report:
        generate_markdown_report(df, args.report)

if __name__ == '__main__':
    main()
