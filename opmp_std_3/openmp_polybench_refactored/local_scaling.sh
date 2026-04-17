#!/bin/bash
# ============================================================================
# local_scaling.sh -- WSL2 pre-deployment scaling study
#
# Runs each benchmark across thread counts, merging results into a single
# CSV per benchmark (with machine metadata). Designed for i5-10210U (4C/8T)
# but auto-detects physical cores if possible.
#
# Usage:
#   ./local_scaling.sh                    # MEDIUM dataset, all benchmarks
#   ./local_scaling.sh LARGE              # LARGE dataset, all benchmarks
#   ./local_scaling.sh MEDIUM 2mm         # MEDIUM, only 2mm
#   ./local_scaling.sh MEDIUM all quick   # MEDIUM, 3 iterations (fast)
# ============================================================================

set -euo pipefail

DATASET="${1:-MEDIUM}"
BENCH_FILTER="${2:-all}"
MODE="${3:-full}"

# Detect cores
PHYS_CORES=$(lscpu 2>/dev/null | awk '/^Core\(s\) per socket:/{print $4}' || echo 4)
LOGICAL=$(nproc 2>/dev/null || echo 8)

# Thread counts: 1 -> physical cores -> logical (HT)
THREAD_COUNTS="1"
for t in 2 4 8 16; do
    [ "$t" -le "$LOGICAL" ] && THREAD_COUNTS="$THREAD_COUNTS $t"
done

if [ "$MODE" = "quick" ]; then
    ITERATIONS=3
    WARMUP=1
else
    ITERATIONS=7
    WARMUP=3
fi

BENCHMARKS="2mm 3mm cholesky correlation nussinov heat3d"
if [ "$BENCH_FILTER" != "all" ]; then
    BENCHMARKS="$BENCH_FILTER"
fi

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)
COMPILER=$(gcc --version 2>/dev/null | head -1 || echo "unknown")
CPU=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "unknown")

# Pin threads for reproducible results
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "================================================================"
echo "  Local Scaling Study"
echo "  Dataset:    $DATASET"
echo "  Benchmarks: $BENCHMARKS"
echo "  Threads:    $THREAD_COUNTS"
echo "  Iterations: $ITERATIONS (warmup: $WARMUP)"
echo "  Host:       $HOSTNAME"
echo "  CPU:        $CPU ($PHYS_CORES cores, $LOGICAL threads)"
echo "  Compiler:   $COMPILER"
echo "================================================================"
echo ""

for BENCH in $BENCHMARKS; do
    BINARY="./benchmark_${BENCH}"
    if [ ! -x "$BINARY" ]; then
        echo "[SKIP] $BINARY not found -- run 'make' first"
        continue
    fi

    OUTFILE="${RESULTS_DIR}/scaling_${BENCH}_${DATASET}_${TIMESTAMP}.csv"

    # Write metadata + header
    echo "# host=${HOSTNAME},cpu=${CPU},compiler=${COMPILER}" > "$OUTFILE"
    echo "# dataset=${DATASET},mode=${MODE},date=${TIMESTAMP}" >> "$OUTFILE"
    echo "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations" >> "$OUTFILE"

    echo "--- $BENCH ($DATASET) ---"

    for T in $THREAD_COUNTS; do
        export OMP_NUM_THREADS=$T

        printf "  threads=%-2d  " "$T"

        $BINARY --dataset "$DATASET" --threads "$T" \
                --iterations "$ITERATIONS" --warmup "$WARMUP" \
                --output csv 2>/dev/null

        # Find the CSV just written (most recent)
        TEMP=$(ls -t ${RESULTS_DIR}/${BENCH}_${DATASET}_*.csv 2>/dev/null | head -1)
        if [ -f "$TEMP" ]; then
            # Append data rows only (skip comments and header)
            grep -v '^#' "$TEMP" | tail -n +2 >> "$OUTFILE"
            PASS_COUNT=$(grep -c "PASS" "$TEMP" || true)
            FAIL_COUNT=$(grep -c "FAIL" "$TEMP" || true)
            printf "PASS=%d FAIL=%d\n" "$PASS_COUNT" "$FAIL_COUNT"
            rm -f "$TEMP"
        else
            printf "ERROR: no CSV produced\n"
        fi
    done

    ROW_COUNT=$(grep -cv '^#\|^benchmark' "$OUTFILE" || true)
    echo "  => $OUTFILE ($ROW_COUNT data rows)"
    echo ""
done

echo "================================================================"
echo "  Done. All scaling CSVs in $RESULTS_DIR/"
echo ""
echo "  Quick analysis:"
echo "    column -t -s, results/scaling_*.csv | less -S"
echo ""
echo "  Or with Python:"
echo "    python3 scripts/visualize_benchmarks.py results/scaling_*.csv"
echo "================================================================"