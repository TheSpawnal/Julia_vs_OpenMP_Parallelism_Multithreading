#!/bin/bash
# ============================================================================
# local_scaling.sh -- WSL2 pre-deployment scaling study
#
# Runs each benchmark across thread counts, merging results into a single
# CSV per benchmark with a metadata block matching scripts/bench_io.py.
#
# Metadata format: `# key=value; key=value; ...` on the first comment lines.
# Values are sanitized (no ',', ';', '=', '#', quotes, or whitespace).
#
# Usage:
#   ./local_scaling.sh                    # MEDIUM dataset, all benchmarks
#   ./local_scaling.sh LARGE              # LARGE dataset, all benchmarks
#   ./local_scaling.sh MEDIUM 2mm         # MEDIUM, only 2mm
#   ./local_scaling.sh MEDIUM all quick   # 3 iterations (fast)
# ============================================================================

set -euo pipefail

DATASET="${1:-MEDIUM}"
BENCH_FILTER="${2:-all}"
MODE="${3:-full}"

# ---- helpers ---------------------------------------------------------------
# Replace chars forbidden by our CSV metadata convention.
sanitize() {
    # tr maps each char in set1 to the same-index char in set2; both sets must
    # match in length. Forbidden: ; , = # " ' <space> <tab> <newline> <cr>
    printf '%s' "${1-}" | tr ';,=#"'"'"' \t\n\r' '__________'
}

# ---- platform probing ------------------------------------------------------
HOSTNAME_RAW="$(hostname 2>/dev/null || echo unknown)"
PHYS_CORES="$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2; exit}')"
PHYS_CORES="${PHYS_CORES:-$(nproc --all 2>/dev/null || echo 0)}"
LOGICAL="$(nproc 2>/dev/null || echo 0)"
CPU_RAW="$(lscpu 2>/dev/null | awk -F: '/^Model name/{sub(/^ +/,"",$2); print $2; exit}')"
CPU_RAW="${CPU_RAW:-unknown}"
OS_RAW="$(uname -sr 2>/dev/null | tr ' ' '-')"
COMPILER_RAW="$(gcc -dumpversion 2>/dev/null || echo unknown)"
COMPILER_NAME="gcc"
GIT_COMMIT_RAW="$(git -C "${SCRIPT_DIR:-.}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
RUN_DATE_RAW="$(date -Is 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"

# Sanitize everything that goes into the metadata line.
HOST=$(sanitize "$HOSTNAME_RAW")
CPU=$(sanitize "$CPU_RAW")
OS=$(sanitize "$OS_RAW")
COMPILER=$(sanitize "$COMPILER_NAME")
CC_VER=$(sanitize "$COMPILER_RAW")
GIT_COMMIT=$(sanitize "$GIT_COMMIT_RAW")
RUN_DATE=$(sanitize "$RUN_DATE_RAW")

# Thread counts: 1 -> logical cap
THREAD_COUNTS="1"
for t in 2 4 8 16; do
    [ "$t" -le "${LOGICAL:-1}" ] && THREAD_COUNTS="$THREAD_COUNTS $t"
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Pin threads for reproducible results
export OMP_PROC_BIND=close
export OMP_PLACES=cores

OMP_BIND_SAFE=$(sanitize "${OMP_PROC_BIND-}")
OMP_PLACES_SAFE=$(sanitize "${OMP_PLACES-}")
OMP_SCHED_SAFE=$(sanitize "${OMP_SCHEDULE-unset}")

echo "host=$HOST  cpu=$CPU  cores=$LOGICAL (phys=${PHYS_CORES:-?})"
echo "compiler=${COMPILER}-${CC_VER}  omp_bind=$OMP_BIND_SAFE  omp_places=$OMP_PLACES_SAFE"
echo "dataset=$DATASET  threads=$THREAD_COUNTS  iterations=$ITERATIONS (warmup=$WARMUP)  mode=$MODE"
echo "git=$GIT_COMMIT  date=$RUN_DATE"
echo

for BENCH in $BENCHMARKS; do
    BINARY="./benchmark_${BENCH}"
    if [ ! -x "$BINARY" ]; then
        echo "[SKIP] $BINARY not found -- run 'make' first"
        continue
    fi

    OUTFILE="${RESULTS_DIR}/scaling_${BENCH}_${DATASET}_${TIMESTAMP}.csv"

    # ---- Metadata block (matches scripts/bench_io.py schema=1) -----------
    {
        printf '# schema=1; language=openmp; host=%s; os=%s; cpu=%s; cores_logical=%s; cores_physical=%s; compiler=%s; compiler_version=%s; run_date=%s; project=openmp_polybench_refactored; git_commit=%s\n' \
            "$HOST" "$OS" "$CPU" "${LOGICAL:-0}" "${PHYS_CORES:-0}" \
            "$COMPILER" "$CC_VER" "$RUN_DATE" "$GIT_COMMIT"
        printf '# benchmark=%s; dataset=%s; iterations=%d; warmup=%d; mode=%s; omp_proc_bind=%s; omp_places=%s; omp_schedule=%s\n' \
            "$BENCH" "$DATASET" "$ITERATIONS" "$WARMUP" "$MODE" \
            "$OMP_BIND_SAFE" "$OMP_PLACES_SAFE" "$OMP_SCHED_SAFE"
        printf 'benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations\n'
    } > "$OUTFILE"

    echo "--- $BENCH ($DATASET) ---"

    for T in $THREAD_COUNTS; do
        export OMP_NUM_THREADS="$T"
        printf "  threads=%-2d  " "$T"

        # Run the benchmark. It emits its own CSV with metadata + header.
        if ! "$BINARY" --dataset "$DATASET" --threads "$T" \
                       --iterations "$ITERATIONS" --warmup "$WARMUP" \
                       --output csv >/dev/null 2>&1 ; then
            printf "ERROR: benchmark exited nonzero\n"
            continue
        fi

        # Locate the most recent per-run CSV produced by the binary.
        TEMP="$(ls -1t "${RESULTS_DIR}/${BENCH}_${DATASET}_"*.csv 2>/dev/null | head -1 || true)"
        if [ -n "${TEMP:-}" ] && [ -f "$TEMP" ]; then
            # Append data rows only: drop '#'-comment lines and the header.
            grep -v '^#' "$TEMP" | tail -n +2 >> "$OUTFILE"
            PASS_COUNT="$(grep -c ',PASS,' "$TEMP" 2>/dev/null || true)"
            FAIL_COUNT="$(grep -c ',FAIL,' "$TEMP" 2>/dev/null || true)"
            printf "PASS=%s FAIL=%s\n" "${PASS_COUNT:-0}" "${FAIL_COUNT:-0}"
            rm -f "$TEMP"
        else
            printf "ERROR: no CSV produced\n"
        fi
    done

    ROW_COUNT="$(grep -cv '^#\|^benchmark,' "$OUTFILE" 2>/dev/null || true)"
    echo "  => $OUTFILE (${ROW_COUNT:-0} data rows)"
    echo ""
done

echo "done. results in $RESULTS_DIR/"
echo "visualize:  python3 scripts/visualize_benchmarks.py results/scaling_*.csv"
echo "inspect:    head -3 results/scaling_*.csv   # metadata + header"
