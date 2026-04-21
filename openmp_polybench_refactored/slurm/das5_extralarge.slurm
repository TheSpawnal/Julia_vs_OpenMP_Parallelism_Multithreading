#!/bin/bash
#===============================================================================
# DAS-5 EXTRALARGE Dataset Benchmark Script (OpenMP)
#
# Time budget rationale:
#   - cholesky XL (n=4000):    ~7  min/strategy at 1 thread, ~30s at 16 threads
#   - 2mm      XL (~26 GFLOP): ~4  min/strategy at 1 thread
#   - 3mm      XL (~42 GFLOP): ~8  min/strategy at 1 thread
#   - correlation XL:          ~8  min/strategy at 1 thread
#   - nussinov XL (n=5500):    ~15 min/strategy at 1 thread
#   - heat3d   XL:             ~10 min/strategy at 1 thread
#   Per-benchmark budget at 16 threads: 5-15 min incl. warmup
#   Full sweep (6 benches): 60-90 min
#
# USAGE:
#   sbatch das5_extralarge.slurm [benchmark]
#   sbatch das5_extralarge.slurm all
#   sbatch das5_extralarge.slurm 2mm
#
# OFF-HOURS SUBMISSION (required for >15 min daytime):
#   sbatch --begin=22:00 das5_extralarge.slurm all
#   sbatch --begin=saturday das5_extralarge.slurm all
#
# ENV OVERRIDES:
#   SKIP_BROKEN=1   skip strategies quarantined as non-verifying (default)
#   SKIP_BROKEN=0   run all strategies including known-failing ones
#===============================================================================

#SBATCH --job-name=openmp_XL
#SBATCH --output=openmp_XL_%j.out
#SBATCH --error=openmp_XL_%j.err
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --partition=defq
#SBATCH -C cpunode

set -u
set -o pipefail

# DAS-5 environment
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load prun

# Config
BENCHMARK="${1:-all}"
DATASET="EXTRALARGE"
ITERATIONS="${ITERATIONS:-5}"
WARMUP="${WARMUP:-2}"
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
SKIP_BROKEN="${SKIP_BROKEN:-1}"

PROJECT_DIR="$HOME/latest/Julia_vs_OpenMP_Parallelism_Multithreading/openmp_polybench_refactored"
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

# OpenMP runtime tuning for Haswell dual-socket NUMA
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_STACKSIZE=256M

# Quarantined strategies (from local LARGE verification failures, 20260418 run)
# cholesky: threads_static, tiled, tasks all FAIL verification
# 3mm:      tasks FAIL at >= 4 threads
# nussinov: tiled FAIL (off-by-one)
declare -A QUARANTINE
QUARANTINE[cholesky]="threads_static tiled tasks"
QUARANTINE[3mm]="tasks"
QUARANTINE[nussinov]="tiled"

echo "=============================================================="
echo "DAS-5 EXTRALARGE Benchmark (OpenMP)"
echo "=============================================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Start:        $(date -Iseconds)"
echo "Cores:        $SLURM_CPUS_PER_TASK"
echo "Threads:      $THREADS"
echo "Iterations:   $ITERATIONS (warmup $WARMUP)"
echo "Benchmark:    $BENCHMARK"
echo "Skip broken:  $SKIP_BROKEN"
echo "Project:      $PROJECT_DIR"
echo "=============================================================="

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 2; }

# Verify binary exists and is executable
check_binary() {
    local bin="./benchmark_$1"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin missing or not executable. Run 'make das5' first."
        return 1
    fi
    return 0
}

# Run a single benchmark with optional strategy quarantine list
run_benchmark() {
    local bench="$1"
    local bin="./benchmark_${bench}"

    check_binary "$bench" || return 1

    local skip_list=""
    if [[ "$SKIP_BROKEN" == "1" && -n "${QUARANTINE[$bench]:-}" ]]; then
        skip_list="${QUARANTINE[$bench]}"
        echo ""
        echo "--- $bench (XL) --- quarantined: $skip_list"
    else
        echo ""
        echo "--- $bench (XL) ---"
    fi

    local start_t=$SECONDS
    # The benchmark binaries do not yet accept --skip-strategies.
    # Quarantine is enforced post-hoc by the filter script below.
    # For now we log the intent and the filter removes bad rows from the CSV.
    "$bin" \
        --dataset "$DATASET" \
        --threads "$THREADS" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output csv
    local rc=$?
    local elapsed=$((SECONDS - start_t))
    echo "--- $bench done in ${elapsed}s (rc=$rc)"

    # Post-hoc filter: remove quarantined rows from the most recent CSV
    if [[ "$SKIP_BROKEN" == "1" && -n "$skip_list" ]]; then
        local latest_csv
        latest_csv=$(ls -t "$RESULTS_DIR/${bench}_${DATASET}_"*.csv 2>/dev/null | head -1)
        if [[ -f "$latest_csv" ]]; then
            local tmp="${latest_csv}.filtered"
            local pattern
            pattern=$(echo "$skip_list" | tr ' ' '|')
            awk -v pat="$pattern" 'NR==1 || $0 !~ ","pat"," { print }' \
                FS=, OFS=, "$latest_csv" > "$tmp" && mv "$tmp" "$latest_csv"
            echo "    -> filtered quarantined strategies from $(basename "$latest_csv")"
        fi
    fi

    return $rc
}

# Dispatch
FAILED=0
if [[ "$BENCHMARK" == "all" ]]; then
    for b in 2mm 3mm cholesky correlation heat3d nussinov; do
        run_benchmark "$b" || FAILED=$((FAILED + 1))
    done
else
    run_benchmark "$BENCHMARK" || FAILED=1
fi

echo ""
echo "=============================================================="
echo "End:          $(date -Iseconds)"
echo "Failures:     $FAILED"
echo "Results dir:  $RESULTS_DIR"
echo "=============================================================="
ls -la "$RESULTS_DIR"/*"${DATASET}"*.csv 2>/dev/null | tail -20

exit $FAILED
