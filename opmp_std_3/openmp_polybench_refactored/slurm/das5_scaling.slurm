#!/bin/bash
#===============================================================================
# DAS-5 Thread Scaling Study
#
# Runs benchmark across thread counts (1, 2, 4, 8, 16)
#
# USAGE:
#   sbatch das5_scaling.slurm [benchmark] [dataset]
#   sbatch das5_scaling.slurm 2mm LARGE
#===============================================================================

#SBATCH --job-name=scaling_study
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH -C cpunode

. /etc/bashrc
. /etc/profile.d/lmod.sh
module load prun

BENCHMARK="${1:-2mm}"
DATASET="${2:-LARGE}"
THREAD_COUNTS="1 2 4 8 16"
ITERATIONS=10
WARMUP=3

PROJECT_DIR="$HOME/openmp_polybench_refactored"
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_DIR" || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$RESULTS_DIR/scaling_${BENCHMARK}_${DATASET}_${TIMESTAMP}.csv"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "========================================"
echo "Scaling Study: $BENCHMARK"
echo "Dataset: $DATASET"
echo "Thread counts: $THREAD_COUNTS"
echo "Node: $(hostname)"
echo "========================================"

echo "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations" > "$OUTPUT"

for T in $THREAD_COUNTS; do
    echo ""
    echo "--- Threads: $T ---"
    export OMP_NUM_THREADS=$T
    
    ./benchmark_${BENCHMARK} --dataset $DATASET --threads $T \
        --iterations $ITERATIONS --warmup $WARMUP --output csv 2>/dev/null
    
    TEMP_CSV=$(ls -t results/${BENCHMARK}_${DATASET}_*.csv 2>/dev/null | head -1)
    if [ -f "$TEMP_CSV" ]; then
        tail -n +2 "$TEMP_CSV" >> "$OUTPUT"
    fi
done

echo ""
echo "Combined results: $OUTPUT"
echo "Completed at $(date)"
