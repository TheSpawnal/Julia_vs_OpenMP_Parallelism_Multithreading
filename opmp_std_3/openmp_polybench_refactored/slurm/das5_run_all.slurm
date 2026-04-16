#!/bin/bash
#===============================================================================
# DAS-5 Run All Benchmarks
#
# USAGE:
#   sbatch das5_run_all.slurm [dataset]
#   sbatch das5_run_all.slurm LARGE
#===============================================================================

#SBATCH --job-name=all_benchmarks
#SBATCH --output=all_bench_%j.out
#SBATCH --error=all_bench_%j.err
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH -C cpunode

. /etc/bashrc
. /etc/profile.d/lmod.sh
module load prun

DATASET="${1:-LARGE}"
THREADS=16
ITERATIONS=10
WARMUP=3

BENCHMARKS="2mm 3mm cholesky correlation nussinov"

PROJECT_DIR="$HOME/openmp_polybench_refactored"
cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "========================================"
echo "Running All OpenMP Benchmarks"
echo "Dataset: $DATASET | Threads: $THREADS"
echo "Node: $(hostname)"
echo "========================================"

for BENCH in $BENCHMARKS; do
    echo ""
    echo ">>> $BENCH <<<"
    ./benchmark_${BENCH} --dataset $DATASET --threads $THREADS \
        --iterations $ITERATIONS --warmup $WARMUP --output csv
done

echo ""
echo "All benchmarks completed at $(date)"
echo "Results in: $PROJECT_DIR/results/"
