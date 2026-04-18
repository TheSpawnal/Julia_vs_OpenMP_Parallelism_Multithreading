#!/bin/bash
#===============================================================================
# DAS-5 OpenMP Benchmark Runner
#
# USAGE:
#   sbatch das5_benchmark.slurm [benchmark] [dataset]
#   sbatch das5_benchmark.slurm 2mm LARGE
#   sbatch das5_benchmark.slurm correlation MEDIUM
#===============================================================================

#SBATCH --job-name=openmp_bench
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --time=00:10:00
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
THREADS=16
ITERATIONS=10
WARMUP=3

PROJECT_DIR="$HOME/openmp_polybench_refactored"
cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "========================================"
echo "OpenMP Benchmark: $BENCHMARK"
echo "Dataset: $DATASET"
echo "Threads: $THREADS"
echo "Node: $(hostname)"
echo "========================================"

./benchmark_${BENCHMARK} --dataset $DATASET --threads $THREADS \
    --iterations $ITERATIONS --warmup $WARMUP --output csv

echo "Completed at $(date)"
