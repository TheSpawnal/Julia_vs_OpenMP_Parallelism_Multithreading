#!/bin/bash
#===============================================================================
# DAS-5 EXTRALARGE Dataset Benchmark Script
# 
# For heavy workloads requiring extended time
# DAS-5 nodes have 16 cores (dual 8-core Xeon E5-2630-v3)
# 
# USAGE:
#   sbatch das5_extralarge.slurm [benchmark]
#   sbatch das5_extralarge.slurm 2mm
#   sbatch das5_extralarge.slurm 3mm
#   sbatch das5_extralarge.slurm all    # Run all benchmarks
#
# SUBMIT DURING OFF-HOURS for >15 minute jobs:
#   sbatch --begin=22:00 das5_extralarge.slurm all
#   sbatch --begin=saturday das5_extralarge.slurm all
#===============================================================================

#SBATCH --job-name=julia_XL
#SBATCH --output=julia_XL_%j.out
#SBATCH --error=julia_XL_%j.err
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH -C cpunode

# Source DAS-5 environment
. /etc/bashrc
. /etc/profile.d/lmod.sh

# Load modules
module load prun
module load julia/1.11.4 2>/dev/null || module load julia

# Verify Julia
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not available"
    exit 1
fi

# Environment configuration
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH="$HOME/.julia"

# Project paths
PROJECT_DIR="$HOME/Julia_versus_OpenMP/julia_polybench_refactored"
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Parse argument
BENCHMARK="${1:-all}"

# Header
echo "======================================================================"
echo "DAS-5 EXTRALARGE Dataset Benchmark"
echo "======================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Cores:      $SLURM_CPUS_PER_TASK"
echo "Threads:    $JULIA_NUM_THREADS"
echo "Julia:      $(julia --version)"
echo "Benchmark:  $BENCHMARK"
echo "Dataset:    EXTRALARGE"
echo "======================================================================"

cd "$PROJECT_DIR" || exit 1

# Configuration
DATASET="EXTRALARGE"
ITERATIONS=5      # Fewer iterations for large dataset
WARMUP=3

run_benchmark() {
    local bench=$1
    local script="scripts/run_${bench}.jl"
    
    if [[ -f "$script" ]]; then
        echo ""
        echo ">>> Running: $bench (EXTRALARGE)"
        echo "----------------------------------------------------------------------"
        
        julia -t "$JULIA_NUM_THREADS" "$script" \
            --dataset "$DATASET" \
            --iterations "$ITERATIONS" \
            --warmup "$WARMUP" \
            --output csv
    else
        echo "SKIP: $script not found"
    fi
}

if [[ "$BENCHMARK" == "all" ]]; then
    BENCHMARKS=("2mm" "3mm" "cholesky" "correlation" "jacobi2d" "nussinov")
    for bench in "${BENCHMARKS[@]}"; do
        run_benchmark "$bench"
    done
else
    run_benchmark "$BENCHMARK"
fi

echo ""
echo "======================================================================"
echo "EXTRALARGE benchmark complete"
echo "End:        $(date)"
echo "Results:    $RESULTS_DIR"
echo "======================================================================"

# Show generated files
echo ""
echo "Generated CSV files:"
ls -la "$RESULTS_DIR"/*EXTRALARGE*.csv 2>/dev/null | tail -10
