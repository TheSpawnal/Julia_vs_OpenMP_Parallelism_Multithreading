#!/bin/bash
#===============================================================================
# DAS-5 Comprehensive Scaling Study
#
# Runs benchmark across multiple thread counts (1, 2, 4, 8, 16)
# Produces single CSV with all scaling data for visualization
#
# USAGE:
#   sbatch das5_scaling_study.slurm [benchmark] [dataset]
#   sbatch das5_scaling_study.slurm 2mm LARGE
#   sbatch das5_scaling_study.slurm correlation MEDIUM
#
# OUTPUT:
#   results/scaling_study_{benchmark}_{dataset}_{timestamp}.csv
#===============================================================================

#SBATCH --job-name=scaling_study
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --time=00:15:00
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

# Environment
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH="$HOME/.julia"

# Parse arguments
BENCHMARK="${1:-2mm}"
DATASET="${2:-LARGE}"
THREAD_COUNTS="1 2 4 8 16"

# Project paths
PROJECT_DIR="$HOME/Julia_versus_OpenMP/julia_polybench_refactored"
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="$RESULTS_DIR/scaling_study_${BENCHMARK}_${DATASET}_${TIMESTAMP}.csv"

# Header
echo "======================================================================"
echo "DAS-5 Thread Scaling Study"
echo "======================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Benchmark:   $BENCHMARK"
echo "Dataset:     $DATASET"
echo "Threads:     $THREAD_COUNTS"
echo "Output:      $OUTPUT_CSV"
echo "======================================================================"

cd "$PROJECT_DIR" || exit 1

# Write CSV header
echo "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations" > "$OUTPUT_CSV"

# Script path
SCRIPT="scripts/run_${BENCHMARK}.jl"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

# Temporary file for each run
TEMP_CSV=$(mktemp)

# Run for each thread count
for THREADS in $THREAD_COUNTS; do
    echo ""
    echo ">>> Running with $THREADS thread(s)..."
    echo "----------------------------------------------------------------------"
    
    export JULIA_NUM_THREADS=$THREADS
    
    # Run benchmark
    julia -t "$THREADS" "$SCRIPT" \
        --dataset "$DATASET" \
        --iterations 10 \
        --warmup 5 \
        --output csv 2>&1
    
    # Find most recent CSV for this run
    LATEST=$(ls -t "$RESULTS_DIR"/${BENCHMARK}_${DATASET}_*.csv 2>/dev/null | head -1)
    
    if [[ -n "$LATEST" && -f "$LATEST" ]]; then
        # Append data rows (skip header)
        tail -n +2 "$LATEST" >> "$OUTPUT_CSV"
        echo "Appended data from: $LATEST"
        
        # Move individual file to temp to avoid duplication
        mv "$LATEST" "${LATEST}.bak" 2>/dev/null
    fi
done

# Clean up backup files
rm -f "$RESULTS_DIR"/${BENCHMARK}_${DATASET}_*.csv.bak 2>/dev/null

echo ""
echo "======================================================================"
echo "Scaling Study Complete"
echo "======================================================================"
echo "Output: $OUTPUT_CSV"
echo ""

# Print summary table
if [[ -f "$OUTPUT_CSV" ]]; then
    echo "Summary:"
    echo "----------------------------------------------------------------------"
    column -t -s',' "$OUTPUT_CSV" | head -30
fi
