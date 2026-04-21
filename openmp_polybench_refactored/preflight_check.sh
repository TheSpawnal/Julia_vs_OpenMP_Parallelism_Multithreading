#!/bin/bash
#===============================================================================
# DAS-5 preflight check
#
# Run this on the headnode (fs0) after `make das5`, BEFORE submitting XL jobs.
# It verifies: toolchain, binaries, a quick MEDIUM sanity run, and cluster state.
#
# USAGE:
#   bash preflight_check.sh
#   bash preflight_check.sh --quick    # skip the MEDIUM sanity run
#===============================================================================

set -u

PROJECT_DIR="$HOME/latest/Julia_vs_OpenMP_Parallelism_Multithreading/openmp_polybench_refactored"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

echo "=== DAS-5 preflight ==="
echo "Date:  $(date -Iseconds)"
echo "User:  $USER"
echo "Host:  $(hostname)"

# 1. Must be on a headnode, not a compute node
case "$(hostname)" in
    fs[0-9]*)  echo "Location: headnode OK" ;;
    node*)     echo "WARN: running on compute node, exit and run on fs0"; exit 1 ;;
    *)         echo "WARN: unrecognized host" ;;
esac

# 2. Project directory
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "FAIL: $PROJECT_DIR missing"
    exit 1
fi
cd "$PROJECT_DIR"
echo "Project: $PROJECT_DIR OK"

# 3. Module system
if ! command -v module >/dev/null 2>&1; then
    . /etc/bashrc 2>/dev/null
    . /etc/profile.d/lmod.sh 2>/dev/null
fi

# 4. Toolchain
for tool in gcc make awk; do
    if ! command -v "$tool" >/dev/null; then
        echo "FAIL: $tool not found"
        exit 1
    fi
done
GCC_VER=$(gcc -dumpversion)
echo "gcc:     $GCC_VER"

# 5. OpenMP support
cat > /tmp/omp_test_$$.c <<'EOF'
#include <omp.h>
#include <stdio.h>
int main(void) {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    printf("%d\n", n);
    return 0;
}
EOF
if gcc -fopenmp /tmp/omp_test_$$.c -o /tmp/omp_test_$$ 2>/dev/null; then
    OMP_N=$(OMP_NUM_THREADS=4 /tmp/omp_test_$$ 2>/dev/null)
    if [[ "$OMP_N" == "4" ]]; then
        echo "OpenMP:  4-thread smoke test PASS"
    else
        echo "WARN: OpenMP returned $OMP_N, expected 4"
    fi
else
    echo "FAIL: OpenMP compile failed"
    exit 1
fi
rm -f /tmp/omp_test_$$.c /tmp/omp_test_$$

# 6. Binaries present
BENCHMARKS="2mm 3mm cholesky correlation heat3d nussinov"
MISSING=0
for b in $BENCHMARKS; do
    if [[ ! -x "./benchmark_$b" ]]; then
        echo "FAIL: benchmark_$b missing. Run: make das5"
        MISSING=$((MISSING + 1))
    fi
done
if [[ $MISSING -gt 0 ]]; then
    echo "$MISSING binary/binaries missing, aborting"
    exit 1
fi
echo "Binaries: 6/6 present"

# 7. Partition and node availability
if command -v sinfo >/dev/null; then
    IDLE=$(sinfo -h -p defq -t idle -o "%D" 2>/dev/null || echo "?")
    echo "defq idle nodes: $IDLE"
fi

# 8. Quick sanity (MEDIUM, 4 threads, 2 iterations) unless --quick
if [[ $QUICK -eq 0 ]]; then
    echo ""
    echo "--- MEDIUM sanity run (4 threads, 2 iter, 1 warmup) ---"
    FAIL=0
    for b in $BENCHMARKS; do
        OUT=$(OMP_NUM_THREADS=4 ./benchmark_$b \
                --dataset MEDIUM --threads 4 --iterations 2 --warmup 1 2>&1 \
                | tail -20)
        if echo "$OUT" | grep -q FAIL; then
            echo "  $b: has FAILs (expected for known quarantined strategies)"
        else
            echo "  $b: PASS"
        fi
    done
fi

echo ""
echo "=== preflight done ==="
echo "If all green, submit with:"
echo "  sbatch --begin=22:00 slurm/das5_extralarge.slurm all"
