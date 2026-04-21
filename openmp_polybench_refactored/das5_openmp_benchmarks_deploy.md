# DAS-5 OpenMP PolyBench Deployment - Inline sbatch Dispatch

One-shot command-line submission. No `.slurm` or `.sh` wrapper scripts.
Each benchmark lands on its own exclusive cpunode (dual E5-2630-v3 Haswell,
16 cores, 64 GB), so the full suite runs in parallel across the cluster.

Target cluster: fs0.das5.cs.vu.nl (VU68).
Partition: `defq`. Constraint: `cpunode`. Threads cap: 16 physical cores.

---

## 0. Prerequisites

Build must be done on fs0 with the DAS-5-safe target (Haswell, no `-march=native`):

    cd "$PROJECT_DIR"
    make clean
    make das5
    ls -lh benchmark_{2mm,3mm,cholesky,correlation,nussinov,heat3d}

Verify cpunode availability:

    sinfo -p defq -o "%20N %6t %10f" | grep cpunode | grep idle

---

## 1. Shell environment (export once per session)

    export PROJECT_DIR="$HOME/ade910/latest/Julia_vs_OpenMP_Parallelism_Multithreading/opmp_std_3/openmp_polybench_refactored"
    export BENCHES="2mm 3mm cholesky correlation nussinov heat3d"
    mkdir -p "$PROJECT_DIR/results"

Override `BENCHES` to run a subset:

    export BENCHES="2mm cholesky heat3d"

---

## 2. XL parallel dispatch (one node per benchmark, full suite concurrent)

Direct analog to your Julia `scaleXL_*` submission. Each iteration submits an
independent job that claims one full cpunode via `--exclusive -C cpunode`.
All jobs enter the queue immediately and run concurrently as nodes free up.

    for bench in $BENCHES; do
        sbatch --job-name="scaleXL_omp_${bench}" \
               --output="${PROJECT_DIR}/results/scaleXL_omp_${bench}_%j.out" \
               --error="${PROJECT_DIR}/results/scaleXL_omp_${bench}_%j.err" \
               --time=03:00:00 \
               -N 1 \
               --ntasks=1 \
               --cpus-per-task=16 \
               --exclusive \
               --partition=defq \
               -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun; \
                       export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false OMP_WAIT_POLICY=active OMP_STACKSIZE=256M; \
                       cd ${PROJECT_DIR}; \
                       mkdir -p results; \
                       for t in 8 16; do \
                           echo \"=== Running ${bench} XL with \$t threads on \$(hostname) ===\"; \
                           export OMP_NUM_THREADS=\$t; \
                           ./benchmark_${bench} --dataset EXTRALARGE --threads \$t --iterations 5 --warmup 2 --output csv; \
                       done; \
                       echo \"=== ${bench} complete on \$(hostname) at \$(date -Iseconds) ===\""
    done

Behaviour:
- 6 jobs submitted, 6 cpunodes consumed concurrently (subject to availability).
- `${bench}` expands at submit-time (loop variable).
- `\$t`, `\$(hostname)`, `\$(date ...)` are escaped so they expand inside the job.
- Thread counts `8 16` mirror your Julia XL command.
- 3 h wall budget covers worst-case: nussinov-XL + cholesky-XL at 1 thread.
- Must run off-hours (see §7).

---

## 3. Thread scaling sweep (LARGE, 1-2-4-8-16)

Fits daytime 15-min cap per benchmark. Produces clean Amdahl-friendly curves.

    for bench in $BENCHES; do
        sbatch --job-name="scale_omp_${bench}" \
               --output="${PROJECT_DIR}/results/scale_omp_${bench}_%j.out" \
               --error="${PROJECT_DIR}/results/scale_omp_${bench}_%j.err" \
               --time=00:15:00 \
               -N 1 \
               --ntasks=1 \
               --cpus-per-task=16 \
               --exclusive \
               --partition=defq \
               -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun; \
                       export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false OMP_WAIT_POLICY=active OMP_STACKSIZE=256M; \
                       cd ${PROJECT_DIR}; \
                       mkdir -p results; \
                       for t in 1 2 4 8 16; do \
                           echo \"=== Running ${bench} LARGE with \$t threads on \$(hostname) ===\"; \
                           export OMP_NUM_THREADS=\$t; \
                           ./benchmark_${bench} --dataset LARGE --threads \$t --iterations 10 --warmup 3 --output csv; \
                       done; \
                       echo \"=== ${bench} LARGE scaling complete on \$(hostname) ===\""
    done

Expected runtime per job at LARGE: ~3-8 min (compute-bound benches),
up to ~12 min (nussinov). Stays under the 15 min daytime cap.

---

## 4. Quick single-benchmark sanity check (MEDIUM, 16 threads)

For smoke-testing a single kernel without blocking a queue slot for long.

    sbatch --job-name="quick_omp_2mm" \
           --output="${PROJECT_DIR}/results/quick_omp_2mm_%j.out" \
           --error="${PROJECT_DIR}/results/quick_omp_2mm_%j.err" \
           --time=00:05:00 \
           -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
           --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun; \
                   export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16; \
                   cd ${PROJECT_DIR}; \
                   ./benchmark_2mm --dataset MEDIUM --threads 16 --iterations 10 --warmup 3 --output csv"

---

## 5. Strategy-filtered run (skip known-broken strategies)

Your existing `das5_extralarge.slurm` quarantines:
- `cholesky`: `threads_static`, `tiled`, `tasks` (verification FAIL)
- `3mm`: `tasks` (FAIL at >= 4 threads)
- `nussinov`: `tiled` (off-by-one)

The binary's `--strategies` flag accepts a CSV of strategy names to run.
Example: run only verified strategies for cholesky XL.

    sbatch --job-name="scaleXL_omp_cholesky_clean" \
           --output="${PROJECT_DIR}/results/scaleXL_omp_cholesky_clean_%j.out" \
           --time=03:00:00 \
           -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
           --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun; \
                   export OMP_PROC_BIND=close OMP_PLACES=cores; \
                   cd ${PROJECT_DIR}; \
                   for t in 8 16; do \
                       export OMP_NUM_THREADS=\$t; \
                       ./benchmark_cholesky --dataset EXTRALARGE --threads \$t --iterations 5 --warmup 2 --output csv --strategies sequential,threads_dynamic,simd; \
                   done"

---

## 6. Monitoring

    squeue -u $USER                                                # all your jobs
    squeue -u $USER -o "%.10i %.24j %.8T %.10M %.6D %.20R"         # compact view
    scontrol show job <JOBID>                                      # full detail
    sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

Tail a running job:

    tail -f "${PROJECT_DIR}/results/scaleXL_omp_2mm_<JOBID>.out"

Cancel:

    scancel <JOBID>
    scancel -n scaleXL_omp_cholesky        # by name
    scancel -u $USER                       # all your jobs

---

## 7. Off-hours scheduling (mandatory for >15 min jobs on working days)

DAS-5 policy caps daytime jobs at 15 min. XL dispatches (`--time=03:00:00`)
must be deferred. Prefix each `sbatch` with one of:

    --begin=22:00           # tonight 22:00
    --begin=saturday        # next Saturday 00:00
    --begin=now+6hours      # 6 h from submit
    --begin=2026-04-22T18:00

Full XL launch deferred to 22:00:

    for bench in $BENCHES; do
        sbatch --begin=22:00 \
               --job-name="scaleXL_omp_${bench}" \
               --output="${PROJECT_DIR}/results/scaleXL_omp_${bench}_%j.out" \
               --error="${PROJECT_DIR}/results/scaleXL_omp_${bench}_%j.err" \
               --time=03:00:00 \
               -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
               --partition=defq -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun; \
                       export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false OMP_WAIT_POLICY=active OMP_STACKSIZE=256M; \
                       cd ${PROJECT_DIR}; \
                       for t in 8 16; do \
                           export OMP_NUM_THREADS=\$t; \
                           ./benchmark_${bench} --dataset EXTRALARGE --threads \$t --iterations 5 --warmup 2 --output csv; \
                       done"
    done

---

## 8. Post-run inspection and retrieval

On fs0, once jobs complete:

    cd "$PROJECT_DIR"
    ls -lh results/*EXTRALARGE*.csv
    ls -lh results/*LARGE*.csv
    wc -l results/*.csv
    awk -F, 'NR>1 && $13=="FAIL" {print FILENAME": "$3" "$4"T"}' results/*.csv

From your laptop:

    rsync -avz --include='*.csv' --include='*.out' --exclude='*' \
          das5:${PROJECT_DIR}/results/  ./das5_openmp_results/

---

## 9. Visualization pipeline

Expected CSV schema (emitted natively by every binary):

    benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,
    std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations

File naming: `results/<bench>_<DATASET>_<yyyymmdd_hhmmss>.csv`.
Per-second timestamps prevent collisions across concurrent jobs.

Dashboard + heatmap + scaling plots:

    cd path/to/openmp_polybench_refactored
    python3 scripts/visualize_benchmarks.py das5_openmp_results/*.csv \
            -o ./plots_das5 -t "DAS-5 VU68"

Scaling-only plots (for the sweep from §3):

    python3 scripts/visualize_benchmarks.py --scaling \
            das5_openmp_results/*LARGE*.csv \
            -o ./plots_das5_scaling

Julia vs OpenMP cross-language report:

    python3 scripts/compare_benchmarks.py \
            julia_results/*.csv \
            das5_openmp_results/*.csv \
            --report das5_julia_vs_openmp.md

---

## 10. Resource sizing reference

| Benchmark    | Dataset XL  | Seq time (1T) | 16T target | Memory peak |
|--------------|-------------|---------------|------------|-------------|
| 2mm          | NI/J/K/L=4000 | ~4 min      | ~15-30 s   | ~500 MB     |
| 3mm          | NI/J/K/L/M=4000 | ~8 min    | ~30-60 s   | ~900 MB     |
| cholesky     | n=4000      | ~7 min        | ~20-40 s   | ~130 MB     |
| correlation  | M/N=4000    | ~8 min        | ~25-45 s   | ~250 MB     |
| nussinov     | n=5500      | ~15 min       | dep-limited| ~250 MB     |
| heat3d       | n=120, tsteps=1000 | ~10 min | ~3-5 min   | ~14 MB      |

Per-job XL budget at 16T with 5 iterations and all strategies: 5-15 min.
3 h `--time` leaves substantial safety margin for thread=8 runs and slow strategies.

---

## 11. Security and operational notes

- No credentials in any command. `$USER`, `$HOME` only.
- `--exclusive` blocks co-tenancy on the node. Correct for timing; also
  minimises information leakage via `/proc`, `/sys`, and shared memory probes.
- `-C cpunode` pins to identical hardware across all runs. Mandatory for
  valid cross-benchmark and cross-run comparisons.
- Variable expansion discipline:
  - Loop variables (`${bench}`) expand at submit-time on fs0. Intended.
  - Runtime variables (`\$t`, `\$(hostname)`, `\$(date ...)`) are escaped,
    expanded on the compute node when the job runs. Intended.
- Consider restricting the results directory:
      chmod 700 "${PROJECT_DIR}/results"
  Output logs may contain hostnames and job IDs; exposure is low-risk but
  best practice is least-privilege.
- Never submit `sbatch --wrap=` strings assembled from untrusted input.
  The wrap body is executed verbatim by bash inside the job.
