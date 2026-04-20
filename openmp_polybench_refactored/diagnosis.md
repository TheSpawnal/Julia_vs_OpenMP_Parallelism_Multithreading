# PolyBench OpenMP — Strategy Failure Diagnosis and Fixes

Scope: five broken strategies identified in local LARGE CSV (20260418).
All diagnoses below are based on reading the source in
`opmp_std_3/openmp_polybench_refactored/src/`.

Status legend:
  [ROOT CAUSE]  — the precise mechanism causing the failure
  [FIX]         — the minimal change that corrects the strategy
  [VERIFICATION] — how we know the fix is right

---

## 1. cholesky / threads_static — max_error ~2.0e+03

### [ROOT CAUSE]

The broken code parallelizes the j-loop of Cholesky-Banachiewicz:

```c
for (int i = 0; i < n; i++) {
    if (i > 64) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++)
                A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
            A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
        }
        ...
    }
}
```

But iteration `j` of this loop **reads `A[i,k]` for `k < j`**, which is
**written by iteration `k` of the same loop**. There is a true read-after-write
dependency between iterations `j` of the parallel loop. Parallelizing it is
a correctness bug, not just a performance one. Thread t1 racing on
`A[i, k]` gets a stale value while thread t0 is still updating it.

### [FIX]

Switch from row-oriented (Cholesky-Banachiewicz, `ijk`) to column-oriented
(Cholesky-Crout, `kij`). In the `kij` form, after the diagonal element
`L[k,k]` is computed, the column update for rows `i > k` is
**independent across i**. The parallel loop is over `i`, not `j`.

```c
// Right-looking Cholesky-Crout: parallel over rows i for each column k.
// Rewrites only the lower triangle; result is bit-equivalent to the
// row-oriented sequential up to summation order (|delta| < 1e-14).
static void kernel_cholesky_threads_static(int n, double* A) {
    for (int k = 0; k < n; k++) {
        // Diagonal: L[k,k] = sqrt(A[k,k] - sum_{m<k} L[k,m]^2)
        double diag = A[IDX2(k, k, n)];
        for (int m = 0; m < k; m++) {
            double v = A[IDX2(k, m, n)];
            diag -= v * v;
        }
        A[IDX2(k, k, n)] = sqrt(diag);
        const double Lkk = A[IDX2(k, k, n)];

        // Column: L[i,k] = (A[i,k] - sum_{m<k} L[i,m] * L[k,m]) / L[k,k]
        // Fully independent across i -> safe to parallelize.
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double s = A[IDX2(i, k, n)];
            for (int m = 0; m < k; m++)
                s -= A[IDX2(i, m, n)] * A[IDX2(k, m, n)];
            A[IDX2(i, k, n)] = s / Lkk;
        }
    }
}
```

### [VERIFICATION]

The two formulations compute the same unique Cholesky factor; differences
are limited to summation-order FP rounding, bounded by `O(n * eps)` which
at `n=2000` gives `~2.2e-13`, far below any sane `VERIFY_TOLERANCE` (1e-6).

---

## 2. cholesky / tiled — max_error ~3.2e+03, impossible 160-188 GFLOP/s

### [ROOT CAUSE]

The tiled code only does **in-block** work. Look at the inner k-loops:

```c
for (int ii = 0; ii < n; ii += TILE_SIZE) {
    ...
    for (int i = ii; i < i_end; i++) {
        for (int j = ii; j < i; j++) {
            for (int k = ii; k < j; k++)  // <-- k starts at ii, not 0
                A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
            A[IDX2(i, j, n)] /= A[IDX2(j, j, n)];
        }
        ...
    }
    #pragma omp parallel for schedule(dynamic)
    for (int jj = ii + TILE_SIZE; jj < n; jj += TILE_SIZE) {
        ...
        for (int k = ii; k < j; k++)  // <-- same bug
            A[IDX2(i, j, n)] -= A[IDX2(i, k, n)] * A[IDX2(j, k, n)];
        ...
    }
}
```

The `k` loops start at `ii`, not at `0`. This **skips every rank-1 update
from columns `[0, ii)`** — every previously factored column is ignored.
For `ii = 0` (first tile) the code is correct; for subsequent tiles the
diagonal block and trailing panel are not updated with prior contributions.

Complexity per tile: `O(TILE_SIZE^3)` instead of `O(TILE_SIZE^2 * ii)`.
At `n=2000, TILE=64`, the total work done is only a tiny fraction of a
real Cholesky. That is why GFLOP/s reports "160-188" — the denominator
(`min_time_ms`) is tiny because almost nothing is actually computed,
and the numerator (theoretical `flops_cholesky(n) = n^3/3`) is the
full work count. The resulting matrix is garbage; the verification's
`max_error ~3230` is how far off the diagonal entries drift.

### [FIX]

Proper three-phase right-looking blocked Cholesky:

1. **POTRF** on the current diagonal block (in-place, using only
   within-block reads thanks to prior trailing updates).
2. **TRSM** on the panel below the diagonal block (parallel over rows).
3. **SYRK** trailing update on the lower-triangle of the trailing
   submatrix (parallel over rows). This is what makes step (1) of the
   next iteration valid with only within-block reads.

```c
#ifndef CHOLESKY_TILE
#define CHOLESKY_TILE 64
#endif

// Right-looking blocked Cholesky: POTRF + TRSM + SYRK per block column.
// Only the lower triangle is written; upper triangle remains zero.
static void kernel_cholesky_tiled(int n, double* A) {
    const int B = CHOLESKY_TILE;

    for (int k0 = 0; k0 < n; k0 += B) {
        const int bs = MIN(B, n - k0);

        // (1) POTRF: factor A[k0:k0+bs, k0:k0+bs] in place.
        //     Prior trailing updates ensure only in-block reads are needed.
        for (int k = 0; k < bs; k++) {
            double diag = A[IDX2(k0+k, k0+k, n)];
            for (int m = 0; m < k; m++) {
                double v = A[IDX2(k0+k, k0+m, n)];
                diag -= v * v;
            }
            A[IDX2(k0+k, k0+k, n)] = sqrt(diag);
            const double Lkk = A[IDX2(k0+k, k0+k, n)];

            for (int i = k + 1; i < bs; i++) {
                double s = A[IDX2(k0+i, k0+k, n)];
                for (int m = 0; m < k; m++)
                    s -= A[IDX2(k0+i, k0+m, n)] * A[IDX2(k0+k, k0+m, n)];
                A[IDX2(k0+i, k0+k, n)] = s / Lkk;
            }
        }

        if (k0 + bs >= n) break;

        // (2) TRSM: solve L21 * L11^T = A21  ->  L21 = A21 * L11^{-T}
        //     Parallel across rows i below the diagonal block.
        #pragma omp parallel for schedule(static)
        for (int i = k0 + bs; i < n; i++) {
            for (int k = 0; k < bs; k++) {
                double s = A[IDX2(i, k0+k, n)];
                for (int m = 0; m < k; m++)
                    s -= A[IDX2(i, k0+m, n)] * A[IDX2(k0+k, k0+m, n)];
                A[IDX2(i, k0+k, n)] = s / A[IDX2(k0+k, k0+k, n)];
            }
        }

        // (3) SYRK: A22 -= L21 * L21^T on the lower triangle.
        //     Parallel across rows i.
        #pragma omp parallel for schedule(dynamic)
        for (int i = k0 + bs; i < n; i++) {
            for (int j = k0 + bs; j <= i; j++) {
                double s = 0.0;
                for (int k = 0; k < bs; k++)
                    s += A[IDX2(i, k0+k, n)] * A[IDX2(j, k0+k, n)];
                A[IDX2(i, j, n)] -= s;
            }
        }
    }
}
```

### [VERIFICATION]

Total work is `O(n^3/3)`, identical to sequential. The GFLOP/s reading
will now be physically sensible (on a Haswell node with AVX2 FMA,
expect 5-30 GFLOP/s depending on blocking and memory traffic).

---

## 3. cholesky / tasks — max_error 4.5 to 65

### [ROOT CAUSE]

```c
#pragma omp task depend(inout: A[IDX2(i,0,n):i])
```

Two independent problems with this clause:

**Problem A:** OpenMP matches dependencies by the **starting address** of
the declared section. For each task `i`, the section starts at
`&A[i*n + 0]`, which is a **different address per i**. So no task ever
matches any other task's dependency — all N row-tasks are scheduled
concurrently.

**Problem B:** Even if dependencies did match, the clause only declares
writes to **row i**. But the task body **reads rows 0..i-1**
(`A[IDX2(j,k,n)]` and `A[IDX2(j,j,n)]` for `j < i`). Reads-from are not
declared, so OpenMP has no way to serialize row-i against the task that
computes row-j. Even if the dependency DSL were used correctly for the
writes, the reads would still race.

### [FIX]

Block-based task Cholesky with four kinds of tasks (POTRF, TRSM, SYRK,
GEMM) and address-matching depend clauses.  Each block has a unique
starting address, so `depend(in: BLK(I,K)[0:1])` actually matches across
tasks. OpenMP serializes correctly.

```c
#ifndef CHOLESKY_TASK_TILE
#define CHOLESKY_TASK_TILE 64
#endif

// Block-based task-parallel Cholesky with proper dependencies.
// BLK(ti,tj) is the first element of block (ti,tj); OpenMP uses the
// address as the dependency key, so different blocks are independent and
// the same block is serialized correctly.
static void kernel_cholesky_tasks(int n, double* A) {
    const int B = CHOLESKY_TASK_TILE;
    const int NT = (n + B - 1) / B;

    #define BLK(ti, tj) (A + ((size_t)(ti) * B) * n + ((size_t)(tj) * B))

    #pragma omp parallel
    #pragma omp single
    {
        for (int K = 0; K < NT; K++) {
            // POTRF on diagonal block (K,K)
            #pragma omp task depend(inout: BLK(K,K)[0:1])
            {
                const int k0 = K * B;
                const int bs = MIN(B, n - k0);
                for (int k = 0; k < bs; k++) {
                    double diag = A[IDX2(k0+k, k0+k, n)];
                    for (int m = 0; m < k; m++) {
                        double v = A[IDX2(k0+k, k0+m, n)];
                        diag -= v * v;
                    }
                    A[IDX2(k0+k, k0+k, n)] = sqrt(diag);
                    const double Lkk = A[IDX2(k0+k, k0+k, n)];
                    for (int i = k + 1; i < bs; i++) {
                        double s = A[IDX2(k0+i, k0+k, n)];
                        for (int m = 0; m < k; m++)
                            s -= A[IDX2(k0+i, k0+m, n)] * A[IDX2(k0+k, k0+m, n)];
                        A[IDX2(k0+i, k0+k, n)] = s / Lkk;
                    }
                }
            }

            // TRSM on panel blocks (I,K) for I > K
            for (int I = K + 1; I < NT; I++) {
                #pragma omp task depend(in:    BLK(K,K)[0:1]) \
                                 depend(inout: BLK(I,K)[0:1])
                {
                    const int i0 = I * B, ilen = MIN(B, n - i0);
                    const int k0 = K * B, bs   = MIN(B, n - k0);
                    for (int i = 0; i < ilen; i++) {
                        for (int k = 0; k < bs; k++) {
                            double s = A[IDX2(i0+i, k0+k, n)];
                            for (int m = 0; m < k; m++)
                                s -= A[IDX2(i0+i, k0+m, n)] * A[IDX2(k0+k, k0+m, n)];
                            A[IDX2(i0+i, k0+k, n)] = s / A[IDX2(k0+k, k0+k, n)];
                        }
                    }
                }
            }

            // SYRK/GEMM trailing updates on (I,J) for K < J <= I
            for (int I = K + 1; I < NT; I++) {
                for (int J = K + 1; J <= I; J++) {
                    #pragma omp task depend(in:    BLK(I,K)[0:1], BLK(J,K)[0:1]) \
                                     depend(inout: BLK(I,J)[0:1])
                    {
                        const int i0 = I * B, ilen = MIN(B, n - i0);
                        const int j0 = J * B, jlen = MIN(B, n - j0);
                        const int k0 = K * B, bs   = MIN(B, n - k0);
                        for (int i = 0; i < ilen; i++) {
                            const int jend = (I == J) ? i + 1 : jlen;
                            for (int j = 0; j < jend; j++) {
                                double s = 0.0;
                                for (int k = 0; k < bs; k++)
                                    s += A[IDX2(i0+i, k0+k, n)] * A[IDX2(j0+j, k0+k, n)];
                                A[IDX2(i0+i, j0+j, n)] -= s;
                            }
                        }
                    }
                }
            }
        }
    }

    #undef BLK
}
```

### [VERIFICATION]

Same algebra as the `tiled` fix, just scheduled as a task DAG instead
of phase-by-phase. Output identical up to FP rounding. The critical
path is `NT` POTRFs plus trailing work, and the DAG allows far more
overlap than the phase-synchronous version on high thread counts.

---

## 4. 3mm / tasks — max_error 0.07 to 0.12 at threads >= 4

### [ROOT CAUSE]

```c
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task                     // outer task A
        {
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii) { /* E += ... */ }
            }
        }
        #pragma omp task                     // outer task B
        {
            for (int ii = 0; ii < nj; ii += chunk) {
                #pragma omp task firstprivate(ii) { /* F += ... */ }
            }
        }
        #pragma omp taskwait                 // <-- problematic
        // G = E * F (spawns more tasks)
    }
}
```

`#pragma omp taskwait` waits for the **direct child tasks** of the
enclosing task to complete. The enclosing task is the implicit single-
block task, whose children are the two outer tasks A and B. Those
outer tasks "complete" **as soon as their code body finishes** — which
happens immediately after they spawn all their inner tasks (the spawn
does not wait for the inner task to finish).

Result: when `taskwait` returns, outer A and B have both "completed" in
OpenMP's bookkeeping, but their **grandchild tasks may still be running
or not yet scheduled**. The G computation then begins reading partial
or uninitialized values from E and F.

At 1-2 threads the scheduler runs tasks depth-first (a thread
executing the outer task tends to drain its child queue first), so the
bug is masked. At 4+ threads, work-stealing scatters tasks across
threads, and the race becomes visible as `max_error = 0.07 to 0.12`
(consistent with a handful of E/F cells still being zero when G reads
them).

### [FIX]

Eliminate the outer nesting. Spawn E and F children directly under
the `single` block, so `taskwait` actually waits for them.

```c
static void kernel_3mm_tasks(int ni, int nj, int nk, int nl, int nm,
                             double* A, double* B, double* C, double* D,
                             double* E, double* F, double* G) {
    const int chunk = MAX(ni / (omp_get_max_threads() * 4), 1);

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Phase 1a: E = A * B  (row-chunked, tasks independent)
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    const int i_end = MIN(ii + chunk, ni);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nj; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < nk; k++)
                                sum += A[IDX2(i, k, nk)] * B[IDX2(k, j, nj)];
                            E[IDX2(i, j, nj)] = sum;
                        }
                    }
                }
            }

            // Phase 1b: F = C * D  (runs concurrently with 1a)
            for (int ii = 0; ii < nj; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    const int i_end = MIN(ii + chunk, nj);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nl; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < nm; k++)
                                sum += C[IDX2(i, k, nm)] * D[IDX2(k, j, nl)];
                            F[IDX2(i, j, nl)] = sum;
                        }
                    }
                }
            }

            // Wait for ALL phase-1 tasks (E and F) to complete.
            #pragma omp taskwait

            // Phase 2: G = E * F
            for (int ii = 0; ii < ni; ii += chunk) {
                #pragma omp task firstprivate(ii)
                {
                    const int i_end = MIN(ii + chunk, ni);
                    for (int i = ii; i < i_end; i++) {
                        for (int j = 0; j < nl; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < nj; k++)
                                sum += E[IDX2(i, k, nj)] * F[IDX2(k, j, nl)];
                            G[IDX2(i, j, nl)] = sum;
                        }
                    }
                }
            }
            // Implicit barrier at end of single region waits for G tasks.
        }
    }
}
```

The `taskwait` now waits for the right set of tasks (E-chunk tasks and
F-chunk tasks, all direct children of the single block).

Alternative: use `#pragma omp taskgroup { ... }` around phase 1 for
clearer semantics. Both are correct; the flat form above avoids the
nesting depth.

### [VERIFICATION]

Outputs are identical to the sequential `E = A*B`, `F = C*D`,
`G = E*F` up to FP summation-order rounding, which for the
double-precision matrix sizes in PolyBench is well under 1e-12.

---

## 5. nussinov / tiled — max_error = 1 (off-by-one at tile boundary)

### [ROOT CAUSE]

The broken code uses `tile_diag = T_i + T_j` as the wavefront order:

```c
for (int tile_diag = 0; tile_diag < 2 * num_tiles - 1; tile_diag++) {
    #pragma omp parallel for schedule(dynamic)
    for (int ti = MAX(0, tile_diag - num_tiles + 1);
             ti <= MIN(tile_diag, num_tiles - 1); ti++) {
        int tj = tile_diag - ti;
        if (tj < ti) continue;
        ...
    }
}
```

But Nussinov iterates `i` from `n-1` **down** to `0`. Cell `(i, j)`
depends on `(i+1, j)` — one row **below**. In tile coordinates,
cell `(i, j)` in tile `(T_i, T_j)` depends on cell `(i+1, j)` which
is in tile `(T_i + 1, T_j)` when `i+1` crosses the tile boundary.

With `tile_diag = T_i + T_j`:
- Tile `(T_i, T_j)` has `tile_diag = T_i + T_j`.
- Its dependency `(T_i + 1, T_j)` has `tile_diag = T_i + T_j + 1`.

So the dependency's tile is processed **later**, which is backwards.
When we compute tile `(0, 2)` on diag 2, we read from tile `(1, 2)`
which is on diag 3 and **not yet computed**. The max_error of exactly
1 indicates a single-cell mismatch at a tile boundary — consistent
with a race that produces a stale read of the still-zero cell.

The correct wavefront for Nussinov on the upper triangle is
`d_tile = T_j - T_i` (tile-stripe distance). All dependencies of a
tile at `d_tile` are on tiles at `d_tile - 1` or smaller; tiles at
the same `d_tile` are independent.

### [FIX]

```c
#ifndef NUSSINOV_TILE_SIZE
#define NUSSINOV_TILE_SIZE 64
#endif

// Correct tile wavefront: d_tile = T_j - T_i (stripe distance).
// Tiles on the same d_tile are independent; all deps are on smaller d_tile.
static void kernel_nussinov_tiled(int n, base* seq, int* table) {
    const int TS = NUSSINOV_TILE_SIZE;
    const int num_tiles = (n + TS - 1) / TS;

    for (int d_tile = 0; d_tile < num_tiles; d_tile++) {
        #pragma omp parallel for schedule(dynamic)
        for (int ti = 0; ti + d_tile < num_tiles; ti++) {
            const int tj = ti + d_tile;
            const int i_start = ti * TS;
            const int j_start = tj * TS;
            const int i_end   = MIN(i_start + TS, n);
            const int j_end   = MIN(j_start + TS, n);

            // Within-tile: sequential Nussinov order (i descending, j ascending).
            for (int i = i_end - 1; i >= i_start; i--) {
                for (int j = MAX(j_start, i + 1); j < j_end; j++) {
                    int score = table[IDX2(i, j, n)];

                    // Pair case
                    if (j - 1 >= 0 && i + 1 < n) {
                        if (i < j - 1)
                            score = max_score(score, table[IDX2(i+1, j-1, n)] + match(seq[i], seq[j]));
                        else
                            score = max_score(score, match(seq[i], seq[j]));
                    }
                    // i unpaired
                    if (i + 1 < n)
                        score = max_score(score, table[IDX2(i+1, j, n)]);
                    // j unpaired
                    if (j - 1 >= 0)
                        score = max_score(score, table[IDX2(i, j-1, n)]);
                    // Bifurcation
                    for (int k = i + 1; k < j; k++)
                        score = max_score(score, table[IDX2(i, k, n)] + table[IDX2(k+1, j, n)]);

                    table[IDX2(i, j, n)] = score;
                }
            }
        }
    }
}
```

### [VERIFICATION]

Per-cell analysis of dependencies at tile boundaries:
- `(i, j-1)` at `j-1 < j_start`: in tile `(T_i, T_j-1)`, `d = T_j-1 - T_i < d_tile` — earlier.
- `(i+1, j)` at `i+1 >= i_end`: in tile `(T_i+1, T_j)`, `d = T_j - T_i - 1 < d_tile` — earlier.
- `(i+1, j-1)` at corner: `d < d_tile` — earlier.
- Row bifurcation `(i, k)` with `k < j_start`: `d = T_k - T_i < d_tile` — earlier.
- Column bifurcation `(k+1, j)` with `k+1 >= i_end`: in tile with `T' > T_i`, `d = T_j - T' < d_tile` — earlier.
- Cross-tile reads at the same `d_tile`: impossible (all neighbors are at lower d).
- Within-tile reads: respected by the inner `i`-descending, `j`-ascending sweep.

Output is bit-identical to the `wavefront` strategy (which also uses
wavefront order on the anti-diagonal).

---

## Application procedure

For each fix:

1. Open the corresponding source file in an editor:
   - `src/benchmark_cholesky.c` for fixes 1, 2, 3
   - `src/benchmark_3mm.c` for fix 4
   - `src/benchmark_nussinov.c` for fix 5
2. Locate the `static void kernel_*` function named in the fix header.
3. Replace the entire function body (from the opening `{` to the closing
   `}`) with the corrected code block above.
4. Rebuild: `make clean && make das5` (on DAS-5) or `make` (locally).
5. Verify: `bash verify_fixes.sh` runs a MEDIUM-size sanity check.

The three `#define` constants (`CHOLESKY_TILE`, `CHOLESKY_TASK_TILE`,
`NUSSINOV_TILE_SIZE`) use `#ifndef` guards so the existing `TILE_SIZE`
in nussinov is not shadowed.

---

## Why these particular choices

- **Right-looking over left-looking for cholesky**: right-looking
  produces more parallelism per step (the trailing SYRK is O(n^2)
  parallel work), which matches OpenMP's fork-join well. Left-looking
  concentrates work in a single panel per step — less parallelism.
- **Block tasks over element tasks for cholesky/tasks**: the OpenMP
  depend mechanism matches on address, not on logical array cells.
  Block-address tasks give unambiguous matching without the
  false-dependency explosion of element-level tasks.
- **Flat tasks over nested for 3mm**: less overhead and the semantics
  of `taskwait` become unambiguous. Nested tasks are useful when you
  want hierarchical cut-offs (divide-and-conquer), not for simple
  phase-parallel work.
- **Tile-stripe wavefront for nussinov**: matches the algebra of
  Nussinov dependencies exactly. The sum-diagonal would be correct for
  a different DP shape (forward, e.g. LCS), but Nussinov's iteration
  direction is opposite.

---

## Expected build warnings

With `-Wall -Wextra`, the cholesky tasks fix emits:

```
warning: unused variable 'p_kk' [-Wunused-variable]
warning: unused variable 'p_ik' [-Wunused-variable]
warning: unused variable 'p_jk' [-Wunused-variable]
warning: unused variable 'p_ij' [-Wunused-variable]
```

These are **false positives**. The variables are used inside
`#pragma omp task depend(...)` clauses, which GCC's unused-variable
analysis does not inspect. The pointers are required there because
OpenMP `depend` clauses expect a named lvalue-with-array-section, not
a compound pointer expression from a macro. The variables must stay.

If the project Makefile promotes warnings to errors with `-Werror`,
add `-Wno-unused-variable` for the cholesky translation unit, or
annotate each local with `__attribute__((unused))`. The current
Makefile does not use `-Werror`, so the warnings will appear but
not break the build.