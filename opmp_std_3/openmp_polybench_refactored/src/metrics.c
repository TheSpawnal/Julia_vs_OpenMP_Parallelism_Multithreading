/*
 * metrics.c -- benchmark result collection, CSV/JSON export.
 *
 * What changed relative to the previous version:
 *   - metrics_export_csv() now writes a structured, sanitized metadata block:
 *
 *       # schema=1; language=openmp; host=<h>; os=<os>; cpu=<cpu>; ...
 *       # benchmark=<name>; dataset=<size>; threads=<n>; omp_proc_bind=<>; ...
 *       benchmark,dataset,strategy,threads,is_parallel,min_ms,...
 *
 *     Every value is pre-sanitized to contain no ',', ';', '=', '#', quote,
 *     or whitespace. This survives round-trip through pandas(comment='#')
 *     and through scripts/bench_io.py.
 *   - Extra metadata fields: os, cpu model, logical cores, compiler name,
 *     compiler version, compiler flags (optional -DBENCH_CFLAGS=...),
 *     OpenMP version, ISO-8601 run_date, project name + version,
 *     optional git commit (-DBENCH_GIT_COMMIT=...), OMP_* env snapshot.
 *
 * CSV header unchanged: 15 Julia-compatible fields.
 * All other public symbols (metrics_init, timing_*, compute_*, etc.) keep
 * identical signatures and behavior.
 */

/* POSIX feature test macros: must come before any header include.
 * They enable gethostname, localtime_r, sysconf, etc. under -std=c11. */
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#  define _DEFAULT_SOURCE
#endif

#include "metrics.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* -------------------------------------------------------------------------- *
 * Build-time identification (all optional, overridable via -D...)            *
 * -------------------------------------------------------------------------- */
#if defined(__clang__)
#  define BENCH_CC_NAME    "clang"
#  define BENCH_CC_VERSION __clang_version__
#elif defined(__INTEL_LLVM_COMPILER)
#  define BENCH_CC_NAME    "icx"
#  define BENCH_CC_VERSION __VERSION__
#elif defined(__INTEL_COMPILER)
#  define BENCH_CC_NAME    "icc"
#  define BENCH_CC_VERSION __VERSION__
#elif defined(__GNUC__)
#  define BENCH_CC_NAME    "gcc"
#  define BENCH_CC_VERSION __VERSION__
#else
#  define BENCH_CC_NAME    "unknown"
#  define BENCH_CC_VERSION "unknown"
#endif

#ifndef BENCH_CFLAGS
#  define BENCH_CFLAGS "unknown"
#endif
#ifndef BENCH_PROJECT_NAME
#  define BENCH_PROJECT_NAME "openmp_polybench_refactored"
#endif
#ifndef BENCH_PROJECT_VERSION
#  define BENCH_PROJECT_VERSION "0.3"
#endif
#ifndef BENCH_GIT_COMMIT
#  define BENCH_GIT_COMMIT "unknown"
#endif

/* -------------------------------------------------------------------------- *
 * Internal helpers                                                           *
 * -------------------------------------------------------------------------- */

static int compare_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

/* Replace every forbidden char (CSV / metadata grammar) with '_'.
 * In-place; caller owns the buffer. */
static void sanitize_meta(char *s) {
    if (!s) return;
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == ';' || c == ',' || c == '=' || c == '#' ||
            c == '"' || c == '\'' ||
            c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            *s = '_';
        } else if (c < 0x20 || c == 0x7f) {
            *s = '_';
        }
    }
}

/* Copy src into dst (cap bytes), NUL-terminate, sanitize, fall back to
 * "unknown" if src is NULL or empty after sanitization. */
static void set_meta(char *dst, size_t cap, const char *src) {
    if (!dst || cap == 0) return;
    if (!src) { dst[0] = '\0'; }
    else {
        size_t n = strlen(src);
        if (n >= cap) n = cap - 1;
        memcpy(dst, src, n);
        dst[n] = '\0';
        sanitize_meta(dst);
    }
    if (dst[0] == '\0') {
        const char *u = "unknown";
        size_t ul = strlen(u);
        if (ul >= cap) ul = cap - 1;
        memcpy(dst, u, ul);
        dst[ul] = '\0';
    }
}

/* Best-effort: read "model name" from /proc/cpuinfo (Linux). */
static void read_cpu_model(char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    buf[0] = '\0';
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                size_t L = strlen(colon);
                while (L > 0 && (colon[L-1] == '\n' || colon[L-1] == '\r'))
                    colon[--L] = '\0';
                strncpy(buf, colon, cap - 1);
                buf[cap - 1] = '\0';
                break;
            }
        }
    }
    fclose(fp);
}

static int read_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0 && n <= 4096) ? (int)n : 0;
}

/* ISO-8601 local time: "2026-04-17T12:14:41+0200" */
static void get_iso_timestamp(char *buf, size_t cap) {
    if (!buf || cap < 26) { if (buf && cap) buf[0] = '\0'; return; }
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(buf, cap, "%Y-%m-%dT%H:%M:%S%z", &tm_info);
}

static void read_env_or(const char *name, char *buf, size_t cap, const char *fallback) {
    const char *v = getenv(name);
    if (v && *v) set_meta(buf, cap, v);
    else         set_meta(buf, cap, fallback);
}

/* -------------------------------------------------------------------------- *
 * metrics_init + timing_*                                                    *
 * -------------------------------------------------------------------------- */

void metrics_init(MetricsCollector* mc, const char* benchmark,
                  const char* dataset, int threads) {
    memset(mc->benchmark_name, 0, sizeof(mc->benchmark_name));
    memset(mc->dataset,        0, sizeof(mc->dataset));
    strncpy(mc->benchmark_name, benchmark, sizeof(mc->benchmark_name) - 1);
    strncpy(mc->dataset,        dataset,   sizeof(mc->dataset) - 1);
    mc->threads = threads;
    mc->num_results = 0;
    mc->sequential_time_ms = -1.0;  /* not set yet */
}

void timing_init(TimingData* td) {
    td->count = 0;
}

void timing_record(TimingData* td, double time_ms) {
    if (td->count < MAX_ITERATIONS) {
        td->times_ms[td->count++] = time_ms;
    }
}

double timing_min(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double m = td->times_ms[0];
    for (int i = 1; i < td->count; i++)
        if (td->times_ms[i] < m) m = td->times_ms[i];
    return m;
}

double timing_max(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double m = td->times_ms[0];
    for (int i = 1; i < td->count; i++)
        if (td->times_ms[i] > m) m = td->times_ms[i];
    return m;
}

double timing_median(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double sorted[MAX_ITERATIONS];
    memcpy(sorted, td->times_ms, (size_t)td->count * sizeof(double));
    qsort(sorted, (size_t)td->count, sizeof(double), compare_double);
    if (td->count % 2 == 0)
        return (sorted[td->count/2 - 1] + sorted[td->count/2]) / 2.0;
    return sorted[td->count/2];
}

double timing_mean(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < td->count; i++) sum += td->times_ms[i];
    return sum / td->count;
}

double timing_std(const TimingData* td) {
    if (td->count < 2) return 0.0;
    double mean = timing_mean(td);
    double sum_sq = 0.0;
    for (int i = 0; i < td->count; i++) {
        double d = td->times_ms[i] - mean;
        sum_sq += d * d;
    }
    return sqrt(sum_sq / (td->count - 1));  /* sample stddev */
}

/* -------------------------------------------------------------------------- *
 * compute_*                                                                  *
 * -------------------------------------------------------------------------- */

double compute_speedup(double baseline_ms, double current_ms) {
    if (current_ms <= 0.0) return 1.0;
    return baseline_ms / current_ms;
}

double compute_efficiency(const char* strategy, double speedup, int threads) {
    if (!strategy_is_parallel(strategy)) return NAN;
    if (threads <= 0) return NAN;
    return (speedup / threads) * 100.0;
}

double compute_gflops(double flops, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    return flops / (time_ms / 1000.0) / 1e9;
}

/* -------------------------------------------------------------------------- *
 * metrics_record                                                             *
 * -------------------------------------------------------------------------- */

void metrics_record(MetricsCollector* mc,
                    const char* strategy,
                    const TimingData* timing,
                    double flops,
                    int verified,
                    double max_error) {
    if (mc->num_results >= 32) return;

    BenchmarkResult* r = &mc->results[mc->num_results];

    memset(r->benchmark, 0, sizeof(r->benchmark));
    memset(r->dataset,   0, sizeof(r->dataset));
    memset(r->strategy,  0, sizeof(r->strategy));
    strncpy(r->benchmark, mc->benchmark_name, sizeof(r->benchmark) - 1);
    strncpy(r->dataset,   mc->dataset,        sizeof(r->dataset)   - 1);
    strncpy(r->strategy,  strategy,           sizeof(r->strategy)  - 1);

    r->threads     = mc->threads;
    r->is_parallel = strategy_is_parallel(strategy);

    r->min_ms    = timing_min(timing);
    r->median_ms = timing_median(timing);
    r->mean_ms   = timing_mean(timing);
    r->std_ms    = timing_std(timing);

    r->gflops = compute_gflops(flops, r->min_ms);

    if (strcmp(strategy, "sequential") == 0 || strcmp(strategy, "seq") == 0) {
        mc->sequential_time_ms = r->min_ms;
        r->speedup = 1.0;
    } else if (mc->sequential_time_ms > 0.0) {
        r->speedup = compute_speedup(mc->sequential_time_ms, r->min_ms);
    } else {
        r->speedup = 1.0;  /* no baseline yet */
    }

    r->efficiency_pct = compute_efficiency(strategy, r->speedup, r->threads);
    r->verified       = verified;
    r->max_error      = max_error;
    r->allocations    = 0;  /* not tracked on the C side */

    mc->num_results++;
}

/* -------------------------------------------------------------------------- *
 * Console printing                                                           *
 * -------------------------------------------------------------------------- */

void metrics_print_header(void) {
    printf("%-20s %-10s %-8s %-12s %-12s %-10s %-10s %-10s %s\n",
           "Strategy", "Threads", "Parallel", "Min(ms)", "Median(ms)",
           "GFLOP/s", "Speedup", "Eff(%)", "Verified");
    printf("--------------------------------------------------------------------------------\n");
}

void metrics_print_result(const BenchmarkResult* r) {
    char eff_str[16];
    if (isnan(r->efficiency_pct))
        snprintf(eff_str, sizeof(eff_str), "-");
    else
        snprintf(eff_str, sizeof(eff_str), "%.1f", r->efficiency_pct);

    printf("%-20s %-10d %-8s %-12.3f %-12.3f %-10.2f %-10.2f %-10s %s\n",
           r->strategy,
           r->threads,
           r->is_parallel ? "true" : "false",
           r->min_ms,
           r->median_ms,
           r->gflops,
           r->speedup,
           eff_str,
           r->verified ? "PASS" : "FAIL");
}

void metrics_print_summary(const MetricsCollector* mc) {
    printf("\n");
    printf("Benchmark: %s | Dataset: %s | Threads: %d\n",
           mc->benchmark_name, mc->dataset, mc->threads);
    metrics_print_header();
    for (int i = 0; i < mc->num_results; i++)
        metrics_print_result(&mc->results[i]);
    printf("\n");
}

/* -------------------------------------------------------------------------- *
 * CSV export (the only function whose format changed)                        *
 * -------------------------------------------------------------------------- */

void metrics_export_csv(const MetricsCollector* mc, const char* filepath) {
    if (!mc || !filepath) return;

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s for writing: %s\n",
                filepath, strerror(errno));
        return;
    }

    /* ---- Gather metadata into sanitized, length-bounded buffers ---- */
    char host[128];
    {
        char raw[128] = {0};
        if (gethostname(raw, sizeof(raw) - 1) != 0)
            strncpy(raw, "unknown", sizeof(raw) - 1);
        set_meta(host, sizeof(host), raw);
    }

    char os_name[128];
    {
        struct utsname un;
        char raw[384];  /* generous; set_meta truncates safely */
        if (uname(&un) == 0)
            snprintf(raw, sizeof(raw), "%s-%s", un.sysname, un.release);
        else
            snprintf(raw, sizeof(raw), "unknown");
        set_meta(os_name, sizeof(os_name), raw);
    }

    char cpu[192];
    {
        char raw[192] = {0};
        read_cpu_model(raw, sizeof(raw));
        if (raw[0] == '\0') strncpy(raw, "unknown", sizeof(raw) - 1);
        set_meta(cpu, sizeof(cpu), raw);
    }

    int cores_logical = read_logical_cores();

    char compiler[32];    set_meta(compiler,    sizeof(compiler),    BENCH_CC_NAME);
    char cc_version[96];  set_meta(cc_version,  sizeof(cc_version),  BENCH_CC_VERSION);
    char cflags[256];     set_meta(cflags,      sizeof(cflags),      BENCH_CFLAGS);

    int omp_ver = 0;
#ifdef _OPENMP
    omp_ver = _OPENMP;
#endif

    char run_date[40];
    {
        char raw[40];
        get_iso_timestamp(raw, sizeof(raw));
        set_meta(run_date, sizeof(run_date), raw);
    }

    char project[64]; set_meta(project, sizeof(project), BENCH_PROJECT_NAME);
    char pver[32];    set_meta(pver,    sizeof(pver),    BENCH_PROJECT_VERSION);
    char git[64];     set_meta(git,     sizeof(git),     BENCH_GIT_COMMIT);

    char omp_bind[32];    read_env_or("OMP_PROC_BIND", omp_bind,    sizeof(omp_bind),    "unset");
    char omp_places[64];  read_env_or("OMP_PLACES",    omp_places,  sizeof(omp_places),  "unset");
    char omp_sched[64];   read_env_or("OMP_SCHEDULE",  omp_sched,   sizeof(omp_sched),   "unset");

    char bench_name[64];  set_meta(bench_name, sizeof(bench_name), mc->benchmark_name);
    char dataset[32];     set_meta(dataset,    sizeof(dataset),    mc->dataset);

    /* ---- Line 1: build + environment identity ---- */
    fprintf(fp,
        "# schema=1; language=openmp; host=%s; os=%s; cpu=%s; cores_logical=%d; "
        "compiler=%s; compiler_version=%s; compiler_flags=%s; openmp=%d; "
        "run_date=%s; project=%s; project_version=%s; git_commit=%s\n",
        host, os_name, cpu, cores_logical,
        compiler, cc_version, cflags, omp_ver,
        run_date, project, pver, git);

    /* ---- Line 2: run-specific parameters ---- */
    fprintf(fp,
        "# benchmark=%s; dataset=%s; threads=%d; "
        "omp_proc_bind=%s; omp_places=%s; omp_schedule=%s\n",
        bench_name, dataset, mc->threads,
        omp_bind, omp_places, omp_sched);

    /* ---- CSV header: 15 Julia-compatible fields ---- */
    fprintf(fp, "benchmark,dataset,strategy,threads,is_parallel,"
                "min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,"
                "verified,max_error,allocations\n");

    for (int i = 0; i < mc->num_results; i++) {
        const BenchmarkResult* r = &mc->results[i];

        char eff_str[32];
        if (isnan(r->efficiency_pct)) eff_str[0] = '\0';
        else snprintf(eff_str, sizeof(eff_str), "%.2f", r->efficiency_pct);

        fprintf(fp,
                "%s,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%s,%s,%.2e,%ld\n",
                r->benchmark,
                r->dataset,
                r->strategy,
                r->threads,
                r->is_parallel ? "true" : "false",
                r->min_ms,
                r->median_ms,
                r->mean_ms,
                r->std_ms,
                r->gflops,
                r->speedup,
                eff_str,
                r->verified ? "PASS" : "FAIL",
                r->max_error,
                (long)r->allocations);
    }

    fclose(fp);
    printf("CSV exported: %s\n", filepath);
}

/* -------------------------------------------------------------------------- *
 * JSON export (unchanged from original, minus the empty file-header block)   *
 * -------------------------------------------------------------------------- */

void metrics_export_json(const MetricsCollector* mc, const char* filepath) {
    if (!mc || !filepath) return;

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s for writing: %s\n",
                filepath, strerror(errno));
        return;
    }

    char ts[64];
    get_timestamp(ts, sizeof(ts));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"metadata\": {\n");
    fprintf(fp, "    \"timestamp\": \"%s\",\n", ts);
    fprintf(fp, "    \"benchmark\": \"%s\",\n", mc->benchmark_name);
    fprintf(fp, "    \"dataset\": \"%s\",\n",   mc->dataset);
    fprintf(fp, "    \"threads\": %d,\n",       mc->threads);
    fprintf(fp, "    \"language\": \"openmp\"\n");
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"results\": [\n");

    for (int i = 0; i < mc->num_results; i++) {
        const BenchmarkResult* r = &mc->results[i];

        fprintf(fp, "    {\n");
        fprintf(fp, "      \"strategy\": \"%s\",\n",    r->strategy);
        fprintf(fp, "      \"threads\": %d,\n",         r->threads);
        fprintf(fp, "      \"is_parallel\": %s,\n",     r->is_parallel ? "true" : "false");
        fprintf(fp, "      \"min_ms\": %.4f,\n",        r->min_ms);
        fprintf(fp, "      \"median_ms\": %.4f,\n",     r->median_ms);
        fprintf(fp, "      \"mean_ms\": %.4f,\n",       r->mean_ms);
        fprintf(fp, "      \"std_ms\": %.4f,\n",        r->std_ms);
        fprintf(fp, "      \"gflops\": %.2f,\n",        r->gflops);
        fprintf(fp, "      \"speedup\": %.2f,\n",       r->speedup);
        if (isnan(r->efficiency_pct))
            fprintf(fp, "      \"efficiency_pct\": null,\n");
        else
            fprintf(fp, "      \"efficiency_pct\": %.2f,\n", r->efficiency_pct);
        fprintf(fp, "      \"verified\": %s,\n",        r->verified ? "true" : "false");
        fprintf(fp, "      \"max_error\": %.2e\n",      r->max_error);
        fprintf(fp, "    }%s\n",                        (i < mc->num_results - 1) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    printf("JSON exported: %s\n", filepath);
}

/* -------------------------------------------------------------------------- *
 * get_timestamp -- kept for backward compatibility with benchmark_*.c which  *
 * uses it to build CSV filenames (compact form: YYYYMMDD_HHMMSS).            *
 * -------------------------------------------------------------------------- */

void get_timestamp(char* buf, size_t len) {
    if (!buf || len == 0) return;
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    strftime(buf, len, "%Y%m%d_%H%M%S", &tm_info);
}