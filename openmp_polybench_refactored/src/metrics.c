#include "metrics.h"
#include <float.h>

// Comparison function for qsort
static int compare_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

void metrics_init(MetricsCollector* mc, const char* benchmark, const char* dataset, int threads) {
    strncpy(mc->benchmark_name, benchmark, sizeof(mc->benchmark_name) - 1);
    strncpy(mc->dataset, dataset, sizeof(mc->dataset) - 1);
    mc->threads = threads;
    mc->num_results = 0;
    mc->sequential_time_ms = -1.0;  // Not yet set
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
    double min_val = td->times_ms[0];
    for (int i = 1; i < td->count; i++) {
        if (td->times_ms[i] < min_val) min_val = td->times_ms[i];
    }
    return min_val;
}

double timing_max(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double max_val = td->times_ms[0];
    for (int i = 1; i < td->count; i++) {
        if (td->times_ms[i] > max_val) max_val = td->times_ms[i];
    }
    return max_val;
}

double timing_median(const TimingData* td) {
    if (td->count == 0) return 0.0;
    
    double sorted[MAX_ITERATIONS];
    memcpy(sorted, td->times_ms, td->count * sizeof(double));
    qsort(sorted, td->count, sizeof(double), compare_double);
    
    if (td->count % 2 == 0) {
        return (sorted[td->count/2 - 1] + sorted[td->count/2]) / 2.0;
    } else {
        return sorted[td->count/2];
    }
}

double timing_mean(const TimingData* td) {
    if (td->count == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < td->count; i++) {
        sum += td->times_ms[i];
    }
    return sum / td->count;
}

double timing_std(const TimingData* td) {
    if (td->count < 2) return 0.0;
    double mean = timing_mean(td);
    double sum_sq = 0.0;
    for (int i = 0; i < td->count; i++) {
        double diff = td->times_ms[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (td->count - 1));  // Sample standard deviation
}

double compute_speedup(double baseline_ms, double current_ms) {
    if (current_ms <= 0.0) return 1.0;
    return baseline_ms / current_ms;
}

double compute_efficiency(const char* strategy, double speedup, int threads) {
    // Efficiency only meaningful for parallel strategies
    if (!strategy_is_parallel(strategy)) {
        return NAN;  // Not applicable
    }
    if (threads <= 0) return NAN;
    return (speedup / threads) * 100.0;
}

double compute_gflops(double flops, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    return flops / (time_ms / 1000.0) / 1e9;
}

void metrics_record(MetricsCollector* mc,
                   const char* strategy,
                   const TimingData* timing,
                   double flops,
                   int verified,
                   double max_error) {
    if (mc->num_results >= 32) return;
    
    BenchmarkResult* r = &mc->results[mc->num_results];
    
    // Safe string copy with explicit null termination
    memset(r->benchmark, 0, sizeof(r->benchmark));
    memset(r->dataset, 0, sizeof(r->dataset));
    memset(r->strategy, 0, sizeof(r->strategy));
    strncpy(r->benchmark, mc->benchmark_name, sizeof(r->benchmark) - 1);
    strncpy(r->dataset, mc->dataset, sizeof(r->dataset) - 1);
    strncpy(r->strategy, strategy, sizeof(r->strategy) - 1);
    r->threads = mc->threads;
    r->is_parallel = strategy_is_parallel(strategy);
    
    r->min_ms = timing_min(timing);
    r->median_ms = timing_median(timing);
    r->mean_ms = timing_mean(timing);
    r->std_ms = timing_std(timing);
    
    r->gflops = compute_gflops(flops, r->min_ms);
    
    // Set sequential baseline if this is sequential strategy
    if (strcmp(strategy, "sequential") == 0 || strcmp(strategy, "seq") == 0) {
        mc->sequential_time_ms = r->min_ms;
        r->speedup = 1.0;
    } else if (mc->sequential_time_ms > 0.0) {
        r->speedup = compute_speedup(mc->sequential_time_ms, r->min_ms);
    } else {
        r->speedup = 1.0;  // No baseline yet
    }
    
    r->efficiency_pct = compute_efficiency(strategy, r->speedup, r->threads);
    r->verified = verified;
    r->max_error = max_error;
    r->allocations = 0;  // Not tracked in C
    
    mc->num_results++;
}

void metrics_print_header(void) {
    printf("%-20s %-10s %-8s %-12s %-12s %-10s %-10s %-10s %s\n",
           "Strategy", "Threads", "Parallel", "Min(ms)", "Median(ms)", 
           "GFLOP/s", "Speedup", "Eff(%)", "Verified");
    printf("--------------------------------------------------------------------------------\n");
}

void metrics_print_result(const BenchmarkResult* r) {
    char eff_str[16];
    if (isnan(r->efficiency_pct)) {
        snprintf(eff_str, sizeof(eff_str), "-");
    } else {
        snprintf(eff_str, sizeof(eff_str), "%.1f", r->efficiency_pct);
    }
    
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
    printf("=============================================================================\n");
    printf("Benchmark: %s | Dataset: %s | Threads: %d\n", 
           mc->benchmark_name, mc->dataset, mc->threads);
    printf("=============================================================================\n");
    metrics_print_header();
    for (int i = 0; i < mc->num_results; i++) {
        metrics_print_result(&mc->results[i]);
    }
    printf("\n");
}

void metrics_export_csv(const MetricsCollector* mc, const char* filepath) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filepath);
        return;
    }
    
    // CSV header - aligned with Julia format
    fprintf(fp, "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations\n");
    
    for (int i = 0; i < mc->num_results; i++) {
        const BenchmarkResult* r = &mc->results[i];
        
        // Efficiency: empty string for NaN (Julia convention)
        char eff_str[32];
        if (isnan(r->efficiency_pct)) {
            eff_str[0] = '\0';
        } else {
            snprintf(eff_str, sizeof(eff_str), "%.2f", r->efficiency_pct);
        }
        
        fprintf(fp, "%s,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%s,%s,%.2e,%ld\n",
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

void metrics_export_json(const MetricsCollector* mc, const char* filepath) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filepath);
        return;
    }
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    fprintf(fp, "{\n");
    fprintf(fp, "  \"metadata\": {\n");
    fprintf(fp, "    \"timestamp\": \"%s\",\n", timestamp);
    fprintf(fp, "    \"benchmark\": \"%s\",\n", mc->benchmark_name);
    fprintf(fp, "    \"dataset\": \"%s\",\n", mc->dataset);
    fprintf(fp, "    \"threads\": %d,\n", mc->threads);
    fprintf(fp, "    \"language\": \"openmp\"\n");
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"results\": [\n");
    
    for (int i = 0; i < mc->num_results; i++) {
        const BenchmarkResult* r = &mc->results[i];
        
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"strategy\": \"%s\",\n", r->strategy);
        fprintf(fp, "      \"threads\": %d,\n", r->threads);
        fprintf(fp, "      \"is_parallel\": %s,\n", r->is_parallel ? "true" : "false");
        fprintf(fp, "      \"min_ms\": %.4f,\n", r->min_ms);
        fprintf(fp, "      \"median_ms\": %.4f,\n", r->median_ms);
        fprintf(fp, "      \"mean_ms\": %.4f,\n", r->mean_ms);
        fprintf(fp, "      \"std_ms\": %.4f,\n", r->std_ms);
        fprintf(fp, "      \"gflops\": %.2f,\n", r->gflops);
        fprintf(fp, "      \"speedup\": %.2f,\n", r->speedup);
        if (isnan(r->efficiency_pct)) {
            fprintf(fp, "      \"efficiency_pct\": null,\n");
        } else {
            fprintf(fp, "      \"efficiency_pct\": %.2f,\n", r->efficiency_pct);
        }
        fprintf(fp, "      \"verified\": %s,\n", r->verified ? "true" : "false");
        fprintf(fp, "      \"max_error\": %.2e\n", r->max_error);
        fprintf(fp, "    }%s\n", (i < mc->num_results - 1) ? "," : "");
    }
    
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    
    fclose(fp);
    printf("JSON exported: %s\n", filepath);
}

void get_timestamp(char* buf, size_t len) {
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    strftime(buf, len, "%Y%m%d_%H%M%S", tm_info);
}
