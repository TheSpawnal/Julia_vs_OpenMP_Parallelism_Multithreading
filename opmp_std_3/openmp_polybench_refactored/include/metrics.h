#ifndef METRICS_H
#define METRICS_H

#include "benchmark_common.h"

// Metrics collector - manages benchmark results
typedef struct {
    char benchmark_name[64];
    char dataset[32];
    int threads;
    BenchmarkResult results[32];  // Max 32 strategies
    int num_results;
    double sequential_time_ms;  // Baseline for speedup calculation
} MetricsCollector;

// Initialize metrics collector
void metrics_init(MetricsCollector* mc, const char* benchmark, const char* dataset, int threads);

// Timing functions
void timing_init(TimingData* td);
void timing_record(TimingData* td, double time_ms);

// Statistical calculations
double timing_min(const TimingData* td);
double timing_max(const TimingData* td);
double timing_median(const TimingData* td);
double timing_mean(const TimingData* td);
double timing_std(const TimingData* td);

// Compute metrics
double compute_speedup(double baseline_ms, double current_ms);
double compute_efficiency(const char* strategy, double speedup, int threads);
double compute_gflops(double flops, double time_ms);

// Record a benchmark result
void metrics_record(MetricsCollector* mc, 
                   const char* strategy,
                   const TimingData* timing,
                   double flops,
                   int verified,
                   double max_error);

// Export functions (Julia-compatible CSV format)
void metrics_export_csv(const MetricsCollector* mc, const char* filepath);
void metrics_export_json(const MetricsCollector* mc, const char* filepath);

// Print summary to console
void metrics_print_summary(const MetricsCollector* mc);
void metrics_print_header(void);
void metrics_print_result(const BenchmarkResult* r);

// Get timestamp string for filenames
void get_timestamp(char* buf, size_t len);

#endif // METRICS_H
