#include <benchmark/benchmark.h>
#include <unistd.h>
#include <cstdint>

#ifndef __ARM_FEATURE_SVE
#define __ARM_FREATURE_SVE
#endif

#include <arm_sve.h>

uint64_t someCoolFunc(){
    uint64_t n = 0;
    for (int i = 0; i < 100; i++) {
        n += getpid();
    }
    return n;
}

static void bm_func(benchmark::State &state) {
    for ([[maybe_unused]] auto _ : state) {
        someCoolFunc();
    }
}

BENCHMARK(bm_func)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
