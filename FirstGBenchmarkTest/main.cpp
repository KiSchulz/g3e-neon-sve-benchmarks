#include <benchmark/benchmark.h>
#include <unistd.h>
#include <stdint.h>

uint64_t someCoolFunc(){
    uint64_t n = 0;
    for (int i = 0; i < 100; i++) {
        n += getpid();
    }
    return n;
}

static void bm_func(benchmark::State &state) {
    for (auto _ : state) {
        someCoolFunc();
    }
}

BENCHMARK(bm_func)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
