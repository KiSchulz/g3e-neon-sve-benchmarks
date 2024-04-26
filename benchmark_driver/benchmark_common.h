#ifndef NEON_SVE_BENCH_BENCHMARK_COMMON_H
#define NEON_SVE_BENCH_BENCHMARK_COMMON_H

#include <benchmark/benchmark.h>
#include <kernels/reference/reference_kernels.h>
#include <kernels/neon/neon_kernels.h>
#include <kernels/sve/sve_kernels.h>

namespace ref = reference_kernels;
namespace neon = neon_kernels;
namespace sve = sve_kernels;

#endif // NEON_SVE_BENCH_BENCHMARK_COMMON_H
