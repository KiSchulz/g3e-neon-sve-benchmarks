#ifndef NEON_SVE_BENCH_TEST_COMMON_H
#define NEON_SVE_BENCH_TEST_COMMON_H

#include <gtest/gtest.h>

#include "common/constants.h"
#include "kernels/neon/neon_kernels.h"
#include "kernels/reference/reference_kernels.h"
#include "kernels/sve/sve_kernels.h"

namespace ref = reference_kernels;
namespace neon = neon_kernels;
namespace sve = sve_kernels;

#endif // NEON_SVE_BENCH_TEST_COMMON_H
