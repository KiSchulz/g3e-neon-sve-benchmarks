#include "test_common.h"

TEST(hello_test, Neon) {
  EXPECT_EQ(reference_kernels::helloReference(), neon_kernels::helloNeon());
}

TEST(hello_test, SVE) {
  EXPECT_EQ(reference_kernels::helloReference(), sve_kernels::helloSVE());
}
