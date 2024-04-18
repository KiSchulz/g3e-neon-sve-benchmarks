#include "test_common.h"

#include <unistd.h>

TEST(hello_test, Neon) {
  EXPECT_EQ(getpid(), neon_kernels::helloNeon());
}

TEST(hello_test, SVE) {
  EXPECT_EQ(getpid(), sve_kernels::helloSVE());
}
