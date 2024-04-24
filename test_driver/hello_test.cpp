#include "test_common.h"

TEST(hello_test, Neon) {
  EXPECT_EQ(ref::helloReference(), neon::helloNeon());
}

TEST(hello_test, SVE) {
  EXPECT_EQ(ref::helloReference(), sve::helloSVE());
}
