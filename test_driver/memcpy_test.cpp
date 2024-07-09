#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void *(*)(void *, const void *, std::size_t);

class MemcpyTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void nullptrTest(Func f) { ASSERT_EQ(ref::memcpy(nullptr, nullptr, 0), f(nullptr, nullptr, 0)); }

  void randomDataTest(Func f, std::size_t max_size) {
    char *src = new char[max_size];
    char *dest = new char[max_size];

    for (std::size_t i = 0; i < max_size; i++) {
      initArrWithRand(src, i);
      ref::memset(dest, 0, max_size);

      f(dest, src, i);

      ASSERT_EQ(ref::memcmp(src, dest, i), 0);
      ASSERT_TRUE(std::all_of(dest + i, dest + max_size, [&](char el) { return el == 0; }));
    }

    delete[] src;
    delete[] dest;
  }
};

TEST_P(MemcpyTest, Nullptr) { nullptrTest(GetParam()); }
TEST_P(MemcpyTest, randomData) { randomDataTest(GetParam(), 1 << 12); }

INSTANTIATE_TEST_SUITE_P(Kernels, MemcpyTest, testing::Values(&neon::memcpy, &sve::memcpy<>),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });

class MemcpyTest_SVEVersions : public MemcpyTest {
public:
  void test_nullptr_random(Func f) {
    nullptrTest(f);
    randomDataTest(f, 1 << 12);
  }
};

TEST_P(MemcpyTest_SVEVersions, test_nullptr_random) { test_nullptr_random(GetParam()); }
INSTANTIATE_TEST_SUITE_P(Kernels, MemcpyTest_SVEVersions,
                         testing::Values(&sve::memcpy<0>, &sve::memcpy<1>, &sve::memcpy<2>, &sve::memcpy<3>),
                         [](const auto &paramInfo) { return std::to_string(paramInfo.index); });
