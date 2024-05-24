#include "common/random_data_generator.h"
#include "test_common.h"

using Func = uint64_t (*)(const uint64_t *, std::size_t);

class AccumulateTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void nullptrTest(Func f) { ASSERT_EQ(ref::accumulate(nullptr, 0), f(nullptr, 0)); }

  static void gaussSumTest(Func f, std::size_t max_size) {
    auto *arr = new uint64_t[max_size];
    std::iota(arr, arr + max_size, 1);
    for (std::size_t i = 0; i < max_size; i++) {
      ASSERT_EQ((i * (i + 1)) / 2, f(arr, i)) << i;
    }
    delete[] arr;
  }

  void randomDataTest(Func f, std::size_t max_size) {
    for (std::size_t i = 1; i < max_size; i++) {
      const auto arr = getRandArr(i * sizeof(uint64_t));
      ASSERT_EQ(ref::accumulate((uint64_t *)arr.get(), i), f((uint64_t *)arr.get(), i));
    }
  }
};

TEST_P(AccumulateTest, Nullptr) { nullptrTest(GetParam()); }
TEST_P(AccumulateTest, gausSum) { gaussSumTest(GetParam(), 1 << 12); }
TEST_P(AccumulateTest, randomData) { randomDataTest(GetParam(), 1 << 12); }

INSTANTIATE_TEST_SUITE_P(Kernels, AccumulateTest, testing::Values(&neon::accumulate, &sve::accumulate),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });