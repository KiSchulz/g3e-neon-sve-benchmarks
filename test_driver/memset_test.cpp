#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void *(*)(void *, int, std::size_t);

class MemsetTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void nullptrTest(Func f) { ASSERT_EQ(ref::memset(nullptr, 0, 0), f(nullptr, 0, 0)); }

  void randomDataTest(Func f, std::size_t max_size) {
    char *arr = new char[max_size];
    for (std::size_t i = 0; i < max_size; i++) {
      const int defaultValue = getRandomInt<int>(0, 255);
      const int ch = getRandomInt<int>(0, 255);
      ref::memset(arr, defaultValue, max_size);

      f(arr, ch, i);
      ASSERT_TRUE(std::all_of(arr, arr + i, [&](char el) { return el == ch; }));
      ASSERT_TRUE(std::all_of(arr + i, arr + max_size, [&](char el) { return el == defaultValue; }));
    }
    delete[] arr;
  }
};

TEST_P(MemsetTest, Nullptr) { nullptrTest(GetParam()); }
TEST_P(MemsetTest, randomData) { randomDataTest(GetParam(), 1 << 12); }

INSTANTIATE_TEST_SUITE_P(Kernels, MemsetTest, testing::Values(&neon::memset, &sve::memset),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
