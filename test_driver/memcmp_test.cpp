#include "test_common.h"
#include "common/random_data_generator.h"

#include <memory>

using Func = int (*)(const void *, const void *, std::size_t);

class MemcmpTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static bool memcmpResultEQ(int a, int b) { return (a < 0 && b < 0) || (a > 0 && b > 0) || (a == 0 && b == 0); }

  static void nullptrTest(Func f) { ASSERT_EQ(ref::memcmp(nullptr, nullptr, 0), f(nullptr, nullptr, 0)); }

  void simpleEQTest(Func f) {
    const std::size_t len = 64;
    const auto key = getRandArr(len);
    ASSERT_EQ(f(key.get(), key.get(), len), 0);
  }

  void variableLenRandomKeyTest(Func f, std::size_t min_len, std::size_t max_len) {
    for (std::size_t len = min_len; len < max_len; len++) {
      const auto a = getRandArr(len);
      const auto b = getRandArr(len);

      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(a.get(), b.get(), len), f(a.get(), b.get(), len))) << len;
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), a.get(), len), 0)) << len;
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), b.get(), len), 0)) << len;
      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(b.get(), a.get(), len), f(b.get(), a.get(), len))) << len;
    }
  }

  void variablePrefixLenKeyTest(Func f, std::size_t key_len) {
    for (std::size_t prefix_len = 1; prefix_len <= key_len; prefix_len++) {
      const auto prefix = getRandArr(prefix_len);
      const auto a = getRandArr(key_len, prefix.get(), prefix_len);
      const auto b = getRandArr(key_len, prefix.get(), prefix_len);

      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(a.get(), b.get(), key_len), f(a.get(), b.get(), key_len)));
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), a.get(), key_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), b.get(), prefix_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), a.get(), prefix_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), b.get(), key_len), 0));
      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(b.get(), a.get(), key_len), f(b.get(), a.get(), key_len)));
    }
  }
};

TEST_P(MemcmpTest, Nullptr) { nullptrTest(GetParam()); }
TEST_P(MemcmpTest, SimpleEQ) { simpleEQTest(GetParam()); }
TEST_P(MemcmpTest, ShortKeys) { variableLenRandomKeyTest(GetParam(), 1, 128); }
TEST_P(MemcmpTest, LongKeys) { variableLenRandomKeyTest(GetParam(), 1 << 8, 1 << 12); }
TEST_P(MemcmpTest, VariablePrefixLengthKeys) { variablePrefixLenKeyTest(GetParam(), 1 << 10); }

INSTANTIATE_TEST_SUITE_P(Kernels, MemcmpTest, testing::Values(&neon::memcmp, &sve::memcmp),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
