#include "test_common.h"

#include <algorithm>
#include <memory>
#include <random>

class MemcmpTest : public testing::Test {
protected:
  std::mt19937_64 engine{0};
  std::uniform_int_distribution<char> dist{0, 255};

  std::shared_ptr<char> getRandomKey(std::size_t length) {
    std::shared_ptr<char> key{new char[length]};

    std::generate(key.get(), key.get() + length, [this]() { return dist(engine); });

    return key;
  }

  static bool memcmpResultEQ(int a, int b) { return (a < 0 && b < 0) || (a > 0 && b > 0) || (a == 0 && b == 0); }

  template <class Func> void simpleEQTest(Func f) {
    const std::size_t len = 64;
    const auto key = getRandomKey(len);
    ASSERT_EQ(f(key.get(), key.get(), len), 0);
  }

  template <class Func> void variableLenRandomKeyTest(Func f, std::size_t min_len, std::size_t max_len) {
    for (std::size_t len = 1; len < max_len; len++) {
      const auto a = getRandomKey(len);
      const auto b = getRandomKey(len);

      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(a.get(), b.get(), len), neon::memcmp(a.get(), b.get(), len)));
      ASSERT_TRUE(memcmpResultEQ(neon::memcmp(a.get(), a.get(), len), 0));
      ASSERT_TRUE(memcmpResultEQ(neon::memcmp(b.get(), b.get(), len), 0));
      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(b.get(), a.get(), len), neon::memcmp(b.get(), a.get(), len)));
    }
  }
};

TEST(memcmp_test, Neon_nullptr) { ASSERT_EQ(ref::memcmp(nullptr, nullptr, 0), neon::memcmp(nullptr, nullptr, 0)); }

TEST(memcmp_test, SVE_nullptr) { ASSERT_EQ(ref::memcmp(nullptr, nullptr, 0), sve::memcmp(nullptr, nullptr, 0)); }

TEST_F(MemcmpTest, Neon_simpleEQTest) { simpleEQTest(neon::memcmp); }

TEST_F(MemcmpTest, Neon_shortKeyTest) { variableLenRandomKeyTest(neon::memcmp, 1, 128); }

TEST_F(MemcmpTest, Neon_longKeyTest) { variableLenRandomKeyTest(neon::memcmp, 1 << 8, 1 << 12); }
