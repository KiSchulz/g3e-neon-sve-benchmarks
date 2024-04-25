#include "test_common.h"

#include <algorithm>
#include <memory>
#include <random>

class MemcmpTest : public testing::Test {
protected:
  std::mt19937_64 engine{0};
  std::uniform_int_distribution<char> dist{0, 255};

  std::shared_ptr<char> getRandomKey(std::size_t length, const char *prefix = nullptr, std::size_t prefix_len = 0) {
    std::shared_ptr<char> key{new char[length]};

    if (prefix != nullptr && prefix_len > 0) {
      assert(prefix_len <= length);
      std::copy(prefix, prefix + prefix_len, key.get());
    }
    std::generate(key.get() + prefix_len, key.get() + length, [this]() { return dist(engine); });

    return key;
  }

  static bool memcmpResultEQ(int a, int b) { return (a < 0 && b < 0) || (a > 0 && b > 0) || (a == 0 && b == 0); }

  template <class Func> void nullptrTest(Func f) {
    ASSERT_EQ(ref::memcmp(nullptr, nullptr, 0), f(nullptr, nullptr, 0));
  }

  template <class Func> void simpleEQTest(Func f) {
    const std::size_t len = 64;
    const auto key = getRandomKey(len);
    ASSERT_EQ(f(key.get(), key.get(), len), 0);
  }

  template <class Func> void variableLenRandomKeyTest(Func f, std::size_t min_len, std::size_t max_len) {
    for (std::size_t len = min_len; len < max_len; len++) {
      const auto a = getRandomKey(len);
      const auto b = getRandomKey(len);

      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(a.get(), b.get(), len), f(a.get(), b.get(), len))) << len;
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), a.get(), len), 0)) << len;
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), b.get(), len), 0)) << len;
      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(b.get(), a.get(), len), f(b.get(), a.get(), len))) << len;
    }
  }

  template<class Func>
  void variablePrefixLenKeyTest(Func f, std::size_t key_len) {
    for (std::size_t prefix_len = 1; prefix_len <= key_len; prefix_len++) {
      const auto prefix = getRandomKey(prefix_len);
      const auto a = getRandomKey(key_len, prefix.get(), prefix_len);
      const auto b = getRandomKey(key_len, prefix.get(), prefix_len);

      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(a.get(), b.get(), key_len), f(a.get(), b.get(), key_len)));
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), a.get(), key_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(a.get(), b.get(), prefix_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), a.get(), prefix_len), 0));
      ASSERT_TRUE(memcmpResultEQ(f(b.get(), b.get(), key_len), 0));
      ASSERT_TRUE(memcmpResultEQ(ref::memcmp(b.get(), a.get(), key_len), f(b.get(), a.get(), key_len)));
    }
  }
};

TEST_F(MemcmpTest, Neon_nullptrTest) { nullptrTest(neon::memcmp); }
TEST_F(MemcmpTest, SVE_nullptrTetst) { nullptrTest(sve::memcmp); }

TEST_F(MemcmpTest, Neon_simpleEQTest) { simpleEQTest(neon::memcmp); }
TEST_F(MemcmpTest, SVE_simpleEQTest) { simpleEQTest(sve::memcmp); }

TEST_F(MemcmpTest, Neon_shortKeyTest) { variableLenRandomKeyTest(neon::memcmp, 1, 128); }
TEST_F(MemcmpTest, SVE_shortKeyTest) { variableLenRandomKeyTest(sve::memcmp, 1, 128); }

TEST_F(MemcmpTest, Neon_longKeyTest) { variableLenRandomKeyTest(neon::memcmp, 1 << 8, 1 << 12); }
TEST_F(MemcmpTest, SVE_longKeyTest) { variableLenRandomKeyTest(sve::memcmp, 1 << 8, 1 << 12); }

TEST_F(MemcmpTest, Neon_variablePrefixLenTest) { variablePrefixLenKeyTest(neon::memcmp, 1 << 10); }
TEST_F(MemcmpTest, SVE_variablePrefixLenTest) { variablePrefixLenKeyTest(sve::memcmp, 1 << 10); }
