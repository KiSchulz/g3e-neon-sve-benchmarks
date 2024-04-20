#include "test_common.h"

#include <memory>
#include <random>
#include <algorithm>

class MemcmpTest : public testing::Test {
protected:
  std::mt19937_64 engine{0};
  std::uniform_int_distribution<char> dist{0, 255};

  std::shared_ptr<char> getRandomKey(std::size_t length) {
    std::shared_ptr<char> key{new char[length]};

    std::generate(key.get(), key.get() + length, [this]() { return dist(engine); });

    return key;
  }
};

TEST(memcmp_test, Neon_nullptr) {
  EXPECT_EQ(reference_kernels::memcmp(nullptr, nullptr, 0), neon_kernels::memcmp(nullptr, nullptr, 0));
}

TEST(memcmp_test, SVE_nullptr) {
  EXPECT_EQ(reference_kernels::memcmp(nullptr, nullptr, 0), sve_kernels::memcmp(nullptr, nullptr, 0));
}

TEST_F(MemcmpTest, Neon_shortKeyTest) {
  const std::size_t len = 8;
  const auto lhs = getRandomKey(len);
  const auto rhs = getRandomKey(len);
  EXPECT_EQ(reference_kernels::memcmp(lhs.get(), rhs.get(), len), neon_kernels::memcmp(lhs.get(), rhs.get(), len));
}