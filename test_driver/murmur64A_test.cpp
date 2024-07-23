#include "common/random_data_generator.h"
#include "test_common.h"

using Func = uint64_t (*)(const uint8_t *, size_t, uint64_t);

class Murmur64ATest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void doesNotCrash(Func f) {
    std::size_t len = 0;
    uint64_t seed = 0;

    f(nullptr, len, seed);
  }

  void randomFixedSizeInput(Func f, std::size_t key_len = 16) {
    auto *key = (uint8_t *)alloca(key_len * sizeof(uint8_t));
    uint64_t seed = 0;

    initArrWithRand(key, key_len * sizeof(uint8_t));
    ASSERT_EQ(ref::murmur64A(key, key_len, seed), f(key, key_len, seed));
  }

  void randomVariableSizeInput(Func f, std::size_t max_key_len) {
    for (std::size_t i = 0; i < max_key_len; i++) {
      randomFixedSizeInput(f, i);
    }
  }
};

TEST_P(Murmur64ATest, doesNotCrash) { doesNotCrash(GetParam()); }
TEST_P(Murmur64ATest, randomFiexedSizeInput) { randomFixedSizeInput(GetParam()); }
TEST_P(Murmur64ATest, randomLargeInput) { randomVariableSizeInput(GetParam(), 1 << 13); }

INSTANTIATE_TEST_SUITE_P(Kernels, Murmur64ATest, testing::Values(&neon::murmur64A, &sve::murmur64A),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
