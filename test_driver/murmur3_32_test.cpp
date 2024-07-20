#include "common/random_data_generator.h"
#include "test_common.h"

using Func = uint32_t (*)(const uint8_t *, size_t, uint32_t);

class Murmur3_32Test : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void doesNotCrash(Func f) {
    std::size_t len = 0;
    uint32_t seed = 0;

    f(nullptr, len, seed);
  }

  void randomFixedSizeInput(Func f, std::size_t key_len = 16) {
    auto *key = (uint8_t *)alloca(key_len * sizeof(uint8_t));
    uint32_t seed = 0;

    initArrWithRand(key, key_len * sizeof(uint8_t));
    ASSERT_EQ(ref::murmur3_32(key, key_len, seed), f(key, key_len, seed));
  }

  void randomVariableSizeInput(Func f, std::size_t max_key_len) {
    for (std::size_t i = 0; i < max_key_len; i++) {
      randomFixedSizeInput(f, i);
    }
  }
};

TEST_P(Murmur3_32Test, doesNotCrash) { doesNotCrash(GetParam()); }
TEST_P(Murmur3_32Test, randomFiexedSizeInput) { randomFixedSizeInput(GetParam()); }
TEST_P(Murmur3_32Test, randomLargeInput) { randomVariableSizeInput(GetParam(), 1 << 13); }

INSTANTIATE_TEST_SUITE_P(Kernels, Murmur3_32Test, testing::Values(&neon::murmur3_32, &sve::murmur3_32),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
