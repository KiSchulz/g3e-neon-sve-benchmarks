#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void (*)(const uint8_t *keys, const std::size_t *len, uint32_t seed, uint32_t *out);

class Murmur3_32Test : public testing::TestWithParam<std::tuple<Func, std::size_t>>, public RandomDataGenerator {
public:
  static void nullptrTest(Func f) {}
  //TODO
};

TEST_P(Murmur3_32Test, nullptrTest) { nullptrTest(std::get<0>(GetParam())); }

INSTANTIATE_TEST_SUITE_P(Kernels, Murmur3_32Test, testing::Values(std::make_tuple(&neon::murmur3_32, neon::murmur3_32Width()),
                                                                  std::make_tuple(&sve::murmur3_32, sve::murmur3_32Width())),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
