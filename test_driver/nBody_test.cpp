#include "test_common.h"

#include "common/random_data_generator.h"

using Func = void (*)(double *, double *, double *, double *, double *, double *, const double*, double, std::size_t);

class NBodyTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
};

INSTANTIATE_TEST_SUITE_P(Kernels, NBodyTest, testing::Values(&neon::nBody_step, &sve::nBody_step),
                         [](const auto &paramInfo) { return paramInfo.index == 0 ? "Neon" : "SVE"; });
