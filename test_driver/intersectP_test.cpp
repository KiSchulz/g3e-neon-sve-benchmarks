#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void (*)(const Bounds3f *, const Vec3f *, const float *, const Vec3f *, const int *, bool *);

class IntersectPTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
};

INSTANTIATE_TEST_SUITE_P(Kernels, IntersectPTest,
                         testing::Values(&ref::intersectP, &neon::intersectP, &sve::intersectP),
                         [](const auto &paramInfo) {
                           switch (paramInfo.index) {
                           case 0:
                             return "Ref";
                           case 1:
                             return "Neon";
                           case 2:
                             return "SVE";
                           default:
                             throw std::runtime_error("invalid argument was supplied");
                           }
                         });
