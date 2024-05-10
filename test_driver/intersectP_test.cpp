#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void (*)(const Bounds3f *, const Vec3f *, const float *, const Vec3f *, const int *, bool *);

class IntersectPTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  static void doesHit(Func f) {
    Bounds3f box{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    Vec3f o{-1, 0, 0};
    float tMax = INFINITY_F;
    Vec3f invDir = Vec3f{1, 0, 0}.invertElements();
    int dirIsNeg[3] = {0};
    bool result = false;

    f(&box, &o, &tMax, &invDir, dirIsNeg, &result);
    ASSERT_TRUE(result);
  }

  static void doesNotHit(Func f) {}

  static void negativeDir(Func f) {}

  static void emptyAABB(Func f) {}

  static void smallTMax(Func f) {}

  static void hitsAllRandomAABB(Func f) {}

  static void compareAgainstRef(Func f) {}
};

TEST_P(IntersectPTest, doesHit) { doesHit(GetParam()); }

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
